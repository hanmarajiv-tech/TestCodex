import json
from pathlib import Path

import fakeredis
import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag_mvp import App, RAGPipeline
from rag_mvp.components import DefaultDocumentSchema, IdentityReranker, LlamaIndexRetriever, TextLoader
from rag_mvp.queue import RedisQueue


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    redis = fakeredis.FakeRedis(decode_responses=True)
    qdrant = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=qdrant, collection_name="test")
    index = VectorStoreIndex.from_vector_store(
        store, embed_model=MockEmbedding(embed_dim=8),
        transformations=[SentenceSplitter(chunk_size=64, chunk_overlap=8)])
    pipeline = RAGPipeline(index, TextLoader(tmp_path), DefaultDocumentSchema(),
                           LlamaIndexRetriever(index), IdentityReranker())
    queue = RedisQueue(redis, "test", reclaim_ms=300_000)
    yield App(queue, pipeline), redis, qdrant, tmp_path, store
    qdrant.close()


def submit(app, redis, payload):
    message_id = redis.xadd("rag:jobs", {"payload": json.dumps(payload)})
    assert app.process_next(block_ms=1)
    return json.loads(redis.get(f"rag:jobs:result:{message_id}"))


def test_ingest_search_replace_and_restart(system):
    app, redis, qdrant, root, store = system
    (root / "example.txt").write_text("Qdrant stores searchable documents.")
    result = submit(app, redis, {"job_id": "one", "operation": "ingest",
                                 "file_path": "example.txt"})
    assert result == {"status": "ok", "job_id": "one", "documents_loaded": 1}
    assert qdrant.count("test", exact=True).count == 1
    assert redis.xpending("rag:jobs", "rag:workers")["pending"] == 0
    # Rebuild the index without its old in-memory docstore.
    index = VectorStoreIndex.from_vector_store(store, embed_model=MockEmbedding(embed_dim=8))
    app.pipeline = RAGPipeline(index, TextLoader(root), DefaultDocumentSchema(),
                               LlamaIndexRetriever(index), IdentityReranker())
    (root / "example.txt").write_text("Updated searchable content.")
    assert submit(app, redis, {"job_id": "two", "operation": "ingest",
                               "file_path": "example.txt"})["status"] == "ok"
    assert qdrant.count("test", exact=True).count == 1
    result = submit(app, redis, {"job_id": "three", "operation": "search", "query": "content"})
    assert result["status"] == "ok"
    assert result["hits"][0]["text"] == "Updated searchable content."


@pytest.mark.parametrize("payload", [
    {"job_id": "bad", "operation": "search", "query": " "},
    {"job_id": "bad", "operation": "ingest", "file_path": "missing.txt"},
])
def test_failure_is_reported_and_acknowledged(system, payload):
    app, redis, *_ = system
    assert submit(app, redis, payload)["status"] == "error"
    assert redis.xpending("rag:jobs", "rag:workers")["pending"] == 0


def test_loader_rejects_escape(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (root / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        TextLoader(root).load(Path("link.txt"))


def test_pending_message_recovered_after_crash():
    redis = fakeredis.FakeRedis(decode_responses=True)
    first = RedisQueue(redis, "first")
    message_id = redis.xadd("rag:jobs", {"payload": "{}"})
    assert first.read(1).message_id == message_id
    second = RedisQueue(redis, "second", reclaim_ms=0)
    assert second.read(1).message_id == message_id


def test_custom_reranker_is_used(system):
    app, redis, _, root, _ = system
    (root / "example.txt").write_text("Example content.")
    submit(app, redis, {"job_id": "one", "operation": "ingest", "file_path": "example.txt"})

    class EmptyReranker(IdentityReranker):
        def rerank(self, query, nodes):
            return []

    app.pipeline.reranker = EmptyReranker()
    result = submit(app, redis, {"job_id": "two", "operation": "search", "query": "example"})
    assert result["hits"] == []
