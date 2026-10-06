import logging
import os
import signal
import socket
from pathlib import Path
from threading import Event

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from redis import Redis

from .app import App
from .components import DefaultDocumentSchema, IdentityReranker, LlamaIndexRetriever, TextLoader
from .pipeline import RAGPipeline
from .queue import RedisQueue


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                           decode_responses=True)
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY"))
    stop_event = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    try:
        redis.ping()
        qdrant.get_collections()
        embedding = FastEmbedEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            cache_dir=os.getenv("MODEL_CACHE", ".cache/fastembed"))
        store = QdrantVectorStore(client=qdrant,
                                 collection_name=os.getenv("QDRANT_COLLECTION", "documents"))
        index = VectorStoreIndex.from_vector_store(
            store, embed_model=embedding,
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)])
        pipeline = RAGPipeline(index, TextLoader(Path(os.getenv("DATA_ROOT", "data"))),
                               DefaultDocumentSchema(), LlamaIndexRetriever(index),
                               IdentityReranker())
        app = App(RedisQueue(redis, consumer=f"{socket.gethostname()}-{os.getpid()}"), pipeline)
        app.run(stop_event)
    finally:
        qdrant.close()
        redis.close()
