import hashlib
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .interfaces import IDocumentSchema, ILoader, IReranker, IRetriever
from .models import SourceDocument


class TextLoader(ILoader):
    """Load UTF-8 .txt and .md files beneath an explicitly allowed root."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve(strict=True)

    def load(self, path: Path) -> list[SourceDocument]:
        path = (self.data_root / path).resolve(strict=True)
        if not path.is_relative_to(self.data_root):
            raise ValueError("File is outside DATA_ROOT")
        if path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("TextLoader supports only .txt and .md")
        # Same path has the same identity, allowing replacement on re-ingestion.
        document_id = hashlib.sha256(str(path).encode()).hexdigest()
        return [SourceDocument(document_id=document_id, text=path.read_text(encoding="utf-8"),
                               metadata={"file_path": str(path)})]


class DefaultDocumentSchema(IDocumentSchema):
    def to_document(self, source: SourceDocument) -> Document:
        return Document(id_=source.document_id, text=source.text, metadata=source.metadata,
                        excluded_embed_metadata_keys=["file_path"])


class LlamaIndexRetriever(IRetriever):
    def __init__(self, index: VectorStoreIndex) -> None:
        self.index = index

    def retrieve(self, query: str, top_k: int) -> list[NodeWithScore]:
        store = self.index.vector_store
        if not store.client.collection_exists(store.collection_name):
            return []
        return self.index.as_retriever(similarity_top_k=top_k).retrieve(query)


class IdentityReranker(IReranker):
    """Default: retain vector relevance ordering; replace for cross-encoding."""

    def rerank(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        return nodes
