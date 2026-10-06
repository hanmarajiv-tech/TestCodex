from pathlib import Path

from llama_index.core import VectorStoreIndex

from .interfaces import IDocumentSchema, ILoader, IPipeline, IReranker, IRetriever
from .models import SearchHit


class RAGPipeline(IPipeline):
    def __init__(self, index: VectorStoreIndex, loader: ILoader,
                 document_schema: IDocumentSchema, retriever: IRetriever,
                 reranker: IReranker) -> None:
        self.index = index
        self.loader = loader
        self.document_schema = document_schema
        self.retriever = retriever
        self.reranker = reranker

    def ingest(self, file_path: str) -> int:
        sources = self.loader.load(Path(file_path))
        documents = [self.document_schema.to_document(source) for source in sources]
        for document in documents:
            # Deletion uses Qdrant's persisted ref_doc_id, even after worker restart.
            store = self.index.vector_store
            if store.client.collection_exists(store.collection_name):
                store.delete(document.doc_id)
            self.index.insert(document)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        nodes = self.retriever.retrieve(query, top_k)
        ranked = self.reranker.rerank(query, nodes)
        return [SearchHit(node_id=node.node.node_id, text=node.node.get_content(),
                          score=node.score, metadata=node.node.metadata)
                for node in ranked[:top_k]]
