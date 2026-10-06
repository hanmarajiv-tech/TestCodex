from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.schema import NodeWithScore

from .models import SearchHit, SourceDocument


class ILoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[SourceDocument]: ...


class IDocumentSchema(ABC):
    @abstractmethod
    def to_document(self, source: SourceDocument) -> Document: ...


class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[NodeWithScore]: ...


class IReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]: ...


class IPipeline(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> int: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[SearchHit]: ...


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    payload: str


class IQueue(ABC):
    @abstractmethod
    def read(self, block_ms: int) -> QueueMessage | None: ...

    @abstractmethod
    def complete(self, message: QueueMessage, result: dict[str, Any]) -> None: ...
