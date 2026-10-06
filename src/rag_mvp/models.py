from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SourceDocument(BaseModel):
    document_id: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    node_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]


class Job(BaseModel):
    job_id: str = Field(min_length=1, max_length=200)
    operation: Literal["ingest", "search"]
    file_path: str | None = None
    query: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def validate_operation(self) -> "Job":
        if self.operation == "ingest" and not self.file_path:
            raise ValueError("ingest requires file_path")
        if self.operation == "search" and (not self.query or not self.query.strip()):
            raise ValueError("search requires a nonempty query")
        return self
