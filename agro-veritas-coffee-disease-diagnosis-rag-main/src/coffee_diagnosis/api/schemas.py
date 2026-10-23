"""Validated request and response models for the HTTP API."""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=2000)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must not be blank")
        return value


class IncidentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    severity: Literal["critical", "high", "medium", "low"]
    description: str = Field(min_length=1, max_length=4000)
    evidence: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "description")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class IncidentRecord(BaseModel):
    id: UUID
    title: str
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    status: Literal["open", "acknowledged", "resolved"]
    evidence: Dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolution: Optional[str] = None


class IncidentResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: str = Field(min_length=1, max_length=2000)

    @field_validator("resolution")
    @classmethod
    def resolution_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("resolution must not be blank")
        return value
