"""
Pydantic Models and Schemas for Smart Revision Generator.

Defines data models for:
- API requests/responses
- Database entities
- Generation output structure
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Generation Output Models (STRICT JSON FORMAT)
# =============================================================================

class Flashcard(BaseModel):
    """A single flashcard with question and answer."""
    question: str
    answer: str


class FAQ(BaseModel):
    """A frequently asked question with answer."""
    question: str
    answer: str


class MockQuestions(BaseModel):
    """Mock exam questions categorized by marks."""
    two_mark: List[str] = Field(default_factory=list, alias="2_mark")
    five_mark: List[str] = Field(default_factory=list, alias="5_mark")
    ten_mark: List[str] = Field(default_factory=list, alias="10_mark")

    class Config:
        populate_by_name = True


class RevisionOutput(BaseModel):
    """
    Complete structured revision output.
    This is the STRICT JSON format for all generations.
    """
    concepts: List[str] = Field(default_factory=list)
    definitions: List[dict] = Field(default_factory=list)  # {"term": "...", "definition": "..."}
    applications: List[str] = Field(default_factory=list)
    flashcards: List[Flashcard] = Field(default_factory=list)
    faqs: List[FAQ] = Field(default_factory=list)
    mock_questions: MockQuestions = Field(default_factory=MockQuestions)
    summary: str = ""


# =============================================================================
# Document Models
# =============================================================================

class DocumentBase(BaseModel):
    """Base document model."""
    filename: str


class DocumentCreate(DocumentBase):
    """Document creation request."""
    pass


class DocumentResponse(DocumentBase):
    """Document response model."""
    id: UUID
    upload_timestamp: datetime
    file_size: Optional[int] = None
    file_type: Optional[str] = None

    class Config:
        from_attributes = True


# =============================================================================
# Chunk Models
# =============================================================================

class ChunkBase(BaseModel):
    """Base chunk model."""
    chunk_text: str
    chunk_index: int


class ChunkCreate(ChunkBase):
    """Chunk creation model."""
    document_id: UUID
    token_count: Optional[int] = None


class ChunkResponse(ChunkBase):
    """Chunk response model."""
    id: UUID
    document_id: UUID
    token_count: Optional[int] = None

    class Config:
        from_attributes = True


# =============================================================================
# Generation Models
# =============================================================================

class GenerationBase(BaseModel):
    """Base generation model."""
    output_json: RevisionOutput


class GenerationCreate(GenerationBase):
    """Generation creation model."""
    document_id: UUID
    compression_ratio: Optional[float] = None
    accuracy_score: Optional[float] = None
    generation_time_seconds: Optional[float] = None


class GenerationResponse(GenerationBase):
    """Generation response model."""
    id: UUID
    document_id: UUID
    compression_ratio: Optional[float] = None
    accuracy_score: Optional[float] = None
    generation_time_seconds: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Feedback Models
# =============================================================================

class FeedbackBase(BaseModel):
    """Base feedback model."""
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None


class FeedbackCreate(FeedbackBase):
    """Feedback creation model."""
    generation_id: UUID


class FeedbackResponse(FeedbackBase):
    """Feedback response model."""
    id: UUID
    generation_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# API Request/Response Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response after document upload."""
    document_id: UUID
    filename: str
    message: str
    chunks_created: int


class GenerateRequest(BaseModel):
    """Request to generate revision content."""
    document_id: UUID
    query: Optional[str] = None  # Optional specific topic focus


class GenerateResponse(BaseModel):
    """Response with generated revision content."""
    generation_id: UUID
    document_id: UUID
    output: RevisionOutput
    metrics: dict


class RetrievalResult(BaseModel):
    """Result from hybrid retrieval."""
    chunk_id: UUID
    chunk_text: str
    dense_score: float
    sparse_score: float
    final_score: float


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for generated content."""
    cosine_similarity: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    compression_ratio: float
