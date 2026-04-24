"""Pydantic models for request/response schemas and MongoDB documents."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── MongoDB Document Schemas ───────────────────────────────────────────

class DocumentModel(BaseModel):
    """Represents an uploaded document in MongoDB."""
    id: str = Field(default="", alias="_id")
    filename: str
    original_filename: str
    file_type: FileType
    file_size: int  # bytes
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = ProcessingStatus.PENDING
    summary: str = ""
    metadata: dict = Field(default_factory=dict)
    chunk_count: int = 0
    duration: float | None = None  # seconds, for audio/video

    class Config:
        populate_by_name = True


class TextChunk(BaseModel):
    """A chunk of extracted text with optional timestamps."""
    id: str = Field(default="", alias="_id")
    document_id: str
    text: str
    chunk_index: int
    page_number: int | None = None  # for PDFs
    start_time: float | None = None  # seconds, for audio/video
    end_time: float | None = None  # seconds, for audio/video
    embedding_index: int | None = None  # index in FAISS


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamps: list[dict] = Field(default_factory=list)  # [{start, end, text}]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatHistory(BaseModel):
    """Chat history for a document."""
    id: str = Field(default="", alias="_id")
    document_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── API Request/Response Models ────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after uploading a file."""
    document_id: str
    filename: str
    file_type: FileType
    status: ProcessingStatus
    message: str


class ChatRequest(BaseModel):
    """Chat request from user."""
    document_id: str
    question: str


class ChatResponse(BaseModel):
    """Chat response from AI."""
    answer: str
    timestamps: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)  # relevant chunks


class SummaryResponse(BaseModel):
    """Summary response."""
    document_id: str
    summary: str
    status: str


class DocumentResponse(BaseModel):
    """Document info response."""
    id: str
    filename: str
    original_filename: str
    file_type: FileType
    file_size: int
    upload_time: datetime
    status: ProcessingStatus
    summary: str
    chunk_count: int
    duration: float | None = None
