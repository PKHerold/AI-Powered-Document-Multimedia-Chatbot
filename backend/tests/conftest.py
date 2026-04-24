"""Shared test fixtures and configuration."""

import os
import sys
import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment variables before importing app modules
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["GROQ_API_KEY"] = "test_groq_key"
os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
os.environ["DB_NAME"] = "docqa_test"
os.environ["UPLOAD_DIR"] = "test_uploads"
os.environ["FFMPEG_PATH"] = "ffmpeg"

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import get_db


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf(temp_dir):
    """Create a minimal PDF for testing."""
    # Create a simple PDF-like file
    pdf_path = temp_dir / "test.pdf"
    # Minimal PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test document content) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000360 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_audio(temp_dir):
    """Create a minimal audio file for testing."""
    audio_path = temp_dir / "test.mp3"
    audio_path.write_bytes(b"\xff\xfb\x90\x00" * 100)  # Minimal MP3 frames
    return audio_path


@pytest.fixture
def mock_db():
    """Create a mock MongoDB database."""
    db = MagicMock()

    # Mock collections with async methods
    for collection in ["documents", "chunks", "chat_history"]:
        coll = MagicMock()
        coll.insert_one = AsyncMock()
        coll.insert_many = AsyncMock()
        coll.find_one = AsyncMock(return_value=None)
        coll.find = MagicMock(return_value=MockCursor([]))
        coll.update_one = AsyncMock()
        coll.delete_one = AsyncMock()
        coll.delete_many = AsyncMock()
        coll.count_documents = AsyncMock(return_value=0)
        coll.create_index = AsyncMock()
        setattr(db, collection, coll)

    return db


class MockCursor:
    """Mock async cursor for MongoDB find()."""

    def __init__(self, items):
        self._items = items
        self._index = 0

    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args):
        return self

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    response = MagicMock()
    response.text = "This is a test AI response about the document."
    return response


@pytest.fixture
def mock_transcription_response():
    """Mock Groq Whisper transcription response."""
    return {
        "text": "This is a test transcription of the audio file.",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "This is a test"},
            {"start": 5.0, "end": 10.0, "text": "transcription of the audio file."},
        ],
        "duration": 10.0,
    }


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "text": "Machine learning is a subset of artificial intelligence.",
            "chunk_index": 0,
            "page_number": 1,
            "start_time": None,
            "end_time": None,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "text": "Neural networks are inspired by the human brain.",
            "chunk_index": 1,
            "page_number": 1,
            "start_time": None,
            "end_time": None,
        },
    ]


@pytest.fixture
def sample_document():
    """Sample document record."""
    from datetime import datetime, timezone
    return {
        "_id": "doc-test-123",
        "filename": "doc-test-123.pdf",
        "original_filename": "test_document.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "file_path": "uploads/doc-test-123.pdf",
        "upload_time": datetime.now(timezone.utc),
        "status": "completed",
        "summary": "A test document about AI.",
        "metadata": {},
        "chunk_count": 2,
        "duration": None,
    }
