"""Tests for API endpoints (upload, documents, chat)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def mock_all_services():
    """Mock database and external services for API tests."""
    with patch("app.routers.upload.get_db") as mock_db_upload, \
         patch("app.routers.documents.get_db") as mock_db_docs, \
         patch("app.routers.chat.get_db") as mock_db_chat, \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):

        from tests.conftest import MockCursor
        from datetime import datetime, timezone

        # Setup mock DB for each router
        for mock_db in [mock_db_upload, mock_db_docs, mock_db_chat]:
            db = MagicMock()
            for coll_name in ["documents", "chunks", "chat_history"]:
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
                setattr(db, coll_name, coll)
            mock_db.return_value = db

        yield {
            "upload_db": mock_db_upload,
            "docs_db": mock_db_docs,
            "chat_db": mock_db_chat,
        }


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root health check."""
    with patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check."""
    with patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_list_documents_empty(mock_all_services):
    """Test listing documents when empty."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.asyncio
async def test_get_document_not_found(mock_all_services):
    """Test getting a non-existent document."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/documents/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_not_found(mock_all_services):
    """Test deleting a non-existent document."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_no_document(mock_all_services):
    """Test chat with non-existent document."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat", json={
            "document_id": "nonexistent",
            "question": "test question",
        })
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_history_not_found(mock_all_services):
    """Test getting chat history for non-existent document."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/history/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_no_file():
    """Test upload without file."""
    with patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/upload")
            assert response.status_code == 422


class TestModels:
    """Test Pydantic models."""

    def test_upload_response(self):
        from app.models import UploadResponse, FileType, ProcessingStatus
        resp = UploadResponse(
            document_id="123", filename="test.pdf",
            file_type=FileType.PDF, status=ProcessingStatus.PROCESSING,
            message="OK"
        )
        assert resp.document_id == "123"

    def test_chat_request(self):
        from app.models import ChatRequest
        req = ChatRequest(document_id="123", question="What?")
        assert req.question == "What?"

    def test_chat_response(self):
        from app.models import ChatResponse
        resp = ChatResponse(answer="42", timestamps=[], sources=[])
        assert resp.answer == "42"

    def test_document_model(self):
        from app.models import DocumentModel, FileType
        doc = DocumentModel(filename="test.pdf", original_filename="test.pdf",
                           file_type=FileType.PDF, file_size=1024)
        assert doc.file_type == FileType.PDF

    def test_file_type_enum(self):
        from app.models import FileType
        assert FileType.PDF.value == "pdf"
        assert FileType.AUDIO.value == "audio"
        assert FileType.VIDEO.value == "video"

    def test_processing_status_enum(self):
        from app.models import ProcessingStatus
        assert ProcessingStatus.COMPLETED.value == "completed"


class TestConfig:
    """Test configuration."""

    def test_settings_loaded(self):
        from app.config import settings
        assert settings.GROQ_API_KEY is not None
        assert settings.DB_NAME == "docqa_test"


    def test_upload_path(self):
        from app.config import settings
        path = settings.get_upload_path()
        assert path.exists()


class TestDatabase:
    """Test database module."""

    def test_get_db_raises_without_init(self):
        from app.database import get_db
        import app.database as db_mod
        original = db_mod.db
        db_mod.db = None
        with pytest.raises(RuntimeError):
            get_db()
        db_mod.db = original
