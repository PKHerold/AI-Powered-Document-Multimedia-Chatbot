"""Additional tests for routers and database to boost coverage."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from datetime import datetime, timezone
from app.main import app
from tests.conftest import MockCursor


def make_mock_db(doc_data=None, chunks_data=None, history_data=None):
    """Create a mock DB with configurable data."""
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

    if doc_data:
        db.documents.find_one = AsyncMock(return_value=doc_data)
        db.documents.find = MagicMock(return_value=MockCursor([doc_data]))
    if chunks_data:
        db.chunks.find = MagicMock(return_value=MockCursor(chunks_data))
    if history_data:
        db.chat_history.find_one = AsyncMock(return_value=history_data)

    return db


SAMPLE_DOC = {
    "_id": "doc-123",
    "filename": "doc-123.pdf",
    "original_filename": "report.pdf",
    "file_type": "pdf",
    "file_size": 2048,
    "file_path": "uploads/doc-123.pdf",
    "upload_time": datetime.now(timezone.utc),
    "status": "completed",
    "summary": "A comprehensive report.",
    "metadata": {},
    "chunk_count": 5,
    "duration": None,
}


@pytest.mark.asyncio
async def test_list_documents_with_data():
    """Test listing documents with data."""
    mock_db = make_mock_db(doc_data=SAMPLE_DOC)
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/documents")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "doc-123"


@pytest.mark.asyncio
async def test_get_document_found():
    """Test getting an existing document."""
    mock_db = make_mock_db(doc_data=SAMPLE_DOC)
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/documents/doc-123")
            assert response.status_code == 200
            assert response.json()["original_filename"] == "report.pdf"


@pytest.mark.asyncio
async def test_get_summary_existing():
    """Test getting an existing summary."""
    mock_db = make_mock_db(doc_data=SAMPLE_DOC)
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/documents/doc-123/summary")
            assert response.status_code == 200
            assert response.json()["summary"] == "A comprehensive report."


@pytest.mark.asyncio
async def test_get_summary_processing():
    """Test summary for processing document."""
    processing_doc = {**SAMPLE_DOC, "status": "processing", "summary": ""}
    mock_db = make_mock_db(doc_data=processing_doc)
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/documents/doc-123/summary")
            assert response.status_code == 200
            assert "processing" in response.json()["status"]


@pytest.mark.asyncio
async def test_delete_document_success():
    """Test deleting an existing document."""
    mock_db = make_mock_db(doc_data=SAMPLE_DOC)
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.routers.documents.delete_file"), \
         patch("app.routers.documents.remove_document_from_index"), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/api/documents/doc-123")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_with_document():
    """Test chat with a completed document."""
    mock_db = make_mock_db(
        doc_data=SAMPLE_DOC,
        chunks_data=[{"text": "AI content", "chunk_index": 0, "start_time": None, "end_time": None, "page_number": 1}],
    )
    with patch("app.routers.chat.get_db", return_value=mock_db), \
         patch("app.routers.chat.search_similar_chunks", return_value=[
             {"text": "AI content", "chunk_index": 0, "start_time": None, "end_time": None, "page_number": 1, "score": 0.9}
         ]), \
         patch("app.routers.chat.answer_question", return_value={
             "answer": "This is about AI.", "timestamps": [], "sources": [{"chunk_index": 0, "text": "AI content"}]
         }), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/chat", json={
                "document_id": "doc-123",
                "question": "What is this about?",
            })
            assert response.status_code == 200
            assert "AI" in response.json()["answer"]


@pytest.mark.asyncio
async def test_chat_document_not_ready():
    """Test chat with a processing document."""
    processing_doc = {**SAMPLE_DOC, "status": "processing"}
    mock_db = make_mock_db(doc_data=processing_doc)
    with patch("app.routers.chat.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/chat", json={
                "document_id": "doc-123",
                "question": "test",
            })
            assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_history_found():
    """Test getting chat history."""
    history = {
        "_id": "hist-1",
        "document_id": "doc-123",
        "messages": [
            {"role": "user", "content": "Hi", "timestamps": [], "created_at": datetime.now(timezone.utc).isoformat()},
        ],
    }
    mock_db = make_mock_db(doc_data=SAMPLE_DOC, history_data=history)
    with patch("app.routers.chat.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/chat/history/doc-123")
            assert response.status_code == 200
            assert len(response.json()["messages"]) == 1


@pytest.mark.asyncio
async def test_clear_chat_history():
    """Test clearing chat history."""
    mock_db = make_mock_db(doc_data=SAMPLE_DOC)
    with patch("app.routers.chat.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/api/chat/history/doc-123")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_stats_endpoint():
    """Test stats endpoint."""
    mock_db = make_mock_db()
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.routers.documents.get_index_stats", return_value={"total_vectors": 0, "total_chunks": 0, "total_documents": 0, "dimension": 768}), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/stats")
            assert response.status_code == 200
            assert "documents" in response.json()


@pytest.mark.asyncio
async def test_upload_invalid_file_type():
    """Test uploading an unsupported file type."""
    with patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/upload", files={
                "file": ("test.exe", b"fake content", "application/octet-stream"),
            })
            assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_valid_pdf():
    """Test uploading a valid PDF."""
    mock_db = make_mock_db()
    with patch("app.routers.upload.get_db", return_value=mock_db), \
         patch("app.routers.upload.media_service.save_upload_file", new_callable=AsyncMock, return_value=("id-1", "uploads/id-1.pdf")), \
         patch("app.routers.upload.process_document", new_callable=AsyncMock), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/upload", files={
                "file": ("report.pdf", b"%PDF-1.4 fake content", "application/pdf"),
            })
            assert response.status_code == 200
            assert response.json()["file_type"] == "pdf"


@pytest.mark.asyncio
async def test_media_not_found():
    """Test serving non-existent media."""
    mock_db = make_mock_db()
    with patch("app.routers.documents.get_db", return_value=mock_db), \
         patch("app.database.connect_to_mongo", new_callable=AsyncMock), \
         patch("app.database.close_mongo_connection", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/media/nonexistent")
            assert response.status_code == 404


class TestDatabaseModule:
    """Additional database tests."""

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        import app.database as db_mod
        with patch("app.database.AsyncIOMotorClient") as MockClient:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_db.documents.create_index = AsyncMock()
            mock_db.chunks.create_index = AsyncMock()
            mock_db.chat_history.create_index = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=mock_db)
            MockClient.return_value = mock_client

            await db_mod.connect_to_mongo()
            assert db_mod.db is not None

            await db_mod.close_mongo_connection()
            mock_client.close.assert_called_once()
