"""Tests to boost coverage to 95%+ across all modules."""

import pytest
import json
import asyncio
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from httpx import AsyncClient, ASGITransport
from app.main import app
from tests.conftest import MockCursor


def make_db(doc=None, chunks=None, history=None):
    db = MagicMock()
    for c in ["documents", "chunks", "chat_history"]:
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
        setattr(db, c, coll)
    if doc:
        db.documents.find_one = AsyncMock(return_value=doc)
        db.documents.find = MagicMock(return_value=MockCursor([doc]))
    if chunks:
        db.chunks.find = MagicMock(return_value=MockCursor(chunks))
    if history:
        db.chat_history.find_one = AsyncMock(return_value=history)
    return db


DOC = {
    "_id": "d1", "filename": "d1.pdf", "original_filename": "test.pdf",
    "file_type": "pdf", "file_size": 1024, "file_path": "uploads/d1.pdf",
    "upload_time": datetime.now(timezone.utc), "status": "completed",
    "summary": "Test summary.", "metadata": {}, "chunk_count": 2, "duration": None,
}

AUDIO_DOC = {
    "_id": "d2", "filename": "d2.mp3", "original_filename": "test.mp3",
    "file_type": "audio", "file_size": 2048, "file_path": "uploads/d2.mp3",
    "upload_time": datetime.now(timezone.utc), "status": "completed",
    "summary": "", "metadata": {"transcription": "hello world"},
    "chunk_count": 1, "duration": 10.0,
}

PATCHES = {
    "connect": patch("app.database.connect_to_mongo", new_callable=AsyncMock),
    "close": patch("app.database.close_mongo_connection", new_callable=AsyncMock),
}


# ── Upload router: process_document branches ─────────────────────────

class TestProcessDocument:
    @pytest.mark.asyncio
    async def test_process_pdf(self):
        from app.routers.upload import process_document
        from app.models import FileType
        db = make_db()
        with patch("app.routers.upload.get_db", return_value=db), \
             patch("app.routers.upload.extract_text_from_pdf", return_value=[
                 {"id": "c1", "text": "hello", "chunk_index": 0, "page_number": 1}
             ]), \
             patch("app.routers.upload.get_full_text_from_pdf", return_value="hello world"), \
             patch("app.routers.upload.summarize_text", return_value="A summary"), \
             patch("app.routers.upload.add_chunks_to_index"):
            await process_document("d1", "uploads/d1.pdf", FileType.PDF, "test.pdf")
            db.documents.update_one.assert_called()
            db.chunks.insert_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_audio(self):
        from app.routers.upload import process_document
        from app.models import FileType
        db = make_db()
        with patch("app.routers.upload.get_db", return_value=db), \
             patch("app.routers.upload.transcribe_audio", return_value={
                 "text": "hello", "segments": [{"start": 0, "end": 5, "text": "hello"}], "duration": 5.0
             }), \
             patch("app.routers.upload.segments_to_chunks", return_value=[
                 {"id": "c1", "text": "hello", "chunk_index": 0, "start_time": 0, "end_time": 5}
             ]), \
             patch("app.routers.upload.summarize_text", return_value="Summary"), \
             patch("app.routers.upload.add_chunks_to_index"):
            await process_document("d2", "uploads/d2.mp3", FileType.AUDIO, "test.mp3")
            assert db.chunks.insert_many.called

    @pytest.mark.asyncio
    async def test_process_video(self):
        from app.routers.upload import process_document
        from app.models import FileType
        db = make_db()
        with patch("app.routers.upload.get_db", return_value=db), \
             patch("app.routers.upload.extract_audio_from_video", return_value="audio.mp3"), \
             patch("app.routers.upload.transcribe_audio", return_value={
                 "text": "video text", "segments": [{"start": 0, "end": 10, "text": "video text"}], "duration": 10.0
             }), \
             patch("app.routers.upload.segments_to_chunks", return_value=[
                 {"id": "c1", "text": "video text", "chunk_index": 0, "start_time": 0, "end_time": 10}
             ]), \
             patch("app.routers.upload.summarize_text", return_value="Video summary"), \
             patch("app.routers.upload.add_chunks_to_index"):
            await process_document("d3", "uploads/d3.mp4", FileType.VIDEO, "test.mp4")
            assert db.chunks.insert_many.called

    @pytest.mark.asyncio
    async def test_process_failure(self):
        from app.routers.upload import process_document
        from app.models import FileType
        db = make_db()
        with patch("app.routers.upload.get_db", return_value=db), \
             patch("app.routers.upload.extract_text_from_pdf", side_effect=Exception("fail")):
            await process_document("d1", "uploads/d1.pdf", FileType.PDF, "test.pdf")
            # Should set status to failed
            call_args = db.documents.update_one.call_args
            assert "failed" in str(call_args)

    @pytest.mark.asyncio
    async def test_process_no_chunks(self):
        from app.routers.upload import process_document
        from app.models import FileType
        db = make_db()
        with patch("app.routers.upload.get_db", return_value=db), \
             patch("app.routers.upload.extract_text_from_pdf", return_value=[]), \
             patch("app.routers.upload.get_full_text_from_pdf", return_value=""), \
             patch("app.routers.upload.summarize_text", return_value="Empty"), \
             patch("app.routers.upload.add_chunks_to_index"):
            await process_document("d1", "uploads/d1.pdf", FileType.PDF, "test.pdf")
            db.chunks.insert_many.assert_not_called()


# ── Upload router: file size check ───────────────────────────────────

class TestUploadValidation:
    @pytest.mark.asyncio
    async def test_upload_too_large(self):
        big_content = b"x" * (51 * 1024 * 1024)  # 51MB
        with PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/upload", files={
                    "file": ("big.pdf", big_content, "application/pdf"),
                })
                assert resp.status_code == 400
                assert "large" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_no_filename(self):
        with PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/upload", files={
                    "file": ("", b"content", "application/pdf"),
                })
                assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_upload_audio_file(self):
        db = make_db()
        with patch("app.routers.upload.get_db", return_value=db), \
             patch("app.routers.upload.media_service.save_upload_file",
                   new_callable=AsyncMock, return_value=("id-2", "uploads/id-2.mp3")), \
             patch("app.routers.upload.process_document", new_callable=AsyncMock), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/upload", files={
                    "file": ("test.mp3", b"\xff\xfb\x90\x00" * 10, "audio/mpeg"),
                })
                assert resp.status_code == 200
                assert resp.json()["file_type"] == "audio"


# ── Chat router: stream and fallback ─────────────────────────────────

class TestChatStream:
    @pytest.mark.asyncio
    async def test_stream_success(self):
        db = make_db(doc=DOC)
        chunks = [{"text": "chunk1", "start_time": 0.0, "end_time": 5.0,
                    "chunk_index": 0, "page_number": None, "score": 0.9}]

        async def mock_stream(*a, **kw):
            yield "Hello "
            yield "world"

        with patch("app.routers.chat.get_db", return_value=db), \
             patch("app.routers.chat.search_similar_chunks", return_value=chunks), \
             patch("app.routers.chat.stream_answer", side_effect=mock_stream), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/chat/stream", params={
                    "document_id": "d1", "question": "What?"
                })
                assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stream_doc_not_found(self):
        db = make_db()
        with patch("app.routers.chat.get_db", return_value=db), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/chat/stream", params={
                    "document_id": "nope", "question": "test"
                })
                assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_doc_not_ready(self):
        db = make_db(doc={**DOC, "status": "processing"})
        with patch("app.routers.chat.get_db", return_value=db), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/chat/stream", params={
                    "document_id": "d1", "question": "test"
                })
                assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_fallback_to_mongo_chunks(self):
        """When FAISS returns nothing, fall back to MongoDB chunks."""
        db = make_db(doc=DOC, chunks=[
            {"text": "from mongo", "chunk_index": 0, "start_time": None,
             "end_time": None, "page_number": 1}
        ])
        with patch("app.routers.chat.get_db", return_value=db), \
             patch("app.routers.chat.search_similar_chunks", return_value=[]), \
             patch("app.routers.chat.answer_question", return_value={
                 "answer": "From mongo.", "timestamps": [], "sources": []
             }), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/chat", json={
                    "document_id": "d1", "question": "test?"
                })
                assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_history_empty(self):
        db = make_db(doc=DOC)
        with patch("app.routers.chat.get_db", return_value=db), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/chat/history/d1")
                assert resp.status_code == 200
                assert resp.json()["messages"] == []

    @pytest.mark.asyncio
    async def test_chat_history_not_found_doc(self):
        db = make_db()
        with patch("app.routers.chat.get_db", return_value=db), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/chat/history/nope")
                assert resp.status_code == 404


# ── Documents router: summary generation, media serving ──────────────

class TestDocumentsSummary:
    @pytest.mark.asyncio
    async def test_generate_summary_for_pdf(self):
        doc = {**DOC, "summary": ""}
        db = make_db(doc=doc)
        with patch("app.routers.documents.get_db", return_value=db), \
             patch("app.routers.documents.get_full_text_from_pdf", return_value="Full text"), \
             patch("app.routers.documents.summarize_text", return_value="New summary"), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/documents/d1/summary")
                assert resp.status_code == 200
                assert resp.json()["summary"] == "New summary"

    @pytest.mark.asyncio
    async def test_generate_summary_for_audio(self):
        doc = {**AUDIO_DOC, "summary": ""}
        db = make_db(doc=doc)
        with patch("app.routers.documents.get_db", return_value=db), \
             patch("app.routers.documents.summarize_text", return_value="Audio summary"), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/documents/d2/summary")
                assert resp.status_code == 200
                assert resp.json()["summary"] == "Audio summary"

    @pytest.mark.asyncio
    async def test_generate_summary_from_chunks(self):
        doc = {**AUDIO_DOC, "summary": "", "metadata": {}}
        db = make_db(doc=doc, chunks=[
            {"text": "chunk text", "chunk_index": 0}
        ])
        with patch("app.routers.documents.get_db", return_value=db), \
             patch("app.routers.documents.summarize_text", return_value="Chunks summary"), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/documents/d2/summary")
                assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_summary_error(self):
        doc = {**DOC, "summary": ""}
        db = make_db(doc=doc)
        with patch("app.routers.documents.get_db", return_value=db), \
             patch("app.routers.documents.get_full_text_from_pdf", side_effect=Exception("fail")), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/documents/d1/summary")
                assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_summary_not_found(self):
        db = make_db()
        with patch("app.routers.documents.get_db", return_value=db), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/documents/nope/summary")
                assert resp.status_code == 404


class TestMediaServing:
    @pytest.mark.asyncio
    async def test_serve_media_file_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\xff\xfb\x90\x00" * 10)
            tmp_path = f.name

        doc = {**AUDIO_DOC, "file_path": tmp_path}
        db = make_db(doc=doc)
        try:
            with patch("app.routers.documents.get_db", return_value=db), \
                 PATCHES["connect"], PATCHES["close"]:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/api/media/d2")
                    assert resp.status_code == 200
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_serve_media_file_missing(self):
        doc = {**AUDIO_DOC, "file_path": "nonexistent/path.mp3"}
        db = make_db(doc=doc)
        with patch("app.routers.documents.get_db", return_value=db), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/media/d2")
                assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_with_index_error(self):
        db = make_db(doc=DOC)
        with patch("app.routers.documents.get_db", return_value=db), \
             patch("app.routers.documents.delete_file"), \
             patch("app.routers.documents.remove_document_from_index", side_effect=Exception("idx err")), \
             PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/api/documents/d1")
                assert resp.status_code == 200  # Should succeed despite index error


# ── LLM service: stream_answer (Groq-based) ──────────────────────────

def _make_stream_mock(parts):
    chunks = []
    for part in parts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = part
        chunks.append(chunk)

    async def _aiter(self):
        for c in chunks:
            yield c

    mock_stream = MagicMock()
    type(mock_stream).__aiter__ = _aiter
    return mock_stream


class TestLLMStream:
    @pytest.mark.asyncio
    async def test_stream_answer(self):
        from app.services.llm_service import stream_answer

        mock_stream = _make_stream_mock(["Hello", " world"])
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("app.services.llm_service.client", mock_client):
            results = []
            async for t in stream_answer("question", [{"text": "content", "start_time": 0.0, "end_time": 5.0, "page_number": None}], "doc.mp3"):
                results.append(t)
            assert "Hello" in results

    @pytest.mark.asyncio
    async def test_stream_answer_with_page(self):
        from app.services.llm_service import stream_answer

        mock_stream = _make_stream_mock(["Answer"])
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("app.services.llm_service.client", mock_client):
            results = []
            async for t in stream_answer("q", [{"text": "pdf content", "start_time": None, "end_time": None, "page_number": 3}], "doc.pdf"):
                results.append(t)
            assert "Answer" in results


# ── Transcription: full flows ────────────────────────────────────────

class TestTranscriptionFlows:
    def test_get_groq_client(self):
        from app.services.transcription import get_groq_client
        with patch("app.services.transcription.Groq") as MockGroq:
            client = get_groq_client()
            MockGroq.assert_called_once()

    def test_transcribe_audio_small(self):
        from app.services.transcription import transcribe_audio

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 5.0
        mock_seg.text = "Hello"

        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = [mock_seg]
        mock_response.duration = 5.0

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\xff" * 100)
            tmp = f.name

        try:
            with patch("app.services.transcription.get_groq_client", return_value=mock_client):
                result = transcribe_audio(tmp)
                assert result["text"] == "Hello world"
                assert len(result["segments"]) == 1
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_transcribe_large_file(self):
        from app.services.transcription import _transcribe_large_file

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 5.0
        mock_seg.text = "Chunk text"

        mock_response = MagicMock()
        mock_response.text = "Chunk text"
        mock_response.segments = [mock_seg]
        mock_response.duration = 600.0

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake chunk file
            chunk_file = Path(tmpdir) / "chunk_000.mp3"
            chunk_file.write_bytes(b"\xff" * 100)

            with patch("app.services.transcription.settings") as mock_settings, \
                 patch("subprocess.run") as mock_run, \
                 patch("tempfile.mkdtemp", return_value=tmpdir):
                mock_settings.FFMPEG_PATH = "ffmpeg"
                mock_run.return_value = MagicMock(returncode=0)

                result = _transcribe_large_file(Path("fake.mp3"), mock_client)
                assert "Chunk text" in result["text"]

    def test_transcribe_file_not_found(self):
        from app.services.transcription import transcribe_audio
        with pytest.raises(FileNotFoundError):
            transcribe_audio("nonexistent.mp3")


# ── Main app lifespan ────────────────────────────────────────────────

class TestAppLifespan:
    @pytest.mark.asyncio
    async def test_root(self):
        with PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/")
                assert resp.status_code == 200
                assert resp.json()["status"] == "running"

    @pytest.mark.asyncio
    async def test_health(self):
        with PATCHES["connect"], PATCHES["close"]:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
                assert resp.status_code == 200
                assert resp.json()["status"] == "healthy"


# ── Embedding: save/load index ───────────────────────────────────────

class TestEmbeddingPersistence:
    def test_save_and_load(self):
        import app.services.embedding as emb
        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            import faiss
            emb._faiss_index = faiss.IndexFlatIP(384)
            vec = np.random.randn(1, 384).astype(np.float32)
            faiss.normalize_L2(vec)
            emb._faiss_index.add(vec)
            emb._chunk_metadata = [{"document_id": "d1", "text": "test", "chunk_index": 0}]

            emb._save_index()
            assert (Path(tmpdir) / "index.faiss").exists()
            assert (Path(tmpdir) / "metadata.json").exists()

            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._load_index()
            assert emb._faiss_index.ntotal == 1
            assert len(emb._chunk_metadata) == 1

            # Reset
            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

    def test_load_missing_files(self):
        import app.services.embedding as emb
        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._load_index()
            assert emb._faiss_index is not None
            assert emb._faiss_index.ntotal == 0

            # Reset
            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

