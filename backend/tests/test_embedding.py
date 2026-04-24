"""Tests for embedding service (sentence-transformers + FAISS)."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.services.embedding import (
    add_chunks_to_index, search_similar_chunks,
    remove_document_from_index, get_index_stats,
    generate_embedding, generate_query_embedding,
    _ensure_index, _save_index, _load_index,
)


def make_model_mock(dim: int = 384, n_vectors: int = 1):
    """Return a mock sentence-transformer model that returns random vectors."""
    mock = MagicMock()
    # encode() is called with either a list (batch add) or a string (query)
    mock.encode.side_effect = lambda x, **kw: (
        np.random.randn(len(x), dim).astype(np.float32)
        if isinstance(x, list) else
        np.random.randn(dim).astype(np.float32)
    )
    return mock


@pytest.fixture(autouse=True)
def reset_index():
    """Reset FAISS index before each test to avoid state bleed."""
    import app.services.embedding as emb
    emb._faiss_index = None
    emb._chunk_metadata = []
    
    # Use a temporary directory for the index so _ensure_index doesn't load real files
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = emb._index_path
        emb._index_path = Path(tmpdir)
        yield
        emb._faiss_index = None
        emb._chunk_metadata = []
        emb._index_path = original_path


# ── generate_embedding / generate_query_embedding ────────────────────

class TestGenerateEmbedding:
    def test_returns_list_of_floats(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock):
            result = generate_embedding("test text")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_query_embedding_returns_list(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock):
            result = generate_query_embedding("test query")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_returns_zeros_when_model_none(self):
        with patch("app.services.embedding._model", None):
            result = generate_embedding("text")
        assert result == [0.0] * 384

    def test_query_returns_zeros_when_model_none(self):
        with patch("app.services.embedding._model", None):
            result = generate_query_embedding("query")
        assert result == [0.0] * 384


# ── add_chunks_to_index ───────────────────────────────────────────────

class TestAddChunks:
    def test_add_chunks_updates_index(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            chunks = [
                {"text": "chunk 1", "chunk_index": 0, "page_number": 1},
                {"text": "chunk 2", "chunk_index": 1, "page_number": 1},
            ]
            add_chunks_to_index(chunks, "doc-1")
        stats = get_index_stats()
        assert stats["total_vectors"] == 2
        assert stats["total_documents"] == 1

    def test_add_empty_chunks_does_nothing(self):
        with patch("app.services.embedding._save_index"):
            add_chunks_to_index([], "doc-1")
        assert get_index_stats()["total_vectors"] == 0

    def test_add_chunks_with_timestamps(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            chunks = [{"text": "audio chunk", "chunk_index": 0,
                       "start_time": 0.0, "end_time": 5.0, "page_number": None}]
            add_chunks_to_index(chunks, "audio-doc")
        stats = get_index_stats()
        assert stats["total_vectors"] == 1

    def test_add_chunks_model_none_skips_vectors(self):
        with patch("app.services.embedding._model", None), \
             patch("app.services.embedding._save_index"):
            chunks = [{"text": "chunk", "chunk_index": 0}]
            add_chunks_to_index(chunks, "doc-1")
        # No vectors when model is None
        assert get_index_stats()["total_vectors"] == 0

    def test_add_multiple_batches(self):
        """Verify batching (batch_size=32): add 33 chunks to cross batch boundary."""
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            chunks = [{"text": f"chunk {i}", "chunk_index": i} for i in range(33)]
            add_chunks_to_index(chunks, "big-doc")
        assert get_index_stats()["total_vectors"] == 33


# ── search_similar_chunks ─────────────────────────────────────────────

class TestSearch:
    def test_search_returns_results(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            add_chunks_to_index([{"text": "AI is great", "chunk_index": 0}], "doc-1")

        results = search_similar_chunks("AI", document_id="doc-1", top_k=1)
        assert len(results) == 1
        assert results[0]["document_id"] == "doc-1"
        assert "score" in results[0]

    def test_search_without_document_filter(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            add_chunks_to_index([{"text": "data", "chunk_index": 0}], "doc-x")

        results = search_similar_chunks("data", top_k=5)
        assert isinstance(results, list)

    def test_search_empty_index(self):
        _ensure_index()
        results = search_similar_chunks("test query")
        assert results == []

    def test_search_model_none(self):
        _ensure_index()
        with patch("app.services.embedding._model", None):
            results = search_similar_chunks("test")
        assert results == []

    def test_search_skips_invalid_indices(self):
        """If FAISS returns idx=-1 (not found), it should be skipped."""
        import app.services.embedding as emb
        import faiss
        _ensure_index()
        emb._faiss_index = faiss.IndexFlatIP(384)
        emb._chunk_metadata = []  # No metadata → idx will be out of range

        mock = make_model_mock()
        with patch("app.services.embedding._model", mock):
            results = search_similar_chunks("query", top_k=1)
        assert results == []


# ── remove_document_from_index ───────────────────────────────────────

class TestRemoveDocument:
    def test_remove_document(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            add_chunks_to_index([{"text": "doc1 text", "chunk_index": 0}], "doc-1")
            add_chunks_to_index([{"text": "doc2 text", "chunk_index": 0}], "doc-2")
            remove_document_from_index("doc-1")

        stats = get_index_stats()
        assert stats["total_documents"] == 1

    def test_remove_nonexistent_document_is_noop(self):
        _ensure_index()
        with patch("app.services.embedding._save_index"):
            remove_document_from_index("nonexistent-doc")
        assert get_index_stats()["total_vectors"] == 0


# ── get_index_stats ───────────────────────────────────────────────────

class TestStats:
    def test_get_stats_empty(self):
        stats = get_index_stats()
        assert stats["total_vectors"] == 0
        assert stats["total_documents"] == 0
        assert stats["dimension"] == 384

    def test_get_stats_with_data(self):
        mock = make_model_mock()
        with patch("app.services.embedding._model", mock), \
             patch("app.services.embedding._save_index"):
            add_chunks_to_index([{"text": "a", "chunk_index": 0}], "d1")
            add_chunks_to_index([{"text": "b", "chunk_index": 0}], "d2")
        stats = get_index_stats()
        assert stats["total_vectors"] == 2
        assert stats["total_documents"] == 2


# ── Index persistence (save/load) ─────────────────────────────────────

class TestIndexPersistence:
    def test_save_and_load_index(self):
        import app.services.embedding as emb
        import faiss

        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            emb._faiss_index = faiss.IndexFlatIP(384)
            vec = np.random.randn(1, 384).astype(np.float32)
            faiss.normalize_L2(vec)
            emb._faiss_index.add(vec)
            emb._chunk_metadata = [{"document_id": "d1", "text": "test",
                                     "chunk_index": 0, "faiss_index": 0}]

            _save_index()
            assert (Path(tmpdir) / "index.faiss").exists()
            assert (Path(tmpdir) / "metadata.json").exists()

            emb._faiss_index = None
            emb._chunk_metadata = []
            _load_index()
            assert emb._faiss_index.ntotal == 1
            assert len(emb._chunk_metadata) == 1

            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

    def test_load_missing_files_creates_empty_index(self):
        import app.services.embedding as emb

        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            emb._faiss_index = None
            emb._chunk_metadata = []
            _load_index()
            assert emb._faiss_index is not None
            assert emb._faiss_index.ntotal == 0

            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

    def test_load_dimension_mismatch_resets_index(self):
        import app.services.embedding as emb
        import faiss

        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            wrong_index = faiss.IndexFlatIP(768)  # Wrong dim
            faiss.write_index(wrong_index, str(Path(tmpdir) / "index.faiss"))
            (Path(tmpdir) / "metadata.json").write_text("[]")

            emb._faiss_index = None
            emb._chunk_metadata = []
            _load_index()

            assert emb._faiss_index.d == 384
            assert emb._faiss_index.ntotal == 0

            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

    def test_load_corrupt_file_resets(self):
        import app.services.embedding as emb

        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            (Path(tmpdir) / "index.faiss").write_bytes(b"corrupt data")
            (Path(tmpdir) / "metadata.json").write_text("[]")

            emb._faiss_index = None
            emb._chunk_metadata = []
            _load_index()
            assert emb._faiss_index is not None  # Recovered gracefully

            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

    def test_save_skips_empty_index(self):
        import app.services.embedding as emb
        import faiss

        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            emb._faiss_index = faiss.IndexFlatIP(384)  # 0 vectors
            _save_index()
            assert not (Path(tmpdir) / "index.faiss").exists()

            emb._faiss_index = None
            emb._chunk_metadata = []
            emb._index_path = Path("faiss_store")

    def test_save_none_index_is_noop(self):
        import app.services.embedding as emb

        with tempfile.TemporaryDirectory() as tmpdir:
            emb._index_path = Path(tmpdir)
            emb._faiss_index = None
            _save_index()  # Should not crash
            assert not (Path(tmpdir) / "index.faiss").exists()

            emb._index_path = Path("faiss_store")
