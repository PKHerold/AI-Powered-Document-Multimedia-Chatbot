"""Tests for PDF service."""

import pytest
from unittest.mock import patch, MagicMock
from app.services.pdf_service import extract_text_from_pdf, get_full_text_from_pdf, _split_into_chunks


class TestSplitIntoChunks:
    def test_splits_text(self):
        segments = [{"text": " ".join(["word"] * 600), "page_number": 1}]
        chunks = _split_into_chunks(segments, chunk_size=500, overlap=50)
        assert len(chunks) >= 2

    def test_empty_segments(self):
        assert _split_into_chunks([]) == []

    def test_preserves_page_number(self):
        segments = [{"text": "hello world", "page_number": 3}]
        chunks = _split_into_chunks(segments)
        assert chunks[0]["page_number"] == 3

    def test_chunk_has_required_fields(self):
        segments = [{"text": "test content", "page_number": 1}]
        chunks = _split_into_chunks(segments)
        assert "id" in chunks[0]
        assert "text" in chunks[0]
        assert "chunk_index" in chunks[0]

    def test_small_text_single_chunk(self):
        segments = [{"text": "short text", "page_number": 1}]
        chunks = _split_into_chunks(segments, chunk_size=500)
        assert len(chunks) == 1


class TestExtractTextFromPdf:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/path.pdf")

    @patch("app.services.pdf_service.pdfplumber.open")
    def test_extracts_text(self, mock_open):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 content here"
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        mock_open.return_value = mock_pdf

        with patch("pathlib.Path.exists", return_value=True):
            chunks = extract_text_from_pdf("test.pdf")

        assert len(chunks) > 0
        assert "Page 1 content" in chunks[0]["text"]

    @patch("app.services.pdf_service.pdfplumber.open")
    def test_empty_pages_skipped(self, mock_open):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        mock_open.return_value = mock_pdf

        with patch("pathlib.Path.exists", return_value=True):
            chunks = extract_text_from_pdf("test.pdf")

        assert chunks == []


class TestGetFullTextFromPdf:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            get_full_text_from_pdf("/nonexistent.pdf")

    @patch("app.services.pdf_service.pdfplumber.open")
    def test_returns_full_text(self, mock_open):
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2"
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [page1, page2]
        mock_open.return_value = mock_pdf

        with patch("pathlib.Path.exists", return_value=True):
            text = get_full_text_from_pdf("test.pdf")

        assert "Page 1" in text
        assert "Page 2" in text
