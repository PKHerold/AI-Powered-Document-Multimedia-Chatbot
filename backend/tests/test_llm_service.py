"""Tests for LLM service (Groq-based)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.llm_service import (
    answer_question, summarize_text, stream_answer,
    _format_time, _parse_time, _extract_timestamps_from_answer,
)


# ── Helper: build a fake async streaming response ─────────────────────

def make_stream_mock(content_parts: list[str]):
    """Build an async-iterable mock for Groq streaming."""
    chunks = []
    for part in content_parts:
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


# ── _format_time ──────────────────────────────────────────────────────

class TestFormatTime:
    def test_format_seconds(self):
        assert _format_time(65) == "01:05"

    def test_format_zero(self):
        assert _format_time(0) == "00:00"

    def test_format_large(self):
        assert _format_time(3599) == "59:59"

    def test_format_minutes_only(self):
        assert _format_time(120) == "02:00"


# ── _parse_time ───────────────────────────────────────────────────────

class TestParseTime:
    def test_parse_mm_ss(self):
        assert _parse_time("02:30") == 150

    def test_parse_zero(self):
        assert _parse_time("00:00") == 0

    def test_parse_one_minute(self):
        assert _parse_time("01:00") == 60


# ── _extract_timestamps_from_answer ──────────────────────────────────

class TestExtractTimestamps:
    def test_extracts_from_answer(self):
        answer = "The topic is discussed at [02:30 - 05:00] in the video."
        available = [{"start": 150, "end": 300, "text": "topic"}]
        result = _extract_timestamps_from_answer(answer, available)
        assert len(result) >= 1

    def test_no_timestamps(self):
        result = _extract_timestamps_from_answer("No timestamps here", [])
        assert result == []

    def test_no_match_in_available_returns_fallback(self):
        answer = "See [10:00 - 11:00] for details."
        available = [{"start": 0, "end": 5, "text": "unrelated"}]
        result = _extract_timestamps_from_answer(answer, available)
        assert isinstance(result, list)


# ── answer_question (async) ───────────────────────────────────────────

class TestAnswerQuestion:
    @pytest.mark.asyncio
    async def test_returns_answer(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "The answer is 42."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("app.services.llm_service.client", mock_client):
            chunks = [{"text": "Context", "chunk_index": 0,
                       "start_time": None, "end_time": None, "page_number": 1}]
            result = await answer_question("What is the answer?", chunks, "test.pdf")

        assert result["answer"] == "The answer is 42."
        assert "sources" in result
        assert "timestamps" in result

    @pytest.mark.asyncio
    async def test_with_timestamp_chunks(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "Discussed at [00:30 - 01:00]"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("app.services.llm_service.client", mock_client):
            chunks = [{"text": "Audio content", "chunk_index": 0,
                       "start_time": 30.0, "end_time": 60.0, "page_number": None}]
            result = await answer_question("What is discussed?", chunks)

        assert result["answer"] is not None
        assert len(result["timestamps"]) > 0

    @pytest.mark.asyncio
    async def test_with_page_number_chunks(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "Found on page 3."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("app.services.llm_service.client", mock_client):
            chunks = [{"text": "PDF content", "chunk_index": 0,
                       "start_time": None, "end_time": None, "page_number": 3}]
            result = await answer_question("Where is it?", chunks, "doc.pdf")

        assert result["answer"] == "Found on page 3."

    @pytest.mark.asyncio
    async def test_no_document_name(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("app.services.llm_service.client", mock_client):
            result = await answer_question("Question?", [{"text": "ctx", "chunk_index": 0}])

        assert "answer" in result


# ── summarize_text (sync) ─────────────────────────────────────────────

class TestSummarizeText:
    def _make_sync_mock(self, content: str):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_sync_client = MagicMock()
        mock_sync_client.chat.completions.create.return_value = mock_response
        return mock_sync_client

    def test_returns_summary(self):
        mock_sync_client = self._make_sync_mock("This document discusses AI.")
        with patch("groq.Groq", return_value=mock_sync_client):
            result = summarize_text("Long document about AI...", "test.pdf")
        assert "AI" in result

    def test_truncates_long_text(self):
        mock_sync_client = self._make_sync_mock("Summary of truncated doc.")
        with patch("groq.Groq", return_value=mock_sync_client):
            long_text = "a" * 200000
            result = summarize_text(long_text)
        assert result == "Summary of truncated doc."

    def test_short_text_not_truncated(self):
        mock_sync_client = self._make_sync_mock("Short summary.")
        with patch("groq.Groq", return_value=mock_sync_client):
            result = summarize_text("Short text", "doc.pdf")
        assert result == "Short summary."


# ── stream_answer (async generator) ──────────────────────────────────

class TestStreamAnswer:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        mock_stream = make_stream_mock(["Hello ", "world"])
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("app.services.llm_service.client", mock_client):
            results = []
            context = [{"text": "ctx", "chunk_index": 0,
                        "start_time": None, "end_time": None, "page_number": 1}]
            async for text in stream_answer("question", context, "doc.pdf"):
                results.append(text)

        assert "Hello " in results
        assert "world" in results

    @pytest.mark.asyncio
    async def test_stream_with_timestamp_chunks(self):
        mock_stream = make_stream_mock(["Answer"])
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("app.services.llm_service.client", mock_client):
            results = []
            context = [{"text": "audio content", "chunk_index": 0,
                        "start_time": 5.0, "end_time": 10.0, "page_number": None}]
            async for text in stream_answer("q?", context, "audio.mp3"):
                results.append(text)

        assert results == ["Answer"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_delta(self):
        """None content delta should not be yielded."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None

        async def _aiter(self):
            yield chunk

        mock_stream = MagicMock()
        type(mock_stream).__aiter__ = _aiter

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("app.services.llm_service.client", mock_client):
            results = []
            async for text in stream_answer("q?", [{"text": "ctx", "chunk_index": 0}]):
                results.append(text)

        assert results == []

    @pytest.mark.asyncio
    async def test_stream_empty_choices(self):
        """Chunk with no choices should not be yielded."""
        chunk = MagicMock()
        chunk.choices = []

        async def _aiter(self):
            yield chunk

        mock_stream = MagicMock()
        type(mock_stream).__aiter__ = _aiter

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("app.services.llm_service.client", mock_client):
            results = []
            async for text in stream_answer("q?", [{"text": "ctx", "chunk_index": 0}]):
                results.append(text)

        assert results == []
