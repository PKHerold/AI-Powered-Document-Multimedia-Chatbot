"""Tests for transcription service."""

import pytest
from unittest.mock import patch, MagicMock
from app.services.transcription import (
    segments_to_chunks,
    extract_audio_from_video,
    transcribe_audio,
)


class TestSegmentsToChunks:
    def test_groups_segments(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": " ".join(["word"] * 100)},
            {"start": 5.0, "end": 10.0, "text": " ".join(["word"] * 100)},
            {"start": 10.0, "end": 15.0, "text": " ".join(["word"] * 50)},
        ]
        chunks = segments_to_chunks("doc-1", segments)
        assert len(chunks) >= 1
        assert chunks[0]["document_id"] == "doc-1"

    def test_empty_segments(self):
        assert segments_to_chunks("doc-1", []) == []

    def test_preserves_timestamps(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": " ".join(["word"] * 250)},
        ]
        chunks = segments_to_chunks("doc-1", segments)
        assert chunks[0]["start_time"] == 0.0

    def test_chunk_has_required_fields(self):
        segments = [{"start": 0.0, "end": 5.0, "text": "hello world test"}]
        chunks = segments_to_chunks("doc-1", segments)
        assert "id" in chunks[0]
        assert "text" in chunks[0]
        assert "chunk_index" in chunks[0]


class TestExtractAudioFromVideo:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_audio_from_video("/nonexistent/video.mp4")

    @patch("app.services.transcription.subprocess.run")
    def test_ffmpeg_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="FFmpeg error")
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="FFmpeg error"):
                extract_audio_from_video("test.mp4")

    @patch("app.services.transcription.subprocess.run")
    def test_successful_extraction(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with patch("pathlib.Path.exists", return_value=True):
            result = extract_audio_from_video("test.mp4", "/output/test.mp3")
            assert result == "/output/test.mp3"


class TestTranscribeAudio:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            transcribe_audio("/nonexistent/audio.mp3")

    @patch("app.services.transcription.get_groq_client")
    def test_successful_transcription(self, mock_client_fn):
        mock_client = MagicMock()
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 5.0
        mock_seg.text = "Test transcription"

        mock_response = MagicMock()
        mock_response.text = "Test transcription"
        mock_response.segments = [mock_seg]
        mock_response.duration = 5.0

        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=1000)
                with patch("builtins.open", MagicMock()):
                    result = transcribe_audio("test.mp3")

        assert result["text"] == "Test transcription"
        assert len(result["segments"]) == 1
        assert result["duration"] == 5.0
