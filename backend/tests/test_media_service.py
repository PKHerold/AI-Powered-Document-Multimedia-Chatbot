"""Tests for media_service."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.media_service import (
    save_upload_file, get_file_path, delete_file,
    get_file_size, file_exists,
)


class TestGetFilePath:
    def test_returns_path(self):
        result = get_file_path("test.pdf")
        assert isinstance(result, Path)
        assert result.name == "test.pdf"


class TestDeleteFile:
    def test_delete_nonexistent(self):
        # Should not raise
        delete_file("/nonexistent/file.pdf")

    @patch("app.services.media_service.Path")
    def test_delete_existing(self, MockPath):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_audio = MagicMock()
        mock_audio.exists.return_value = False
        mock_path.with_suffix.return_value = mock_audio
        MockPath.return_value = mock_path

        delete_file("test.pdf")
        mock_path.unlink.assert_called_once()


class TestFileExists:
    def test_nonexistent(self):
        assert file_exists("/nonexistent/path.pdf") is False


class TestGetFileSize:
    def test_existing_file(self, temp_dir):
        f = temp_dir / "test.txt"
        f.write_text("hello")
        assert get_file_size(str(f)) == 5


@pytest.mark.asyncio
async def test_save_upload_file(temp_dir):
    """Test saving an uploaded file."""
    with patch("app.services.media_service.settings") as mock_settings:
        mock_settings.get_upload_path.return_value = temp_dir

        mock_file = AsyncMock()
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"fake pdf content")

        file_id, file_path = await save_upload_file(mock_file)
        assert file_id
        assert file_path.endswith(".pdf")
        assert Path(file_path).exists()
