"""Media file handling service — storage, serving, and audio extraction."""

import os
import shutil
from pathlib import Path
from fastapi import UploadFile
from app.config import settings
from app.utils import generate_id, sanitize_filename, get_file_extension


async def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    """
    Save an uploaded file to the uploads directory.

    Returns:
        (file_id, file_path) tuple
    """
    upload_dir = settings.get_upload_path()
    file_id = generate_id()
    ext = get_file_extension(upload_file.filename or "file")
    safe_name = f"{file_id}.{ext}"
    file_path = upload_dir / safe_name

    # Stream file to disk
    with open(file_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)

    return file_id, str(file_path)


def get_file_path(filename: str) -> Path:
    """Get the full path of a stored file."""
    return settings.get_upload_path() / filename


def delete_file(file_path: str):
    """Delete a file from disk."""
    path = Path(file_path)
    if path.exists():
        path.unlink()

    # Also delete any extracted audio files
    audio_path = path.with_suffix(".mp3")
    if audio_path.exists():
        audio_path.unlink()


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()
