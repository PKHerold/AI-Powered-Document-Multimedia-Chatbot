"""Utility helper functions."""

import uuid
import re


def generate_id() -> str:
    """Generate a unique ID for documents."""
    return str(uuid.uuid4())


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(timestamp_str: str) -> float:
    """Parse HH:MM:SS or MM:SS format to seconds."""
    parts = timestamp_str.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(timestamp_str)


def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def classify_file_type(filename: str) -> str:
    """Classify file type based on extension."""
    ext = get_file_extension(filename)
    if ext == "pdf":
        return "pdf"
    elif ext in ("mp3", "wav", "ogg", "flac", "m4a", "aac", "wma"):
        return "audio"
    elif ext in ("mp4", "avi", "mkv", "mov", "webm", "wmv", "flv"):
        return "video"
    return "unknown"


ALLOWED_EXTENSIONS = {
    "pdf", "mp3", "wav", "ogg", "flac", "m4a", "aac",
    "mp4", "avi", "mkv", "mov", "webm"
}


def is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove path separators and special chars
    name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(name) > 200:
        ext = get_file_extension(name)
        name = name[:195] + "." + ext
    return name


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
