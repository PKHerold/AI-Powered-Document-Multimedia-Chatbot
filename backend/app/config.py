"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings:
    """Application settings from environment variables."""

    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # MongoDB
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "docqa")

    # App Settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")

    # FFmpeg
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")

    # Embedding dimensions (all-MiniLM-L6-v2 sentence-transformer)
    EMBEDDING_DIM: int = 384

    # Chunk settings
    CHUNK_SIZE: int = 500  # tokens per chunk
    CHUNK_OVERLAP: int = 50

    @classmethod
    def get_upload_path(cls) -> Path:
        """Get the upload directory path, creating it if needed."""
        path = Path(cls.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
