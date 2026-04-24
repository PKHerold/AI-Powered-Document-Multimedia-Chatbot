"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.routers import upload, documents, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    # ── Startup ──────────────────────────────────────────────
    print("[*] Starting AI Document Q&A Server...")
    await connect_to_mongo()

    # Ensure upload directory exists
    settings.get_upload_path()

    print("[+] Server ready!")
    yield

    # ── Shutdown ─────────────────────────────────────────────
    print("[-] Shutting down...")
    await close_mongo_connection()


app = FastAPI(
    title="AI Document & Multimedia Q&A",
    description="Upload PDFs, audio, and video files. Ask questions with AI-powered answers and timestamp playback.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint — health check."""
    return {
        "name": "AI Document & Multimedia Q&A API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
