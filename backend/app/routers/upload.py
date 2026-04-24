"""File upload endpoints."""

import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.database import get_db
from app.models import UploadResponse, FileType, ProcessingStatus
from app.utils import generate_id, classify_file_type, is_allowed_file, get_file_extension
from app.services import media_service
from app.services.pdf_service import extract_text_from_pdf, get_full_text_from_pdf
from app.services.transcription import (
    transcribe_audio,
    extract_audio_from_video,
    segments_to_chunks,
)
from app.services.embedding import add_chunks_to_index
from app.services.llm_service import summarize_text
from app.config import settings
from datetime import datetime, timezone

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF, audio, or video file for processing.

    The file is saved, then processed asynchronously:
    - PDFs: text extraction + chunking + embedding
    - Audio: transcription + chunking + embedding
    - Video: audio extraction + transcription + chunking + embedding
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: PDF, MP3, WAV, OGG, FLAC, M4A, MP4, AVI, MKV, MOV, WEBM",
        )

    # Check file size
    content = await file.read()
    file_size = len(content)
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Reset file position after reading
    await file.seek(0)

    # Save file to disk
    file_id, file_path = await media_service.save_upload_file(file)

    # Determine file type
    file_type_str = classify_file_type(file.filename)
    file_type = FileType(file_type_str)

    # Create document record in MongoDB
    db = get_db()
    doc_record = {
        "_id": file_id,
        "filename": f"{file_id}.{get_file_extension(file.filename)}",
        "original_filename": file.filename,
        "file_type": file_type.value,
        "file_size": file_size,
        "file_path": file_path,
        "upload_time": datetime.now(timezone.utc),
        "status": ProcessingStatus.PROCESSING.value,
        "summary": "",
        "metadata": {},
        "chunk_count": 0,
        "duration": None,
    }
    await db.documents.insert_one(doc_record)

    # Process file in background
    background_tasks.add_task(process_document, file_id, file_path, file_type, file.filename)

    return UploadResponse(
        document_id=file_id,
        filename=file.filename,
        file_type=file_type,
        status=ProcessingStatus.PROCESSING,
        message=f"File uploaded successfully. Processing started.",
    )


async def process_document(
    doc_id: str,
    file_path: str,
    file_type: FileType,
    original_filename: str,
):
    """Background task to process an uploaded document."""
    db = get_db()

    try:
        chunks = []

        if file_type == FileType.PDF:
            # Extract text and create chunks
            chunks = extract_text_from_pdf(file_path)

            # Generate summary
            full_text = get_full_text_from_pdf(file_path)
            summary = summarize_text(full_text, original_filename)

            await db.documents.update_one(
                {"_id": doc_id},
                {"$set": {"summary": summary}},
            )

        elif file_type == FileType.AUDIO:
            # Transcribe audio
            result = transcribe_audio(file_path)
            chunks = segments_to_chunks(doc_id, result["segments"])

            # Generate summary from transcription
            summary = summarize_text(result["text"], original_filename)

            await db.documents.update_one(
                {"_id": doc_id},
                {"$set": {
                    "summary": summary,
                    "duration": result.get("duration", 0),
                    "metadata.transcription": result["text"][:5000],
                }},
            )

        elif file_type == FileType.VIDEO:
            # Extract audio from video
            audio_path = extract_audio_from_video(file_path)

            # Transcribe extracted audio
            result = transcribe_audio(audio_path)
            chunks = segments_to_chunks(doc_id, result["segments"])

            # Generate summary
            summary = summarize_text(result["text"], original_filename)

            await db.documents.update_one(
                {"_id": doc_id},
                {"$set": {
                    "summary": summary,
                    "duration": result.get("duration", 0),
                    "metadata.transcription": result["text"][:5000],
                }},
            )

        # Store chunks in MongoDB
        if chunks:
            for chunk in chunks:
                chunk["document_id"] = doc_id
                if "_id" not in chunk:
                    chunk["_id"] = chunk.pop("id", generate_id())
            await db.chunks.insert_many(chunks)

            # Add chunks to FAISS vector index
            add_chunks_to_index(chunks, doc_id)

        # Update document status
        await db.documents.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": ProcessingStatus.COMPLETED.value,
                "chunk_count": len(chunks),
            }},
        )

        print(f"[+] Document {doc_id} processed: {len(chunks)} chunks")

    except Exception as e:
        print(f"[!] Error processing document {doc_id}: {e}")
        await db.documents.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": ProcessingStatus.FAILED.value,
                "metadata.error": str(e),
            }},
        )
