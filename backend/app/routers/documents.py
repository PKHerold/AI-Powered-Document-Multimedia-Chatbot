"""Document listing, details, summary, and media serving endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.database import get_db
from app.models import DocumentResponse, SummaryResponse, ProcessingStatus
from app.services.llm_service import summarize_text
from app.services.pdf_service import get_full_text_from_pdf
from app.services.embedding import remove_document_from_index, get_index_stats
from app.services.media_service import delete_file

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """List all uploaded documents."""
    db = get_db()
    cursor = db.documents.find().sort("upload_time", -1)
    docs = []
    async for doc in cursor:
        docs.append(DocumentResponse(
            id=doc["_id"],
            filename=doc.get("filename", ""),
            original_filename=doc.get("original_filename", ""),
            file_type=doc["file_type"],
            file_size=doc.get("file_size", 0),
            upload_time=doc["upload_time"],
            status=doc.get("status", "pending"),
            summary=doc.get("summary", ""),
            chunk_count=doc.get("chunk_count", 0),
            duration=doc.get("duration"),
        ))
    return docs


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get details of a specific document."""
    db = get_db()
    doc = await db.documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=doc["_id"],
        filename=doc.get("filename", ""),
        original_filename=doc.get("original_filename", ""),
        file_type=doc["file_type"],
        file_size=doc.get("file_size", 0),
        upload_time=doc["upload_time"],
        status=doc.get("status", "pending"),
        summary=doc.get("summary", ""),
        chunk_count=doc.get("chunk_count", 0),
        duration=doc.get("duration"),
    )


@router.get("/documents/{document_id}/summary", response_model=SummaryResponse)
async def get_or_generate_summary(document_id: str):
    """Get or generate a summary for a document."""
    db = get_db()
    doc = await db.documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.get("status") != ProcessingStatus.COMPLETED.value:
        return SummaryResponse(
            document_id=document_id,
            summary="",
            status=f"Document is still {doc.get('status', 'processing')}. Please wait.",
        )

    # If summary already exists, return it
    if doc.get("summary"):
        return SummaryResponse(
            document_id=document_id,
            summary=doc["summary"],
            status="completed",
        )

    # Generate summary
    try:
        file_path = doc.get("file_path", "")
        if doc["file_type"] == "pdf":
            full_text = get_full_text_from_pdf(file_path)
        else:
            # Use stored transcription for audio/video
            full_text = doc.get("metadata", {}).get("transcription", "")
            if not full_text:
                # Reconstruct from chunks
                chunks = []
                async for chunk in db.chunks.find({"document_id": document_id}).sort("chunk_index", 1):
                    chunks.append(chunk["text"])
                full_text = " ".join(chunks)

        summary = summarize_text(full_text, doc.get("original_filename", ""))

        await db.documents.update_one(
            {"_id": document_id},
            {"$set": {"summary": summary}},
        )

        return SummaryResponse(
            document_id=document_id,
            summary=summary,
            status="completed",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all associated data."""
    db = get_db()
    doc = await db.documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete file from disk
    file_path = doc.get("file_path", "")
    if file_path:
        delete_file(file_path)

    # Remove from FAISS index
    try:
        remove_document_from_index(document_id)
    except Exception:
        pass  # Index might not have this document

    # Delete from MongoDB
    await db.documents.delete_one({"_id": document_id})
    await db.chunks.delete_many({"document_id": document_id})
    await db.chat_history.delete_many({"document_id": document_id})

    return {"message": "Document deleted successfully", "document_id": document_id}


@router.get("/media/{document_id}")
async def serve_media(document_id: str):
    """Serve a media file (audio/video) for playback."""
    db = get_db()
    doc = await db.documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = doc.get("file_path", "")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Media file not found on disk")

    # Determine media type
    ext = file_path.rsplit(".", 1)[-1].lower()
    media_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "mp4": "video/mp4",
        "avi": "video/x-msvideo",
        "mkv": "video/x-matroska",
        "mov": "video/quicktime",
        "webm": "video/webm",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=doc.get("original_filename", f"media.{ext}"),
    )


@router.get("/stats")
async def get_stats():
    """Get application statistics."""
    db = get_db()

    doc_count = await db.documents.count_documents({})
    chunk_count = await db.chunks.count_documents({})
    chat_count = await db.chat_history.count_documents({})
    index_stats = get_index_stats()

    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "chat_sessions": chat_count,
        "vector_index": index_stats,
    }
