"""Chat and Q&A endpoints."""

import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.database import get_db
from app.models import ChatRequest, ChatResponse, ChatMessage
from app.services.embedding import search_similar_chunks
from app.services.llm_service import answer_question, stream_answer
from app.utils import generate_id

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about an uploaded document.

    Uses semantic search to find relevant chunks, then asks Gemini to answer.
    """
    db = get_db()

    # Verify document exists
    doc = await db.documents.find_one({"_id": request.document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document is still {doc.get('status', 'processing')}. Please wait for processing to complete.",
        )

    # Search for relevant chunks using FAISS
    relevant_chunks = search_similar_chunks(
        query=request.question,
        document_id=request.document_id,
        top_k=5,
    )

    if not relevant_chunks:
        # Fallback: get chunks directly from MongoDB
        chunks = []
        async for chunk in db.chunks.find({"document_id": request.document_id}).limit(5):
            chunks.append(chunk)
        relevant_chunks = chunks

    # Get answer from LLM
    result = await answer_question(
        question=request.question,
        context_chunks=relevant_chunks,
        document_name=doc.get("original_filename", ""),
    )

    # Save to chat history
    await _save_chat_message(
        db,
        request.document_id,
        ChatMessage(
            role="user",
            content=request.question,
            created_at=datetime.now(timezone.utc),
        ),
    )
    await _save_chat_message(
        db,
        request.document_id,
        ChatMessage(
            role="assistant",
            content=result["answer"],
            timestamps=result.get("timestamps", []),
            created_at=datetime.now(timezone.utc),
        ),
    )

    return ChatResponse(
        answer=result["answer"],
        timestamps=result.get("timestamps", []),
        sources=result.get("sources", []),
    )


@router.get("/stream")
async def chat_stream(document_id: str, question: str):
    """
    Stream a chat response using Server-Sent Events (SSE).

    Query params:
        document_id: ID of the document to query
        question: The user's question
    """
    db = get_db()

    # Verify document
    doc = await db.documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Document not yet processed")

    # Search for relevant chunks
    relevant_chunks = search_similar_chunks(
        query=question,
        document_id=document_id,
        top_k=5,
    )

    if not relevant_chunks:
        chunks = []
        async for chunk in db.chunks.find({"document_id": document_id}).limit(5):
            chunks.append(chunk)
        relevant_chunks = chunks

    async def event_generator():
        full_answer = ""
        async for text_chunk in stream_answer(
            question=question,
            context_chunks=relevant_chunks,
            document_name=doc.get("original_filename", ""),
        ):
            full_answer += text_chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': text_chunk})}\n\n"

        # Send timestamps at the end
        timestamps = []
        for chunk in relevant_chunks:
            if chunk.get("start_time") is not None:
                timestamps.append({
                    "start": chunk["start_time"],
                    "end": chunk.get("end_time", chunk["start_time"]),
                    "text": chunk.get("text", "")[:100],
                })

        yield f"data: {json.dumps({'type': 'done', 'timestamps': timestamps[:5]})}\n\n"

        # Save to history
        await _save_chat_message(db, document_id, ChatMessage(
            role="user", content=question, created_at=datetime.now(timezone.utc),
        ))
        await _save_chat_message(db, document_id, ChatMessage(
            role="assistant", content=full_answer, timestamps=timestamps,
            created_at=datetime.now(timezone.utc),
        ))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{document_id}")
async def get_chat_history(document_id: str):
    """Get chat history for a document."""
    db = get_db()

    # Verify document exists
    doc = await db.documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    history = await db.chat_history.find_one({"document_id": document_id})
    if not history:
        return {"document_id": document_id, "messages": []}

    return {
        "document_id": document_id,
        "messages": history.get("messages", []),
    }


@router.delete("/history/{document_id}")
async def clear_chat_history(document_id: str):
    """Clear chat history for a document."""
    db = get_db()
    await db.chat_history.delete_many({"document_id": document_id})
    return {"message": "Chat history cleared", "document_id": document_id}


async def _save_chat_message(db, document_id: str, message: ChatMessage):
    """Save a chat message to history."""
    msg_dict = message.model_dump()

    # Upsert: create history document if it doesn't exist, append message
    result = await db.chat_history.update_one(
        {"document_id": document_id},
        {
            "$push": {"messages": msg_dict},
            "$set": {"updated_at": datetime.now(timezone.utc)},
            "$setOnInsert": {
                "_id": generate_id(),
                "document_id": document_id,
                "created_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
    )
