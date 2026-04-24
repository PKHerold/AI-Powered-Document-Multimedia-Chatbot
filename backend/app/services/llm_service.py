"""LLM service using Groq for chat, Q&A, and summarization."""

import json
from groq import AsyncGroq
from app.config import settings

# Configure Groq client
client = AsyncGroq(api_key=settings.GROQ_API_KEY)

# We use a fast model for chat and summarization
MODEL_NAME = "llama-3.1-8b-instant"


async def answer_question(question: str, context_chunks: list[dict], document_name: str = "") -> dict:
    """
    Answer a question using relevant document context chunks.

    Args:
        question: User's question
        context_chunks: List of relevant text chunks with metadata
        document_name: Name of the source document

    Returns:
        {"answer": str, "timestamps": [{"start", "end", "text"}], "sources": [...]}
    """
    # Build context from chunks
    context_parts = []
    timestamps_info = []
    for i, chunk in enumerate(context_chunks):
        chunk_text = chunk.get("text", "")
        start_time = chunk.get("start_time")
        end_time = chunk.get("end_time")
        page = chunk.get("page_number")

        location = ""
        if start_time is not None and end_time is not None:
            location = f" [Timestamp: {_format_time(start_time)} - {_format_time(end_time)}]"
            timestamps_info.append({
                "start": start_time,
                "end": end_time,
                "text": chunk_text[:100],
            })
        elif page is not None:
            location = f" [Page {page}]"

        context_parts.append(f"[Chunk {i+1}]{location}:\n{chunk_text}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided document content.
Use ONLY the information from the provided context to answer the question.
If the answer cannot be found in the context, say so clearly.

When the content comes from audio/video with timestamps, reference the relevant timestamps in your answer using the format [MM:SS - MM:SS].

Document: {document_name}

Context:
{context}"""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    
    answer = response.choices[0].message.content

    # Extract any timestamps referenced in the answer
    referenced_timestamps = _extract_timestamps_from_answer(answer, timestamps_info)

    return {
        "answer": answer,
        "timestamps": referenced_timestamps if referenced_timestamps else timestamps_info[:3],
        "sources": [{"chunk_index": c.get("chunk_index", i), "text": c.get("text", "")[:200]} for i, c in enumerate(context_chunks)],
    }


def summarize_text(text: str, document_name: str = "") -> str:
    """
    Generate a summary of the document text using Groq (Synchronous wrapper).
    Since the caller in `upload.py` and `reprocess.py` might be expecting a sync function,
    we'll run this using asyncio if needed, or use the sync client.
    Wait, `upload.py` and `reprocess.py` call `summarize_text` synchronously.
    Let's use the sync client here.
    """
    from groq import Groq
    sync_client = Groq(api_key=settings.GROQ_API_KEY)
    
    # Truncate if too long (Groq Llama3 has 8192 context window, approx 6000 words)
    max_chars = 25000 
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Text truncated...]"

    system_prompt = "You are a helpful AI assistant that summarizes documents."
    prompt = f"""Provide a comprehensive summary of the following document.
Include the main topics, key points, and important details.
If it's a transcript with timestamps, highlight the main discussion topics and when they occur.

Document: {document_name}

Content:
{text}

Summary:"""

    response = sync_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


async def stream_answer(question: str, context_chunks: list[dict], document_name: str = ""):
    """
    Stream an answer using Groq's streaming API.

    Yields chunks of the response text.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        chunk_text = chunk.get("text", "")
        start_time = chunk.get("start_time")
        end_time = chunk.get("end_time")
        page = chunk.get("page_number")

        location = ""
        if start_time is not None and end_time is not None:
            location = f" [Timestamp: {_format_time(start_time)} - {_format_time(end_time)}]"
        elif page is not None:
            location = f" [Page {page}]"

        context_parts.append(f"[Chunk {i+1}]{location}:\n{chunk_text}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are a helpful AI assistant answering questions about a document.
Use ONLY the context provided. Reference timestamps [MM:SS - MM:SS] when relevant.

Document: {document_name}
Context:
{context}"""

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _format_time(seconds: float) -> str:
    """Format seconds to MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _extract_timestamps_from_answer(answer: str, available_timestamps: list[dict]) -> list[dict]:
    """Extract timestamps that are referenced in the AI's answer."""
    import re
    # Look for patterns like [01:23 - 04:56] or [1:23-4:56]
    pattern = r'\[?(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]?'
    matches = re.findall(pattern, answer)

    referenced = []
    for start_str, end_str in matches:
        start_secs = _parse_time(start_str)
        end_secs = _parse_time(end_str)
        # Find the closest matching timestamp from available ones
        for ts in available_timestamps:
            if abs(ts["start"] - start_secs) < 30:  # within 30 seconds
                referenced.append(ts)
                break
        else:
            referenced.append({"start": start_secs, "end": end_secs, "text": ""})

    return referenced


def _parse_time(time_str: str) -> float:
    """Parse MM:SS to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])
