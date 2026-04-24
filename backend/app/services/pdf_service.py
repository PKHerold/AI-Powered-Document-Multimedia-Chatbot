"""PDF text extraction service using pdfplumber."""

import pdfplumber
from pathlib import Path
from app.models import TextChunk
from app.utils import generate_id


def extract_text_from_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file, split into chunks with page numbers.

    Returns a list of dicts: [{text, page_number, chunk_index}]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    all_text_segments = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                all_text_segments.append({
                    "text": text.strip(),
                    "page_number": page_num,
                })

    # Now split into chunks
    chunks = _split_into_chunks(all_text_segments)
    return chunks


def _split_into_chunks(
    segments: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split text segments into overlapping chunks of approximately chunk_size words.
    """
    chunks = []
    chunk_index = 0

    for segment in segments:
        words = segment["text"].split()
        page_number = segment.get("page_number")

        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "id": generate_id(),
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "start_time": None,
                    "end_time": None,
                })
                chunk_index += 1

            i += chunk_size - overlap

    return chunks


def get_full_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF as a single string."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text.strip())

    return "\n\n".join(full_text)
