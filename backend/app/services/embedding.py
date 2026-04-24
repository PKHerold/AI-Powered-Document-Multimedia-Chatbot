"""Embedding service using local HuggingFace model + FAISS vector store."""

import json
import numpy as np
import faiss
from pathlib import Path
from app.config import settings

# Global FAISS index and metadata
_faiss_index: faiss.IndexFlatIP | None = None
_chunk_metadata: list[dict] = []
_index_path = Path("faiss_store")

# Initialize sentence-transformer
try:
    from sentence_transformers import SentenceTransformer
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    _model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_DIM = 384
except ImportError:
    print("Warning: sentence_transformers not installed. Embeddings won't work.")
    _model = None
    EMBEDDING_DIM = 384


def _ensure_index():
    """Initialize FAISS index if not already done."""
    global _faiss_index, _chunk_metadata
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine sim with normalized vectors)
        _chunk_metadata = []
        _load_index()


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a piece of text using local model."""
    if _model is None:
        return [0.0] * EMBEDDING_DIM
    embedding = _model.encode(text)
    return embedding.tolist()


def generate_query_embedding(query: str) -> list[float]:
    """Generate an embedding for a search query using local model."""
    if _model is None:
        return [0.0] * EMBEDDING_DIM
    embedding = _model.encode(query)
    return embedding.tolist()


def add_chunks_to_index(chunks: list[dict], document_id: str):
    """
    Generate embeddings for text chunks and add them to the FAISS index.

    Args:
        chunks: List of chunk dicts with 'text', 'chunk_index', etc.
        document_id: The document these chunks belong to
    """
    _ensure_index()

    if not chunks:
        return

    # Generate embeddings in batches
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        if _model is None:
            continue

        # Generate embeddings using local model
        embeddings = _model.encode(texts)

        # Convert to numpy array and normalize
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        # Add to FAISS index
        start_idx = _faiss_index.ntotal
        _faiss_index.add(vectors)

        # Store metadata
        for j, chunk in enumerate(batch):
            _chunk_metadata.append({
                "faiss_index": start_idx + j,
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
                "page_number": chunk.get("page_number"),
                "start_time": chunk.get("start_time"),
                "end_time": chunk.get("end_time"),
            })

    # Save index to disk
    _save_index()


def search_similar_chunks(
    query: str,
    document_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Search for the most relevant chunks to a query.

    Args:
        query: Search query text
        document_id: Optional filter by document
        top_k: Number of results to return

    Returns:
        List of chunk dicts with similarity scores
    """
    _ensure_index()

    if _faiss_index.ntotal == 0 or _model is None:
        return []

    # Generate query embedding
    query_emb = generate_query_embedding(query)
    query_vector = np.array([query_emb], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # Search — get more results if filtering by document_id
    search_k = top_k * 5 if document_id else top_k
    search_k = min(search_k, _faiss_index.ntotal)

    distances, indices = _faiss_index.search(query_vector, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_chunk_metadata):
            continue

        meta = _chunk_metadata[idx]

        # Filter by document_id if specified
        if document_id and meta["document_id"] != document_id:
            continue

        results.append({
            **meta,
            "score": float(dist),
        })

        if len(results) >= top_k:
            break

    return results


def remove_document_from_index(document_id: str):
    """
    Remove all chunks for a document from the FAISS index.
    Since FAISS doesn't support deletion, we rebuild the index.
    """
    global _faiss_index, _chunk_metadata
    _ensure_index()

    # Filter out chunks for this document
    remaining = [m for m in _chunk_metadata if m["document_id"] != document_id]

    if len(remaining) == len(_chunk_metadata):
        return  # Nothing to remove

    # Rebuild index
    _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    _chunk_metadata = []

    if remaining and _model is not None:
        # Re-add remaining chunks
        texts = [meta["text"] for meta in remaining]
        embeddings = _model.encode(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        _faiss_index.add(vectors)
        for i, meta in enumerate(remaining):
            meta["faiss_index"] = i
            _chunk_metadata.append(meta)

    _save_index()


def _save_index():
    """Save FAISS index and metadata to disk."""
    _index_path.mkdir(parents=True, exist_ok=True)
    if _faiss_index is not None and _faiss_index.ntotal > 0:
        faiss.write_index(_faiss_index, str(_index_path / "index.faiss"))
        with open(_index_path / "metadata.json", "w") as f:
            json.dump(_chunk_metadata, f)


def _load_index():
    """Load FAISS index and metadata from disk."""
    global _faiss_index, _chunk_metadata

    index_file = _index_path / "index.faiss"
    meta_file = _index_path / "metadata.json"

    if index_file.exists() and meta_file.exists():
        try:
            temp_index = faiss.read_index(str(index_file))
            if temp_index.d == EMBEDDING_DIM:
                _faiss_index = temp_index
                with open(meta_file, "r") as f:
                    _chunk_metadata = json.load(f)
                print(f"📦 Loaded FAISS index with {_faiss_index.ntotal} vectors")
            else:
                print(f"⚠️ FAISS dimension mismatch: {temp_index.d} vs {EMBEDDING_DIM}. Creating new index.")
                _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
                _chunk_metadata = []
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}. Creating new index.")
            _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            _chunk_metadata = []
    else:
        _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _chunk_metadata = []


def get_index_stats() -> dict:
    """Get statistics about the FAISS index."""
    _ensure_index()
    doc_ids = set(m["document_id"] for m in _chunk_metadata)
    return {
        "total_vectors": _faiss_index.ntotal,
        "total_chunks": len(_chunk_metadata),
        "total_documents": len(doc_ids),
        "dimension": EMBEDDING_DIM,
    }
