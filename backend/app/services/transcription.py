"""Audio/video transcription service using Groq Whisper API."""

import os
import subprocess
import tempfile
from pathlib import Path
from groq import Groq
from app.config import settings
from app.utils import generate_id


def get_groq_client() -> Groq:
    """Get a Groq API client."""
    return Groq(api_key=settings.GROQ_API_KEY)


def extract_audio_from_video(video_path: str | Path, output_path: str | None = None) -> str:
    """
    Extract audio from a video file using FFmpeg.
    Returns the path to the extracted audio file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = str(video_path.with_suffix(".mp3"))

    cmd = [
        settings.FFMPEG_PATH,
        "-i", str(video_path),
        "-vn",                    # no video
        "-acodec", "libmp3lame",  # MP3 codec
        "-ab", "128k",            # 128kbps bitrate
        "-ar", "16000",           # 16kHz sample rate (good for speech)
        "-y",                     # overwrite output
        output_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

    return output_path


def transcribe_audio(file_path: str | Path) -> dict:
    """
    Transcribe an audio file using Groq Whisper API.

    Returns: {
        "text": "full transcription",
        "segments": [{"start": 0.0, "end": 5.0, "text": "..."}],
        "duration": 120.5
    }
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    client = get_groq_client()

    # Check file size — Groq has 25MB limit
    file_size = file_path.stat().st_size
    if file_size > 25 * 1024 * 1024:
        # Split into smaller chunks if too large
        return _transcribe_large_file(file_path, client)

    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path.name, audio_file),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # Parse response
    segments = []
    if hasattr(transcription, "segments") and transcription.segments:
        for seg in transcription.segments:
            segments.append({
                "start": seg.get("start", seg.start) if hasattr(seg, "start") else seg.get("start", 0),
                "end": seg.get("end", seg.end) if hasattr(seg, "end") else seg.get("end", 0),
                "text": seg.get("text", seg.text) if hasattr(seg, "text") else seg.get("text", ""),
            })

    return {
        "text": transcription.text,
        "segments": segments,
        "duration": getattr(transcription, "duration", 0.0),
    }


def _transcribe_large_file(file_path: Path, client: Groq) -> dict:
    """Handle files larger than 25MB by splitting with FFmpeg."""
    # Split into 10-minute chunks
    chunk_duration = 600  # seconds
    chunks_dir = Path(tempfile.mkdtemp())
    chunk_pattern = str(chunks_dir / "chunk_%03d.mp3")

    cmd = [
        settings.FFMPEG_PATH,
        "-i", str(file_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-ar", "16000",
        "-y",
        chunk_pattern,
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Transcribe each chunk
    all_text = []
    all_segments = []
    time_offset = 0.0

    chunk_files = sorted(chunks_dir.glob("chunk_*.mp3"))
    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(chunk_file.name, f),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        all_text.append(transcription.text)

        if hasattr(transcription, "segments") and transcription.segments:
            for seg in transcription.segments:
                start = (seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0))
                end = (seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0))
                text = (seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""))
                all_segments.append({
                    "start": start + time_offset,
                    "end": end + time_offset,
                    "text": text,
                })

        chunk_duration_actual = getattr(transcription, "duration", chunk_duration)
        time_offset += chunk_duration_actual

        # Clean up chunk
        chunk_file.unlink(missing_ok=True)

    # Clean up temp dir
    chunks_dir.rmdir()

    return {
        "text": " ".join(all_text),
        "segments": all_segments,
        "duration": time_offset,
    }


def segments_to_chunks(document_id: str, segments: list[dict]) -> list[dict]:
    """
    Convert transcription segments into text chunks for embedding.
    Groups consecutive segments into larger chunks.
    """
    if not segments:
        return []

    chunks = []
    chunk_index = 0
    current_texts = []
    current_start = segments[0]["start"]
    word_count = 0

    for seg in segments:
        seg_words = seg["text"].split()
        current_texts.append(seg["text"])
        word_count += len(seg_words)

        # Create a chunk when we have ~200 words
        if word_count >= 200:
            chunks.append({
                "id": generate_id(),
                "document_id": document_id,
                "text": " ".join(current_texts).strip(),
                "chunk_index": chunk_index,
                "page_number": None,
                "start_time": current_start,
                "end_time": seg["end"],
            })
            chunk_index += 1
            current_texts = []
            word_count = 0
            # Next chunk starts at next segment
            current_start = seg["end"]

    # Remaining text
    if current_texts:
        chunks.append({
            "id": generate_id(),
            "document_id": document_id,
            "text": " ".join(current_texts).strip(),
            "chunk_index": chunk_index,
            "page_number": None,
            "start_time": current_start,
            "end_time": segments[-1]["end"],
        })

    return chunks
