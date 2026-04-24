"""Tests for utility functions."""

from app.utils import (
    generate_id, format_timestamp, parse_timestamp,
    get_file_extension, classify_file_type, is_allowed_file,
    sanitize_filename, truncate_text,
)


class TestGenerateId:
    def test_returns_string(self):
        assert isinstance(generate_id(), str)

    def test_unique_ids(self):
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestFormatTimestamp:
    def test_seconds_only(self):
        assert format_timestamp(45) == "00:45"

    def test_minutes_seconds(self):
        assert format_timestamp(125) == "02:05"

    def test_hours(self):
        assert format_timestamp(3661) == "01:01:01"

    def test_zero(self):
        assert format_timestamp(0) == "00:00"


class TestParseTimestamp:
    def test_mm_ss(self):
        assert parse_timestamp("02:30") == 150.0

    def test_hh_mm_ss(self):
        assert parse_timestamp("01:02:30") == 3750.0

    def test_seconds_string(self):
        assert parse_timestamp("45") == 45.0


class TestFileClassification:
    def test_pdf(self):
        assert classify_file_type("document.pdf") == "pdf"

    def test_audio_mp3(self):
        assert classify_file_type("song.mp3") == "audio"

    def test_audio_wav(self):
        assert classify_file_type("audio.wav") == "audio"

    def test_video_mp4(self):
        assert classify_file_type("video.mp4") == "video"

    def test_video_mkv(self):
        assert classify_file_type("movie.mkv") == "video"

    def test_unknown(self):
        assert classify_file_type("file.xyz") == "unknown"

    def test_get_extension(self):
        assert get_file_extension("test.PDF") == "pdf"

    def test_no_extension(self):
        assert get_file_extension("noext") == ""


class TestAllowedFile:
    def test_pdf_allowed(self):
        assert is_allowed_file("doc.pdf") is True

    def test_mp3_allowed(self):
        assert is_allowed_file("audio.mp3") is True

    def test_exe_not_allowed(self):
        assert is_allowed_file("virus.exe") is False

    def test_txt_not_allowed(self):
        assert is_allowed_file("notes.txt") is False


class TestSanitizeFilename:
    def test_removes_special_chars(self):
        assert "<" not in sanitize_filename('file<name>.pdf')

    def test_truncates_long_names(self):
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 200


class TestTruncateText:
    def test_short_text(self):
        assert truncate_text("hello", 10) == "hello"

    def test_long_text(self):
        result = truncate_text("a" * 600, 500)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")
