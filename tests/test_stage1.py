import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_extraction import extract_pages, extract_text
from src.chunking import chunk_pages, chunk_text, Chunk


def test_chunk_text_basic():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_pages_preserves_metadata():
    from src.pdf_extraction import PageText

    pages = [PageText(filename="doc.pdf", page_number=3, text="hello world " * 100)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=10)
    assert chunks
    assert all(c.filename == "doc.pdf" for c in chunks)
    assert all(c.page_number == 3 for c in chunks)
    assert chunks[0].chunk_index == 0


def test_chunk_validation():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_extract_pages_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError):
        extract_pages("does_not_exist.pdf")
