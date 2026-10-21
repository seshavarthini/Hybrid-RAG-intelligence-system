from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.pdf_extraction import PageText


@dataclass
class Chunk:
    filename: str
    page_number: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Split page text into overlapping character-based chunks, preserving source info."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: List[Chunk] = []
    for page in pages:
        text = (page.text or "").strip()
        if not text:
            continue
        step = chunk_size - overlap
        start = 0
        idx = 0
        while start < len(text):
            piece = text[start : start + chunk_size]
            if piece.strip():
                chunks.append(
                    Chunk(
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_index=idx,
                        text=piece,
                    )
                )
                idx += 1
            start += step
            if start >= len(text):
                break
    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple chunking returning only text strings (no source metadata)."""
    pages = [PageText(filename="", page_number=0, text=text)]
    return [c.text for c in chunk_pages(pages, chunk_size, overlap)]
