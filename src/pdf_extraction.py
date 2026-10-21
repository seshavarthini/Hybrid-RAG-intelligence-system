from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf 


@dataclass
class PageText:
    """Text extracted from a single PDF page."""
    filename: str
    page_number: int  # 1-based
    text: str


def extract_pages(pdf_path: str | Path) -> List[PageText]:
    """Extract text from every page of a PDF, preserving filename and page number."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: List[PageText] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(PageText(filename=path.name, page_number=index, text=text))
    return pages


def extract_text(pdf_path: str | Path) -> str:
    """Convenience helper returning the full document text (pages joined by form feed)."""
    pages = extract_pages(pdf_path)
    return "\f".join(p.text for p in pages)
