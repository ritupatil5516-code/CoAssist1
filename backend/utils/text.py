from __future__ import annotations
import re
from typing import List

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
