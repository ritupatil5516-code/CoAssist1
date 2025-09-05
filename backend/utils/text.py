import re, hashlib
from typing import List
from pathlib import Path

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = normalize_ws(text)
    if not text: return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n: break
        start = max(0, end - overlap)
    return chunks

def dir_mtime_fingerprint(path: str) -> str:
    p = Path(path)
    if not p.exists(): return "missing"
    parts = []
    for child in p.iterdir():
        try:
            stat = child.stat()
            parts.append(f"{child.name}:{int(stat.st_mtime)}:{stat.st_size}")
        except Exception: continue
    return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()
