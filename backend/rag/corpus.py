from __future__ import annotations
from typing import List, Any, Dict
from pathlib import Path
from backend.loaders.json_loader import flatten_for_rag, load_all
from backend.loaders.pdf_loader import extract_pdf_text
from backend.utils.text import chunk_text
from backend.rag.types import Chunk

def _to_chunk(row: Any) -> Chunk:
    if isinstance(row, dict):
        text = str(row.get("text", ""))
        source = str(row.get("source", "unknown"))
        meta: Dict[str, Any] = row.get("meta", {}) or {}
    else:
        text, source, meta = str(row), "unknown", {}
    return Chunk(text=text, source=source, meta=meta)

def build_corpus(data_dir: str) -> List[Chunk]:
    d = Path(data_dir).resolve()
    docs: List[Chunk] = []

    data = load_all(str(d))
    for row in flatten_for_rag(data):
        docs.append(_to_chunk(row))

    pdf_path = d / "agreement.pdf"
    pdf_text = extract_pdf_text(str(pdf_path)) if pdf_path.exists() else ""
    if pdf_text:
        for ch in chunk_text(pdf_text, chunk_size=1000, overlap=200):
            docs.append(Chunk(text="AGREEMENT: " + ch, source="agreement", meta={"file":"agreement.pdf"}))
    else:
        docs.append(Chunk(
            text=("AGREEMENT SUMMARY: Interest uses daily balance method (APR/365). "
                  "Grace period applies only if prior balance was paid in full."),
            source="agreement_summary",
            meta={"file":"agreement.pdf","generated":True}
        ))
    return docs