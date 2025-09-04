from __future__ import annotations
from typing import List
import os
from pathlib import Path
from backend.loaders.json_loader import flatten_for_rag, load_all
from backend.loaders.pdf_loader import extract_pdf_text
from backend.utils.text import chunk_text

def build_corpus(data_dir: str) -> List[str]:
    d = Path(data_dir).resolve()
    docs: List[str] = []

    # JSON docs
    data = load_all(str(d))
    docs.extend(flatten_for_rag(data))

    # Agreement PDF
    pdf_path = d / "agreement.pdf"
    pdf_text = extract_pdf_text(str(pdf_path)) if pdf_path.exists() else ""
    if pdf_text:
        for ch in chunk_text(pdf_text, chunk_size=1000, overlap=200):
            docs.append("AGREEMENT: " + ch)
    else:
        docs.append("AGREEMENT SUMMARY: Interest uses daily balance method (APR/365). "
                    "Grace period applies only if prior balance was paid in full.")
    return docs
