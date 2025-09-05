from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict
import re
from backend.loaders.json_loader import load_all, flatten_for_rag
from backend.loaders.pdf_loader import extract_pdf_text
from backend.utils.text import chunk_text
from backend.rag.types import Chunk

def _num_from(label: str, text: str) -> float | None:
    m = re.search(rf"{label}=([-,\d\.]+)", text)
    if not m: return None
    try: return float(m.group(1).replace(",", ""))
    except Exception: return None

def build_corpus(data_dir: str) -> List[Chunk]:
    d = Path(data_dir).resolve()
    docs: List[Chunk] = []

    data = load_all(str(d))
    flat = flatten_for_rag(data)
    for row in flat:
        docs.append(Chunk(text=row["text"], source=row["source"], meta=row.get("meta", {})))

    # Aggregates
    interest_stmt = defaultdict(float)
    interest_tx = defaultdict(float)
    for row in flat:
        if row["source"] == "statement":
            ym = row["meta"].get("ym"); 
            if not ym: continue
            val = _num_from("interestCharged", row["text"])
            if val is not None: interest_stmt[ym] += val
    for row in flat:
        if row["source"] == "transaction" and row["meta"].get("interest"):
            ym = row["meta"].get("ym")
            if not ym: continue
            val = _num_from("amount", row["text"])
            if val is not None: interest_tx[ym] += abs(val)
    for ym, val in sorted(interest_stmt.items()):
        docs.append(Chunk(text=f"AGGREGATE ym={ym} interest_from_statements_total={val:.2f} (sum of statement interestCharged for ym)", source="aggregate", meta={"ym": ym, "metric": "interest_from_statements_total"}))
    for ym, val in sorted(interest_tx.items()):
        docs.append(Chunk(text=f"AGGREGATE ym={ym} interest_from_interest_transactions_total={val:.2f} (sum of INTEREST transactions for ym)", source="aggregate", meta={"ym": ym, "metric": "interest_from_interest_transactions_total"}))

    # Agreement
    pdf = d / "agreement.pdf"
    if pdf.exists():
        pdf_text = extract_pdf_text(str(pdf))
        for ch in chunk_text(pdf_text, 1000, 200):
            docs.append(Chunk(text="AGREEMENT: " + ch, source="agreement", meta={"file":"agreement.pdf"}))

    # Schema cheat sheet (machine-readable)
    schema_txt = (
        "SCHEMA: STATEMENT{ym, interestCharged, endingBalance, minimumAmountDue, totalAmountDue}; "
        "TRANSACTION{ym, amount, interestFlag, description, type}; "
        "PAYMENT{ym, amount, status}; "
        "PREFER: AGGREGATE.interest_from_statements_total > STATEMENT.interestCharged > TRANSACTION[interestFlag=true]"
    )
    docs.append(Chunk(text=schema_txt, source="schema", meta={"kind": "schema"}))

    return docs
