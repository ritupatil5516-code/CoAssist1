from __future__ import annotations
from typing import List
from pathlib import Path
from collections import defaultdict

from backend.loaders.json_loader import load_all, flatten_for_rag
from backend.loaders.pdf_loader import extract_pdf_text
from backend.utils.text import chunk_text
from backend.rag.types import Chunk

def build_corpus(data_dir: str) -> List[Chunk]:
    d = Path(data_dir).resolve()
    docs: List[Chunk] = []

    data = load_all(str(d))
    flat = flatten_for_rag(data)

    # Push JSON/statement/transaction/payment items
    for row in flat:
        docs.append(Chunk(text=row["text"], source=row["source"], meta=row.get("meta", {})))

    # --- aggregates using meta['raw'] ---
    interest_stmt = defaultdict(float)
    interest_tx = defaultdict(float)

    for row in flat:
        if row["source"] == "statement":
            raw = row["meta"].get("raw", {})
            ym = row["meta"].get("ym") or raw.get("ym")
            val = raw.get("interestCharged")
            if ym and isinstance(val, (int, float)):
                interest_stmt[ym] += float(val)

    for row in flat:
        if row["source"] == "transaction":
            raw = row["meta"].get("raw", {})
            ym = row["meta"].get("ym") or raw.get("ym")
            # consider any of these as interest
            is_interest = bool(
                raw.get("interestFlag") or
                (str(raw.get("transactionType","")).upper() == "INTEREST") or
                (str(raw.get("displayTransactionType","")).lower() == "interest_charged") or
                ("interest" in str(raw.get("merchantName","")).lower())
            )
            if ym and is_interest:
                amt = raw.get("amount")
                if isinstance(amt, (int, float)):
                    interest_tx[ym] += abs(float(amt))

    # Emit aggregates
    for ym, val in sorted(interest_stmt.items()):
        docs.append(Chunk(
            text=f"AGGREGATE ym={ym} interest_from_statements_total={val:.2f}",
            source="aggregate",
            meta={"ym": ym, "metric": "interest_from_statements_total"}
        ))
    for ym, val in sorted(interest_tx.items()):
        docs.append(Chunk(
            text=f"AGGREGATE ym={ym} interest_from_interest_transactions_total={val:.2f}",
            source="aggregate",
            meta={"ym": ym, "metric": "interest_from_interest_transactions_total"}
        ))

    # Latest / overall helpers (keep if you added earlier)
    all_yms = set(interest_stmt.keys()) | set(interest_tx.keys())
    if all_yms:
        latest_ym = sorted(all_yms)[-1]
        docs.append(Chunk(text=f"AGGREGATE latest_ym={latest_ym}", source="aggregate", meta={"latest_ym": latest_ym}))
        docs.append(Chunk(text=f"AGGREGATE overall_interest_from_statements_total={sum(interest_stmt.values()):.2f}",
                          source="aggregate", meta={"metric":"overall_interest_from_statements_total"}))
        docs.append(Chunk(text=f"AGGREGATE overall_interest_from_interest_transactions_total={sum(interest_tx.values()):.2f}",
                          source="aggregate", meta={"metric":"overall_interest_from_interest_transactions_total"}))

    # Agreement
    pdf = d / "agreement.pdf"
    if pdf.exists():
        pdf_text = extract_pdf_text(str(pdf))
        for ch in chunk_text(pdf_text, 1000, 200):
            docs.append(Chunk(text="AGREEMENT: " + ch, source="agreement", meta={"file":"agreement.pdf"}))

    return docs