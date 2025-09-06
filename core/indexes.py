# core/indexes.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timezone

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss  # <-- make sure this is in requirements

EXCLUDE_TYPES = {"payment", "refund", "credit", "interest", "fee reversal"}

def _first(*vals):
    for v in vals:
        if v:
            return v
    return None

def _ym_parts(dt_iso: Optional[str]):
    if not dt_iso:
        return None, None
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt, dt.strftime("%Y-%m")
    except Exception:
        return None, None

def _probe_embed_dim() -> int:
    """
    Robustly get embedding dimension without relying on private APIs.
    We call the public text-embedding once and use its length.
    """
    emb = Settings.embed_model.get_text_embedding("dimension-probe")
    if not emb or not isinstance(emb, list):
        raise RuntimeError("Failed to probe embedding dimension (empty embedding).")
    return len(emb)

class Built:
    def __init__(self, nodes: List[TextNode], vector_index: VectorStoreIndex, bm25: Any):
        self.nodes = nodes
        self.vector_index = vector_index
        self.bm25 = bm25

def build_indexes(data_dir: str) -> Built:
    data_path = Path(data_dir)

    # 1) Load your BankingData here (unchanged) → produces lists: accounts, statements, transactions, payments
    #    ... existing loading code ...

    nodes: List[TextNode] = []

    # 2) Build nodes (keep what you already had for account/statement/payment).
    #    Only the TRANSACTION part shown here for clarity.
    for t in b.transactions:
        r = t.model_dump()
        canonical_dt_iso = _first(r.get("transactionDateTime"), r.get("postingDateTime"))
        dt_obj, ym = _ym_parts(canonical_dt_iso)

        dtype = (r.get("displayTransactionType") or r.get("transactionType") or "").strip().lower()
        is_debit = str(r.get("debitCreditIndicator", "1")) == "1"
        is_spend_candidate = bool(is_debit and dtype not in EXCLUDE_TYPES)

        banner = (
            "DATE_POLICY: use transactionDateTime; fallback postingDateTime; "
            "ignore authDateTime for time windows/latest/spend.\n"
            "SPEND_POLICY: only debit/outflow; exclude payment/refund/credit/interest.\n"
        )
        text = (
            "TRANSACTION\n" + banner +
            f"transactionDateTime={r.get('transactionDateTime')}\n"
            f"postingDateTime={r.get('postingDateTime')}\n"
            f"authDateTime={r.get('authDateTime')}\n"
            "JSON:\n" + json.dumps(r, ensure_ascii=False)
        )

        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "kind": "transaction",
                    "dt_iso": canonical_dt_iso,
                    "ym": ym,
                    "merchantName": r.get("merchantName") or r.get("merchantDescription") or r.get("merchantCategoryName"),
                    "amount": r.get("amount"),
                    "type": dtype,
                    "debit": is_debit,
                    "spend_candidate": is_spend_candidate,
                },
            )
        )

    # 3) FAISS: create index with the correct dimension
    dim = _probe_embed_dim()              # ← critical: make sure dimension matches your embed model
    faiss_index = faiss.IndexFlatIP(dim)  # IP for cosine (with normalized embeddings)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    v_index = VectorStoreIndex(nodes, storage_context=storage_context)

    # 4) BM25 as before (your LangChainBM25 or LlamaIndex BM25)
    from core.bm25_langchain import LangChainBM25Retriever  # or your existing BM25 wrapper
    bm25 = LangChainBM25Retriever.from_nodes(nodes, similarity_top_k=50)

    return Built(nodes=nodes, vector_index=v_index, bm25=bm25)