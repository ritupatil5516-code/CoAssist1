# core/indexes.py
from __future__ import annotations

from pathlib import Path
from typing import List, Any, Optional
from datetime import datetime, timezone
import json

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss  # pip install faiss-cpu

# We’ll use your project’s loader; adjust import path if your loader lives elsewhere.
from core.data import load_banking_data  # expected to return a BankingData object

# If you keep a LangChain BM25 wrapper, use it; otherwise fall back to LI BM25.
try:
    from core.bm25_langchain import LangChainBM25Retriever
    _HAS_LC_BM25 = True
except Exception:
    from llama_index.core.retrievers import BM25Retriever as _LIBM25
    _HAS_LC_BM25 = False


# ---- Policies used by retrieval/LLM (placed into node text & metadata) ----
EXCLUDE_TYPES = {"payment", "refund", "credit", "interest", "fee reversal"}


def _first(*vals):
    for v in vals:
        if v:
            return v
    return None


def _to_utc(dt_iso: Optional[str]) -> Optional[datetime]:
    if not dt_iso:
        return None
    try:
        return datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _ym(dt_iso: Optional[str]) -> Optional[str]:
    dt = _to_utc(dt_iso)
    return dt.strftime("%Y-%m") if dt else None


def _probe_embed_dim() -> int:
    """
    Robustly get embedding dimension from the configured embed model.
    IMPORTANT: Settings.embed_model must be set by app.py BEFORE calling build_indexes().
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


def _load_agreement_text(data_dir: Path) -> str:
    """
    Best-effort read of agreement.pdf. Returns "" if not available.
    Uses PyPDF2 if installed; falls back to plain bytes decode.
    """
    pdf_path = data_dir / "agreement.pdf"
    if not pdf_path.exists():
        return ""

    # Try PyPDF2
    try:
        from PyPDF2 import PdfReader  # pip install PyPDF2
        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        pass

    # Fallback: raw bytes (not great, but prevents crashes)
    try:
        return pdf_path.read_bytes().decode("latin-1", errors="ignore")
    except Exception:
        return ""


def build_indexes(data_dir: str) -> Built:
    """
    Build all nodes (accounts, statements, transactions, payments, agreement),
    construct FAISS index with the right dimension, and create a BM25 retriever.
    """
    data_path = Path(data_dir)
    b = load_banking_data(data_path)  # BankingData with .account_summary/.statements/.transactions/.payments

    nodes: List[TextNode] = []

    # ---- Accounts ----
    for a in b.account_summary:
        r = a.model_dump() if hasattr(a, "model_dump") else a  # allow dicts in dev
        text = (
            "ACCOUNT\n"
            "JSON:\n" + json.dumps(r, ensure_ascii=False)
        )
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "kind": "account",
                    "accountId": r.get("accountId"),
                    "accountStatus": r.get("accountStatus"),
                    "currentBalance": r.get("currentBalance"),
                    "availableCredit": r.get("availableCredit"),
                    "openedDateTime": r.get("openedDate"),
                    "closedDateTime": r.get("closedDate"),
                },
            )
        )

    # ---- Statements ----
    for s in b.statements:
        r = s.model_dump() if hasattr(s, "model_dump") else s
        # prefer closingDateTime for statement month
        dt_iso = _first(r.get("closingDateTime"), r.get("openingDateTime"), r.get("dueDate"))
        text = (
            "STATEMENT\n"
            "JSON:\n" + json.dumps(r, ensure_ascii=False)
        )
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "kind": "statement",
                    "dt_iso": dt_iso,
                    "ym": _ym(dt_iso),
                    "statementId": r.get("statementId"),
                    "dueDate": r.get("dueDate"),
                    "closingDateTime": r.get("closingDateTime"),
                    "interestCharged": r.get("interestCharged"),
                    "minimumPaymentDue": r.get("minimumPaymentDue"),
                    "newBalance": r.get("newBalance"),
                },
            )
        )

    # ---- Transactions ----
    for t in b.transactions:
        r = t.model_dump() if hasattr(t, "model_dump") else t

        # Canonical date: transactionDateTime → postingDateTime (ignore authDateTime)
        canonical_dt_iso = _first(r.get("transactionDateTime"), r.get("postingDateTime"))
        dtype = (r.get("displayTransactionType") or r.get("transactionType") or "").strip().lower()
        is_debit = str(r.get("debitCreditIndicator", "1")) == "1"
        is_spend_candidate = bool(is_debit and dtype not in EXCLUDE_TYPES)

        banner = (
            "DATE_POLICY: use transactionDateTime; fallback postingDateTime; ignore authDateTime.\n"
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
                    "ym": _ym(canonical_dt_iso),
                    "merchantName": r.get("merchantName") or r.get("merchantDescription") or r.get("merchantCategoryName"),
                    "amount": r.get("amount"),
                    "type": dtype,
                    "debit": is_debit,
                    "spend_candidate": is_spend_candidate,
                    "transactionId": r.get("transactionId"),
                },
            )
        )

    # ---- Payments ----
    for p in b.payments:
        r = p.model_dump() if hasattr(p, "model_dump") else p
        dt_iso = _first(r.get("paymentDateTime"), r.get("scheduledPaymentDateTime"))
        text = (
            "PAYMENT\n"
            "JSON:\n" + json.dumps(r, ensure_ascii=False)
        )
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "kind": "payment",
                    "dt_iso": dt_iso,
                    "ym": _ym(dt_iso),
                    "paymentId": r.get("paymentId"),
                    "amount": r.get("amount"),
                    "status": r.get("status") or r.get("paymentStatus"),
                },
            )
        )

    # ---- Agreement (rules) ----
    agr_text = _load_agreement_text(data_path)
    if agr_text:
        nodes.append(
            TextNode(
                text="AGREEMENT\n" + agr_text,
                metadata={"kind": "agreement"},
            )
        )

    if not nodes:
        raise AssertionError("No nodes created — check your data files in the /data folder.")

    # ---- Build FAISS with the correct dimension ----
    dim = _probe_embed_dim()              # IMPORTANT: Settings.embed_model must be set by app.py beforehand
    faiss_index = faiss.IndexFlatIP(dim)  # cosine via IP when embeddings are normalized
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    v_index = VectorStoreIndex(nodes, storage_context=storage_context)

    # ---- BM25 retriever ----
    if _HAS_LC_BM25:
        bm25 = LangChainBM25Retriever.from_nodes(nodes, similarity_top_k=50)
    else:
        bm25 = _LIBM25.from_defaults(nodes, similarity_top_k=50)

    return Built(nodes=nodes, vector_index=v_index, bm25=bm25)