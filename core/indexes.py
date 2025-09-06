from __future__ import annotations
from typing import List
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from core.llm import make_llm, make_embed_model
from core.data import load_bundle, load_agreement_text
from core.retrievers.bm25_langchain import LangChainBM25Retriever

@dataclass
class Built:
    nodes: List[TextNode]
    vector_index: VectorStoreIndex
    bm25: LangChainBM25Retriever

def _first(*vals):
    for v in vals:
        if v:
            return v
    return None


def build_indexes(data_dir: str) -> Built:
    Settings.llm = make_llm()
    Settings.embed_model = make_embed_model()

    b = load_bundle(data_dir)
    nodes: List[TextNode] = []

    def add(kind: str, raw: dict, ym: str | None, dt_iso: str | None):
        meta = {"kind": kind, "ym": ym, "dt_iso": dt_iso, "raw": raw}
        txt = f"{kind.upper()} " + str(raw)
        nodes.append(TextNode(text=txt, metadata=meta))

    for a in b.account_summary:
        r = a.model_dump()
        add("account", r, r.get("ym"), None)

    for s in b.statements:
        r = s.model_dump()
        dt = r.get("closingDateTime") or r.get("openingDateTime") or r.get("dueDate")
        add("statement", r, r.get("ym"), dt)

    for t in b.transactions:
        r = t.model_dump()  # keep all fields

        # ---- canonical date for retrieval/freshness ----
        canonical_dt = _first(
            r.get("transactionDateTime"),
            r.get("postingDateTime"),
        )
        ym = r.get("ym")

        # ---- short banner to steer the LLM (keeps all fields below) ----
        banner = (
            "DATE_POLICY: use transactionDateTime; fallback postingDateTime; "
            "ignore authDateTime for time windows/latest/spend.\n"
        )

        # full JSON is still included for completeness
        text = (
                "TRANSACTION\n"
                + banner
                + f"transactionDateTime={r.get('transactionDateTime')}\n"
                + f"postingDateTime={r.get('postingDateTime')}\n"
                + f"authDateTime={r.get('authDateTime')}\n"
                + "JSON:\n"
                + json.dumps(r, ensure_ascii=False)
        )

        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "kind": "transaction",
                    "dt_iso": canonical_dt,  # used by freshness
                    "ym": ym,
                    "accountId": r.get("accountId"),
                    "merchantName": _first(
                        r.get("merchantName"),
                        r.get("merchantDescription"),
                        r.get("merchantCategoryName"),
                    ),
                    "amount": r.get("amount"),
                    "displayTransactionType": _first(
                        r.get("displayTransactionType"),
                        r.get("transactionType"),
                    ),
                },
            )
        )

    for p in b.payments:
        r = p.model_dump()
        dt = r.get("paymentDateTime") or r.get("scheduledPaymentDateTime")
        add("payment", r, r.get("ym"), dt)

    agr = load_agreement_text(data_dir)
    if agr:
        nodes.append(TextNode(text="AGREEMENT " + agr, metadata={"kind": "agreement"}))

    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(1536))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    v_index = VectorStoreIndex(nodes, storage_context=storage_context)

    bm25 = LangChainBM25Retriever.from_nodes(nodes=nodes, similarity_top_k=50)

    return Built(nodes=nodes, vector_index=v_index, bm25=bm25)
