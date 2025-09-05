from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.retrievers import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss, numpy as np

from core.llm import make_llm, make_embed_model
from core.data import load_bundle, load_agreement_text

@dataclass
class Built:
    nodes: List[TextNode]
    vector_index: VectorStoreIndex
    bm25: BM25Retriever

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
        r = t.model_dump()
        dt = r.get("postingDateTime") or r.get("transactionDateTime")
        add("transaction", r, r.get("ym"), dt)

    for p in b.payments:
        r = p.model_dump()
        dt = r.get("paymentDateTime") or r.get("scheduledPaymentDateTime")
        add("payment", r, r.get("ym"), dt)

    # agreement
    agr = load_agreement_text(data_dir)
    if agr:
        nodes.append(TextNode(text="AGREEMENT " + agr, metadata={"kind": "agreement"}))

    # build FAISS vector index
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(1536))  # dim will auto-fit from first embed
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    v_index = VectorStoreIndex(nodes, storage_context=storage_context)

    # BM25 retriever from nodes
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=50)

    return Built(nodes=nodes, vector_index=v_index, bm25=bm25)
