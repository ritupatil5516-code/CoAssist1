from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

from backend.utils.text import dir_mtime_fingerprint
from backend.rag.corpus import build_corpus
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store
from backend.rag.hybrid import hybrid_merge
from backend.llm.client import LLMClient
from backend.rerankers.llm_reranker import rerank_with_llm
from backend.rerankers.bge_reranker import rerank_with_bge
from backend.rag.types import Chunk

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "data")).resolve()

REQUIRED = [
    "account_summary.json",
    "payments.json",
    "statements.json",
    "transactions.json",
    "agreement.pdf",
]

st.set_page_config(page_title="Banking Co-Pilot — Hybrid FAISS + Re-rank", layout="wide")
st.title("Banking Co-Pilot — Hybrid Retrieval + Re-ranking")
st.caption("Semantic (FAISS) + BM25 hybrid, optional re-ranking (LLM or BGE), metadata tagging, freshness-aware.")

with st.sidebar:
    st.subheader("Data path & files")
    st.code(str(DATA_DIR))
    exists = DATA_DIR.exists()
    st.write("📁 Data directory exists:", exists)
    if exists:
        files = sorted(p.name for p in DATA_DIR.glob("*"))
        st.write("Files found:", files)
    else:
        st.error("Data directory not found. Set DATA_DIR env var or create ./data.")
    missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
    if missing:
        st.error(f"Missing required files: {missing}")
        st.stop()

    st.subheader("Retrieval settings")
    use_hybrid = st.toggle("Use Hybrid (FAISS + BM25)", value=True)
    alpha = st.slider("Hybrid weight α (FAISS share)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    k_candidates = st.number_input("Candidates (top-N) before re-rank", min_value=5, max_value=100, value=30, step=5)
    k_final = st.number_input("Final top-K to Llama", min_value=3, max_value=20, value=8, step=1)

    st.subheader("Re-ranking")
    use_rerank = st.toggle("Enable re-ranking", value=True)
    reranker = st.selectbox("Reranker", options=["LLM (chat JSON scoring)", "BGE (FlagEmbedding)"], index=0)

# Freshness-aware caching key based on data dir fingerprint
fingerprint = dir_mtime_fingerprint(str(DATA_DIR))

@st.cache_resource(show_spinner=False)
def _build_resources(_fingerprint: str):
    chunks = build_corpus(str(DATA_DIR))
    vec = FAISSStore(chunks)
    bm25 = BM25Store(chunks)
    return chunks, vec, bm25

chunks, vec_store, bm25_store = _build_resources(fingerprint)

def retrieve(query: str) -> list[tuple[Chunk, float]]:
    vec = vec_store.search(query, k=int(k_candidates))
    if use_hybrid:
        lex = bm25_store.search(query, k=int(k_candidates))
        merged = hybrid_merge(vec, lex, alpha=float(alpha), k=int(k_candidates))
    else:
        merged = vec
    if use_rerank:
        candidates = [c for c, _ in merged]
        if reranker.startswith("LLM"):
            reranked = rerank_with_llm(query, candidates)
        else:
            reranked = rerank_with_bge(query, candidates)
        return reranked[:int(k_final)]
    else:
        return merged[:int(k_final)]

q = st.text_input("Ask a question", "Why was interest charged last month?")
if st.button("Ask"):
    results = retrieve(q)
    with st.expander("Retrieved Context (with metadata)", expanded=False):
        for i, (chunk, score) in enumerate(results, 1):
            st.markdown(f"**{i}.** [{chunk.source}] score={score:.3f} meta={chunk.meta}")
            st.write(chunk.text[:800] + ("..." if len(chunk.text)>800 else ""))

    llm = LLMClient()
    context = "\n\n".join([c.text for c, _ in results])
    system = "You are a precise banking assistant. Only use the provided context. Show math when applicable."
    user = "Context:\n" + context + f"\n\nQuestion: {q}\nAnswer concisely and cite which context lines you used."
    ans = llm.chat(system, user)
    st.markdown("### Llama Answer")
    st.write(ans)
else:
    st.info("Enter a question and click **Ask**. Tune hybrid α and re-ranking above.")
