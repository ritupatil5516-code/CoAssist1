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
st.title("Banking Co-Pilot — FAISS + Llama")
st.caption("Answers from agreement.pdf + JSON data.")

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

fingerprint = dir_mtime_fingerprint(str(DATA_DIR))

@st.cache_resource(show_spinner=False)
def _build_resources(_fingerprint: str):
    chunks = build_corpus(str(DATA_DIR))     # robust Chunk list
    vec = FAISSStore(chunks)
    bm25 = BM25Store(chunks)
    return chunks, vec, bm25

chunks, vec_store, bm25_store = _build_resources(fingerprint)

st.subheader("Ask")
q = st.text_input("Question", "Why was interest charged last month?")

col = st.columns(3)
with col[0]:
    use_hybrid = st.toggle("Use Hybrid (FAISS + BM25)", value=True)
with col[1]:
    alpha = st.slider("Hybrid weight α", 0.0, 1.0, 0.6, 0.05)
with col[2]:
    k_final = st.number_input("Top-K", min_value=3, max_value=20, value=8, step=1)

if st.button("Ask"):
    vec_res = vec_store.search(q, k=30)
    if use_hybrid:
        lex_res = bm25_store.search(q, k=30)
        merged = hybrid_merge(vec_res, lex_res, alpha=alpha, k=30)
    else:
        merged = vec_res
    results = merged[:int(k_final)]

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
    st.info("Enter a question and click **Ask**. Set DATA_DIR if files are not under ./data.")