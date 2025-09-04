from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

from backend.rag.corpus import build_corpus
from backend.rag.faiss_store import FAISSStore
from backend.llm.client import LLMClient

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "data")).resolve()

REQUIRED = [
    "account_summary.json",
    "payments.json",
    "statements.json",
    "transactions.json",
    "agreement.pdf",
]

st.set_page_config(page_title="Banking Co-Pilot (FAISS + Llama)", layout="wide")
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

@st.cache_resource(show_spinner=False)
def _store():
    corpus = build_corpus(str(DATA_DIR))
    return FAISSStore(corpus)

store = _store()

q = st.text_input("Ask a question", "How was interest calculated last month?")
if st.button("Ask"):
    ctx = [c for c, _ in store.search(q, k=6)]
    with st.expander("Retrieved Context", expanded=False):
        for i, c in enumerate(ctx, 1):
            st.write(f"{i}. {c[:600]}...")

    llm = LLMClient()
    system = "You are a precise banking assistant. Only use the provided context. Show math when applicable."
    user = "Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\nAnswer concisely and cite which context lines you used."
    ans = llm.chat(system, user)
    st.markdown("### Llama Answer")
    st.write(ans)
else:
    st.info("Enter a question and click Ask. Set DATA_DIR if files are not under ./data.")
