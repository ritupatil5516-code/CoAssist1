from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.base.response.schema import Response

from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

USE_HYBRID = os.getenv("USE_HYBRID","true").lower() in ("1","true","yes")
ALPHA = float(os.getenv("ALPHA","0.6"))
K_CANDIDATES = int(os.getenv("K_CANDIDATES","40"))
K_FINAL = int(os.getenv("K_FINAL","8"))
FRESHNESS_LAMBDA = float(os.getenv("FRESHNESS_LAMBDA","0.01"))
RERANKER = os.getenv("RERANKER","llm")

st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co‑pilot — LlamaIndex")

with st.sidebar:
    st.subheader("Data files in ./data")
    st.write(sorted(p.name for p in DATA_DIR.glob("*")))
    USE_HYBRID = st.toggle("Use Hybrid (FAISS+BM25)", value=USE_HYBRID)
    ALPHA = st.slider("Hybrid α", 0.0, 1.0, ALPHA, 0.05)
    K_CANDIDATES = st.number_input("Candidates N", 10, 200, K_CANDIDATES, 5)
    K_FINAL = st.number_input("Final K to LLM", 3, 20, K_FINAL, 1)
    FRESHNESS_LAMBDA = st.slider("Freshness λ (per day)", 0.0, 0.05, FRESHNESS_LAMBDA, 0.005)
    RERANKER = st.selectbox("Reranker", ["none","llm"], index=1 if RERANKER=="llm" else 0)

@st.cache_resource(show_spinner=False)
def _build():
    Settings.llm = make_llm()
    Settings.embed_model = make_embed_model()
    built = build_indexes(str(DATA_DIR))
    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    system_prompt = system + "\n\n" + style
    return built, system_prompt

built, SYSTEM = _build()

def answer(q: str):
    nodes = hybrid_with_freshness(built, q, alpha=ALPHA, lam=FRESHNESS_LAMBDA, kN=K_CANDIDATES) if USE_HYBRID             else built.vector_index.as_retriever(similarity_top_k=K_CANDIDATES).retrieve(q)
    if RERANKER == "llm":
        nodes = rerank_nodes(nodes, q, k=K_FINAL)
    else:
        nodes = nodes[:K_FINAL]

    # compose messages
    numbered = []
    for i, n in enumerate(nodes, 1):
        txt = n.node.get_content()[:1100]
        numbered.append(f"[{i}] {txt}")
    from llama_index.core.llms import ChatMessage
    messages = [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(role="user", content="Context:\n" + "\n\n".join(numbered) + f"\n\nQuestion: {q}")
    ]
    out = Settings.llm.chat(messages)  # type: ignore
    return out.message.content, nodes

if "history" not in st.session_state:
    st.session_state.history = [{"role":"assistant","content":"Hi! Ask about: interest this month/total, interest-causing transactions, last payment (date/amount), statement balance, account status, last posted transaction, top merchants, and spend this month/year."}]

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Type your question…")
if not q:
    st.stop()

st.session_state.history.append({"role":"user","content":q})
with st.chat_message("user"):
    st.markdown(q)

ans, used = answer(q)

with st.chat_message("assistant"):
    st.markdown(ans)

st.session_state.history.append({"role":"assistant","content":ans})

with st.expander("Retrieved context", expanded=False):
    for i,n in enumerate(used,1):
        st.markdown(f"**[{i}]** kind={n.node.metadata.get('kind')} ym={n.node.metadata.get('ym')} dt={n.node.metadata.get('dt_iso')}")
        st.write(n.node.get_content()[:1000] + ("..." if len(n.node.get_content())>1000 else ""))
