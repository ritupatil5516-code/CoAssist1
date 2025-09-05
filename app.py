# app.py
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# project modules
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes

# ─────────────────────────────────────────────────────────────
# Page + env
# ─────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Defaults / controls
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() in ("1", "true", "yes")
ALPHA = float(os.getenv("ALPHA", "0.6"))
K_CANDIDATES = int(os.getenv("K_CANDIDATES", "40"))
K_FINAL = int(os.getenv("K_FINAL", "8"))
FRESHNESS_LAMBDA = float(os.getenv("FRESHNESS_LAMBDA", "0.01"))
RERANKER = os.getenv("RERANKER", "llm")  # "llm" | "none"

with st.sidebar:
    st.subheader("Data files (./data)")
    st.write(sorted(p.name for p in DATA_DIR.glob("*")))
    USE_HYBRID = st.toggle("Use Hybrid (FAISS + BM25)", value=USE_HYBRID)
    ALPHA = st.slider("Hybrid α (FAISS share)", 0.0, 1.0, ALPHA, 0.05)
    K_CANDIDATES = st.number_input("Candidates N", 10, 200, K_CANDIDATES, 5)
    K_FINAL = st.number_input("Final K to LLM", 3, 20, K_FINAL, 1)
    FRESHNESS_LAMBDA = st.slider("Freshness λ/day", 0.0, 0.05, FRESHNESS_LAMBDA, 0.005)
    RERANKER = st.selectbox("Reranker", ["none", "llm"], index=1 if RERANKER == "llm" else 0)

# ─────────────────────────────────────────────────────────────
# Build (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build():
    Settings.llm = make_llm()
    Settings.embed_model = make_embed_model()
    built = build_indexes(str(DATA_DIR))

    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    system_prompt = system + "\n\n" + style

    # Quick BM25 check
    with st.sidebar:
        st.markdown("### Startup checks")
        try:
            nodes = built.bm25.retrieve("statement balance")
            st.success(f"BM25 OK — {len(nodes)} nodes for 'statement balance'.")
        except Exception as e:
            st.error(f"BM25 check failed: {e}")

    return built, system_prompt

built, SYSTEM = _build()

# ─────────────────────────────────────────────────────────────
# Session state (history + pending)
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": (
            "Hi! Ask about: **interest this month/total**, **interest-causing transactions**, "
            "**last payment** (date/amount), **statement balance**, **account status**, "
            "**last posted transaction**, **top merchants**, **spend this month/year**."
        ),
    }]

# pending holds the last question while we compute
# {"q": str, "assistant_index": int}  where assistant_index points to the thinking bubble in history
if "pending" not in st.session_state:
    st.session_state.pending = None

# ─────────────────────────────────────────────────────────────
# Render chat history (once per run)
# ─────────────────────────────────────────────────────────────
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ─────────────────────────────────────────────────────────────
# If we have a pending question, do the heavy work NOW (Pass B)
# This runs AFTER the "🤔 Thinking…" bubble is already visible.
# ─────────────────────────────────────────────────────────────
if st.session_state.pending:
    q = st.session_state.pending["q"]
    idx = st.session_state.pending["assistant_index"]

    # Retrieval
    if USE_HYBRID:
        candidates = hybrid_with_freshness(built, q, alpha=ALPHA,
                                           lam=FRESHNESS_LAMBDA, kN=K_CANDIDATES)
    else:
        candidates = built.vector_index.as_retriever(
            similarity_top_k=K_CANDIDATES).retrieve(q)

    nodes = candidates[:K_FINAL]
    if RERANKER == "llm":
        nodes = rerank_nodes(candidates, q, k=K_FINAL)

    # Build compact context and ask LLM (single-shot for robustness)
    numbered = []
    for i, n in enumerate(nodes, 1):
        txt = n.node.get_content()[:1100]
        numbered.append(f"[{i}] {txt}")

    messages = [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(role="user", content="Context:\n" + "\n\n".join(numbered) + f"\n\nQuestion: {q}")
    ]

    with st.spinner("Generating answer…"):
        resp = Settings.llm.chat(messages)
        answer_md = resp.message.content

    # Replace the THINKING bubble with the final answer
    st.session_state.history[idx]["content"] = answer_md

    # Clear pending and rerun to paint the final content cleanly
    st.session_state.pending = None
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# Handle new input (Pass A)
# ─────────────────────────────────────────────────────────────
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# Append user message to history
st.session_state.history.append({"role": "user", "content": q})

# Append a THINKING assistant bubble and remember its index
thinking_md = "🤔 **Thinking…**"
st.session_state.history.append({"role": "assistant", "content": thinking_md})
assistant_index = len(st.session_state.history) - 1

# Set pending task and rerun (Pass B will execute above)
st.session_state.pending = {"q": q, "assistant_index": assistant_index}
st.experimental_rerun()