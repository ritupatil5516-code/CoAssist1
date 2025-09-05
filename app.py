# app.py — simple non-streaming chat with left/right bubbles
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# Project modules
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes

# ─────────────────────────────────────────────────────────────
# Page / env
# ─────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

# Left/Right chat bubbles (CSS)
st.markdown("""
    <style>
    /* Container to constrain the chat width a bit */
    .chat-wrapper { max-width: 980px; margin-left: auto; margin-right: auto; }

    /* Base bubble */
    .chat-bubble {
        padding: 0.8em 1em;
        margin: 0.35em 0;
        border-radius: 16px;
        max-width: 80%;
        word-wrap: break-word;
        line-height: 1.38;
        font-size: 0.95rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }

    /* Role rows */
    .row { display: flex; width: 100%; }
    .row.user { justify-content: flex-end; }      /* right align */
    .row.assistant { justify-content: flex-start; } /* left align */

    /* User bubble (right / greenish) */
    .user-bubble {
        background: #DCF8C6; /* WhatsApp-like green */
        color: #111;
        border-top-right-radius: 6px;
    }

    /* Assistant bubble (left / gray) */
    .assistant-bubble {
        background: #ECECEC;
        color: #111;
        border-top-left-radius: 6px;
    }

    /* Small, subtle meta row spacing */
    .msg-spacer { height: 4px; }
    </style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Sidebar controls (defaults via env)
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

    # quick BM25 smoke test
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
# Chat rendering helpers (left/right bubbles)
# ─────────────────────────────────────────────────────────────
def render_message(role: str, content: str):
    """Render a single message using left/right bubble layout."""
    # sanitize content minimal (Streamlit's markdown already escapes code by backticks)
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    row_class = "row user" if role == "user" else "row assistant"
    st.markdown(
        f"""
        <div class="chat-wrapper">
          <div class="{row_class}">
            <div class="chat-bubble {bubble_class}">{content}</div>
          </div>
          <div class="msg-spacer"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Chat history
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

# Render prior messages using custom bubbles
for m in st.session_state.history:
    render_message(m["role"], m["content"])

# ─────────────────────────────────────────────────────────────
# Handle new input → retrieve → (rerank) → answer (single shot)
# ─────────────────────────────────────────────────────────────
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# Render user bubble on the right and store
render_message("user", q)

# Retrieval
if USE_HYBRID:
    candidates = hybrid_with_freshness(
        built, q, alpha=ALPHA, lam=FRESHNESS_LAMBDA, kN=K_CANDIDATES
    )
else:
    candidates = built.vector_index.as_retriever(
        similarity_top_k=K_CANDIDATES
    ).retrieve(q)

nodes = candidates[:K_FINAL]
if RERANKER == "llm":
    nodes = rerank_nodes(candidates, q, k=K_FINAL)

# Build compact context
numbered_ctx = []
for i, n in enumerate(nodes, 1):
    txt = n.node.get_content()[:1100]
    numbered_ctx.append(f"[{i}] {txt}")

messages = [
    ChatMessage(role="system", content=SYSTEM),
    ChatMessage(role="user", content="Context:\n" + "\n\n".join(numbered_ctx) + f"\n\nQuestion: {q}"),
]

# Ask LLM (non-streaming; most reliable)
with st.spinner("Working…"):
    resp = Settings.llm.chat(messages)
answer_md = resp.message.content

# Render assistant bubble on the left and store
render_message("assistant", answer_md)

# Persist chat (so it shows on next rerun without re-rendering current ones)
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": answer_md})

# Context viewer (optional)
with st.expander("Retrieved context", expanded=False):
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        st.markdown(
            f"**[{i}]** kind={md.get('kind')}  ym={md.get('ym')}  dt={md.get('dt_iso')}"
        )
        content = n.node.get_content()
        st.write(content[:1000] + ("..." if len(content) > 1000 else ""))