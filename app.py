# app.py
from __future__ import annotations
import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# your project modules
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Page + env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# defaults
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
    FRESHNESS_LAMBDA = st.slider("Freshness λ/day (freshness bias)", 0.0, 0.05, FRESHNESS_LAMBDA, 0.005)
    RERANKER = st.selectbox("Reranker", ["none", "llm"], index=1 if RERANKER == "llm" else 0)


# ──────────────────────────────────────────────────────────────────────────────
# Build (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build():
    # wire models
    Settings.llm = make_llm()
    Settings.embed_model = make_embed_model()

    built = build_indexes(str(DATA_DIR))

    # system prompt
    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    system_prompt = system + "\n\n" + style

    # quick BM25 smoke test
    with st.sidebar:
        st.markdown("### Startup checks")
        try:
            test_q = "statement balance"
            nodes = built.bm25.retrieve(test_q)
            st.success(f"BM25 OK — {len(nodes)} nodes for '{test_q}'.")
        except Exception as e:
            st.error(f"BM25 check failed: {e}")

    return built, system_prompt

built, SYSTEM = _build()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: stream answer in a chat bubble with a visible “Thinking…”
# ──────────────────────────────────────────────────────────────────────────────
def stream_answer_in_chat(chat_container, q: str, nodes, system_prompt: str) -> str:
    """
    Render '🤔 Thinking…' immediately inside the assistant bubble,
    then stream tokens into the same placeholder. Single output (no duplicates).
    """
    # Build compact, numbered context (fits in context window)
    numbered = []
    for i, n in enumerate(nodes, 1):
        txt = n.node.get_content()[:1100]
        numbered.append(f"[{i}] {txt}")

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content="Context:\n" + "\n\n".join(numbered) + f"\n\nQuestion: {q}",
        ),
    ]

    # placeholder inside the assistant bubble
    ph = chat_container.empty()
    ph.markdown("🤔 **Thinking…**")   # guaranteed to render
    buf = ""
    got_first = False

    try:
        # small delay ensures the “Thinking…” paint before streaming begins
        time.sleep(0.15)

        for chunk in Settings.llm.stream_chat(messages):
            delta = getattr(chunk, "delta", None) or getattr(chunk, "message", None)
            text = delta if isinstance(delta, str) else getattr(delta, "content", "") or ""
            if not text:
                continue

            if not got_first:
                got_first = True
                # first token arrived; we start replacing the indicator
                # (no need to .empty(); updating the same placeholder overwrites it)

            buf += text
            ph.markdown(buf)

        # If nothing streamed, keep the indicator briefly then clear it
        if not got_first:
            time.sleep(0.6)
            ph.empty()

        return buf

    except Exception:
        # non-streaming fallback (still in the same placeholder; no duplication)
        with chat_container:
            with st.spinner("🤔 Thinking…"):
                resp = Settings.llm.chat(messages)
        buf = resp.message.content
        ph.markdown(buf)
        return buf


# ──────────────────────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": (
            "Hi! Ask about: **interest this month/total**, **interest-causing transactions**, "
            "**last payment** (date/amount), **statement balance**, **account status**, "
            "**last posted transaction**, **top merchants**, **spend this month/year**."
        ),
    }]

# Render prior messages once (prevents duplicates)
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ──────────────────────────────────────────────────────────────────────────────
# Input → Retrieval → Rerank → Streamed answer (with “Thinking…”)
# ──────────────────────────────────────────────────────────────────────────────
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# Render current user bubble now (do NOT append to history yet → avoids double render)
with st.chat_message("user"):
    st.markdown(q)

# Assistant bubble container (stable placeholder for the whole response)
assistant_box = st.chat_message("assistant")

# Retrieval
if USE_HYBRID:
    candidates = hybrid_with_freshness(
        built, q, alpha=ALPHA, lam=FRESHNESS_LAMBDA, kN=K_CANDIDATES
    )
else:
    candidates = built.vector_index.as_retriever(
        similarity_top_k=K_CANDIDATES
    ).retrieve(q)

# Optional reranker; otherwise just cap candidates
nodes = candidates[:K_FINAL]
if RERANKER == "llm":
    nodes = rerank_nodes(candidates, q, k=K_FINAL)

# Stream answer in the same assistant bubble (shows Thinking… first)
answer_md = stream_answer_in_chat(assistant_box, q, nodes, SYSTEM)

# Append both messages to history ONCE (they’ll render next run)
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": answer_md})

# Context viewer
with st.expander("Retrieved context", expanded=False):
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        st.markdown(
            f"**[{i}]** kind={md.get('kind')}  ym={md.get('ym')}  dt={md.get('dt_iso')}"
        )
        content = n.node.get_content()
        st.write(content[:1000] + ("..." if len(content) > 1000 else ""))