# app.py
from __future__ import annotations
import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# project imports
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes


# ────────────────────────────────────────────────────────────────
# Page setup
# ────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Defaults
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() in ("1", "true", "yes")
ALPHA = float(os.getenv("ALPHA", "0.6"))
K_CANDIDATES = int(os.getenv("K_CANDIDATES", "40"))
K_FINAL = int(os.getenv("K_FINAL", "8"))
FRESHNESS_LAMBDA = float(os.getenv("FRESHNESS_LAMBDA", "0.01"))
RERANKER = os.getenv("RERANKER", "llm")  # "llm" or "none"

with st.sidebar:
    st.subheader("Data files (./data)")
    st.write(sorted(p.name for p in DATA_DIR.glob("*")))
    USE_HYBRID = st.toggle("Use Hybrid (FAISS + BM25)", value=USE_HYBRID)
    ALPHA = st.slider("Hybrid α (FAISS share)", 0.0, 1.0, ALPHA, 0.05)
    K_CANDIDATES = st.number_input("Candidates N", 10, 200, K_CANDIDATES, 5)
    K_FINAL = st.number_input("Final K to LLM", 3, 20, K_FINAL, 1)
    FRESHNESS_LAMBDA = st.slider("Freshness λ/day", 0.0, 0.05, FRESHNESS_LAMBDA, 0.005)
    RERANKER = st.selectbox("Reranker", ["none", "llm"], index=1 if RERANKER == "llm" else 0)


# ────────────────────────────────────────────────────────────────
# Build (cached)
# ────────────────────────────────────────────────────────────────
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
            st.success(f"BM25 OK — {len(nodes)} nodes for 'statement balance'")
        except Exception as e:
            st.error(f"BM25 check failed: {e}")

    return built, system_prompt


built, SYSTEM = _build()


# ────────────────────────────────────────────────────────────────
# Typing effect + Thinking indicator
# ────────────────────────────────────────────────────────────────
def stream_answer_with_typing(chat_container, q: str, nodes, system_prompt: str) -> str:
    """
    Inside chat_container:
      - show 🤔 Thinking… instantly
      - animate dots while waiting
      - type characters as they arrive
    """
    from llama_index.core.llms import ChatMessage
    from llama_index.core import Settings

    # Build compact context
    numbered = []
    for i, n in enumerate(nodes, 1):
        txt = n.node.get_content()[:900]
        numbered.append(f"[{i}] {txt}")

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content="Context:\n" + "\n\n".join(numbered) + f"\n\nQuestion: {q}"),
    ]

    # placeholder inside assistant bubble
    ph = chat_container.empty()
    ph.markdown("🤔 **Thinking…**")

    dots_frames = ["", ".", "..", "...", "...."]
    dots_idx = 0
    warmup_deadline = time.time() + 3.0

    buffer = []
    screen = ""

    def type_to_screen(new_text: str, delay: float = 0.01):
        nonlocal screen
        for ch in new_text:
            screen += ch
            ph.markdown(screen)
            time.sleep(delay)

    try:
        stream = Settings.llm.stream_chat(messages)
        got_any = False
        last_tick = time.time()

        for chunk in stream:
            delta = getattr(chunk, "delta", None) or getattr(chunk, "message", None)
            text = delta if isinstance(delta, str) else getattr(delta, "content", "") or ""

            if text:
                if not got_any:
                    got_any = True
                    ph.markdown("")  # clear “Thinking…”
                buffer.append(text)
                type_to_screen(text, delay=0.01)
            else:
                if not got_any and time.time() < warmup_deadline:
                    if time.time() - last_tick >= 0.25:
                        ph.markdown(f"🤔 **Thinking{dots_frames[dots_idx % len(dots_frames)]}**")
                        dots_idx += 1
                        last_tick = time.time()

        if not got_any:
            time.sleep(0.5)
            ph.markdown("⚠️ No response from model.")
            return ""

        return "".join(buffer)

    except Exception as e:
        ph.markdown(f"⚠️ Error: {e}")
        return ""


# ────────────────────────────────────────────────────────────────
# Chat history
# ────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": (
            "Hi! Ask about: **interest this month/total**, **interest-causing transactions**, "
            "**last payment** (date/amount), **statement balance**, **account status**, "
            "**last posted transaction**, **top merchants**, **spend this month/year**."
        ),
    }]

# Render history
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ────────────────────────────────────────────────────────────────
# Handle new input
# ────────────────────────────────────────────────────────────────
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# user bubble
with st.chat_message("user"):
    st.markdown(q)

# assistant bubble
assistant_box = st.chat_message("assistant")

# retrieval
if USE_HYBRID:
    candidates = hybrid_with_freshness(built, q, alpha=ALPHA,
                                       lam=FRESHNESS_LAMBDA, kN=K_CANDIDATES)
else:
    candidates = built.vector_index.as_retriever(
        similarity_top_k=K_CANDIDATES).retrieve(q)

nodes = candidates[:K_FINAL]
if RERANKER == "llm":
    nodes = rerank_nodes(candidates, q, k=K_FINAL)

# stream with typing effect
answer_md = stream_answer_with_typing(assistant_box, q, nodes, SYSTEM)

# append to history (only once per run)
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": answer_md})

# context viewer
with st.expander("Retrieved context", expanded=False):
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        st.markdown(f"**[{i}]** kind={md.get('kind')} ym={md.get('ym')} dt={md.get('dt_iso')}")
        st.write(n.node.get_content()[:1000])