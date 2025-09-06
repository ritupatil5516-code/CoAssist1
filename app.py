# app.py — stacked chat, concise answers, no JSON extractor

from __future__ import annotations
import os
from pathlib import Path
import re
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# project modules you already have
from core.llm import make_llm, make_embed_model   # make_llm should set temperature≈0.0–0.2, small max_tokens
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes

# ───────────────── Setup ─────────────────
load_dotenv()
st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Sidebar controls
USE_HYBRID = st.sidebar.toggle("Use Hybrid (FAISS + BM25)", value=True)
ALPHA = st.sidebar.slider("Hybrid α (FAISS share)", 0.0, 1.0, 0.6, 0.05)
K_CANDIDATES = st.sidebar.number_input("Candidates N", 10, 200, 40, 5)
K_FINAL = st.sidebar.number_input("Final K to LLM", 3, 20, 8, 1)
FRESHNESS_LAMBDA = st.sidebar.slider("Freshness λ/day", 0.0, 0.05, 0.01, 0.005)
RERANKER = st.sidebar.selectbox("Reranker", ["none", "llm"], index=1)

st.sidebar.subheader("Data files")
st.sidebar.write(sorted(p.name for p in DATA_DIR.glob("*")))

# ───────────── Build indexes (cached) ─────────────
@st.cache_resource(show_spinner=False)
def _build():
    Settings.llm = make_llm()                # set temp low & small max_tokens in core/llm.py
    Settings.embed_model = make_embed_model()

    built = build_indexes(str(DATA_DIR))

    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    concise_rule = Path("prompts/concise_rules.md").read_text(encoding="utf-8")

    system_prompt = system + "\n\n" + style + "\n\n" + concise_rule
    return built, system_prompt

built, SYSTEM = _build()

# ───────────── Small helpers ─────────────
def _ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def _pretty_date(dt_iso: str) -> str:
    try:
        s = dt_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        base = dt.strftime(f"%B {_ordinal(dt.day)} %Y")
        if dt.hour or dt.minute:
            base += dt.strftime(", %H:%M")
        return base
    except Exception:
        return dt_iso

def first_sentence(text: str) -> str:
    """Keep the answer tiny: first sentence or trimmed line; strip markdown fences."""
    if not text:
        return "I couldn’t find that."
    # remove code fences
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.startswith("```")]
        text = " ".join(lines).strip()
    # prefer first sentence
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    s = (m[0] if m else text).strip()
    # if the model split weirdly, fallback to first line
    if not s:
        s = text.splitlines()[0].strip()
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    # hard cap length
    if len(s) > 160:
        s = s[:157].rstrip() + "…"
    return s

# ───────────── Chat state ─────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": (
            "Hi! Ask about: **interest this month/total**, **last payment** (date/amount), "
            "**statement balance**, **account status**, **last posted transaction**, **top merchants**, "
            "**spend this month/year**."
        ),
    }]

# Render history stacked
for m in st.session_state.history:
    st.chat_message(m["role"]).write(m["content"])

# ───────────── Handle new input ─────────────
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# show user bubble
st.chat_message("user").write(q)

# retrieval
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

# compact, numbered context (like before)
numbered_ctx = []
for i, n in enumerate(nodes, 1):
    numbered_ctx.append(f"[{i}] {n.node.get_content()[:1100]}")
ctx = "\n\n".join(numbered_ctx)

messages = [
    ChatMessage(role="system", content=SYSTEM),
    ChatMessage(
        role="user",
        content=(
            "Use only the context below. Answer with ONE short sentence.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q}"
        ),
    ),
]

# ask LLM (non-streaming)
with st.spinner("Thinking..."):
    resp = Settings.llm.chat(messages)

answer_raw = resp.message.content or ""
answer = first_sentence(answer_raw)

# show assistant bubble
st.chat_message("assistant").write(answer)

# persist
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": answer})

# optional: show retrieved context
with st.expander("Retrieved context", expanded=False):
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        st.markdown(f"**[{i}]** kind={md.get('kind')}  ym={md.get('ym')}  dt={md.get('dt_iso')}")
        st.write(n.node.get_content()[:1000] + ("..." if len(n.node.get_content()) > 1000 else ""))