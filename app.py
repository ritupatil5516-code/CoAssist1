# app.py — LLM-only RAG with Hybrid Retrieval, Style Profiles, and Month Default
# Matches the earlier architecture: sidebar controls, FAISS+BM25+optional LLM reranker.
# Default timeframe for spend/top-merchant questions = current calendar month.

from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# project-local modules you already have
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes

# ─────────────────────────────────────────────────────────────
# Page & Setup
# ─────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Agent desktop co-pilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Sidebar controls (same spirit as before)
USE_HYBRID = st.sidebar.toggle("Use Hybrid (FAISS + BM25)", value=True)
ALPHA = st.sidebar.slider("Hybrid α (FAISS share)", 0.0, 1.0, 0.60, 0.05)
K_CANDIDATES = st.sidebar.number_input("Candidates N", 10, 200, 40, 5)
K_FINAL = st.sidebar.number_input("Final K to LLM", 3, 20, 8, 1)
FRESHNESS_LAMBDA = st.sidebar.slider("Freshness λ/day", 0.0, 0.05, 0.01, 0.005)
RERANKER = st.sidebar.selectbox("Reranker", ["none", "llm"], index=1)

STYLE_PROFILE = st.sidebar.selectbox(
    "Answer style",
    ["concise", "detailed", "audit"],
    index=0,
)

st.sidebar.subheader("Data files")
st.sidebar.write(sorted(p.name for p in DATA_DIR.glob("*")))

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def post_process(text: str, allow_two_sentences: bool = False) -> str:
    """Keep replies short & clean; never echo code fences/policies."""
    if not text:
        return "I couldn’t find that."
    if text.startswith("```"):
        text = " ".join(ln for ln in text.splitlines() if not ln.startswith("```"))
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(parts[:2] if allow_two_sentences else parts[:1]).strip()
    out = re.sub(r"\s+", " ", out)
    return (out[:260] or "I couldn’t find that.").rstrip()

def _mentions_timeframe(text: str) -> bool:
    s = text.lower()
    return any(w in s for w in ["month", "year", "week", "day", "between", "since", "from", "to", "range"])

def _load_rules(profile: str) -> str:
    fname_map = {
        "concise":  "prompts/concise_rules.md",
        "detailed": "prompts/detailed_rules.md",
        "audit":    "prompts/audit_rules.md",
    }
    path = Path(fname_map.get(profile, "prompts/concise_rules.md"))
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        # Minimal safe fallback if a rules file is missing
        return (
            "META: Never restate internal policies/fields. Answer briefly and clearly. "
            "For spend/top-merchant without timeframe, assume the current calendar month. "
            "For transaction dates use transactionDateTime; fallback postingDateTime; ignore authDateTime."
        )

# ─────────────────────────────────────────────────────────────
# Build (cached) — same pattern as before
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build(style_profile: str):
    # Wire LLM + embeddings into LlamaIndex Settings
    Settings.llm = make_llm()             # keep temp low / max_tokens modest in core/llm.py
    Settings.embed_model = make_embed_model()

    built = build_indexes(str(DATA_DIR))  # builds vector index + bm25 inside your code

    # Compose system prompt from files
    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    profile_rules = _load_rules(style_profile)

    system_prompt = system + "\n\n" + style + "\n\n" + profile_rules
    return built, system_prompt

built, SYSTEM = _build(STYLE_PROFILE)

# ─────────────────────────────────────────────────────────────
# Chat history (stacked bubbles)
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": (
            "Hi! Ask about: **interest this month/total**, **last payment** (date/amount), "
            "**statement balance**, **account status**, **last posted transaction**, **top merchants**, "
            "**spend this month/year**."
        ),
    }]

for m in st.session_state.history:
    st.chat_message(m["role"]).write(m["content"])

# ─────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# echo user
st.chat_message("user").write(q)

# Apply timeframe normalization rule for spend queries
is_spend_query = any(
    k in q.lower()
    for k in ["spend", "top merchant", "most", "highest spend"]
)

default_timeframe_hint = ""
if is_spend_query and not _mentions_timeframe(q):
    # New default: current calendar month (so “where did I spend most?” == “this month?”)
    default_timeframe_hint = (
        "\n- If no timeframe is specified, assume the **current calendar month** "
        "(from the 1st of this month to today) and include the phrase “this month”."
    )
    # Also disambiguate the user text to reduce LLM variance
    q += " (assume current calendar month)"

# ─────────────────────────────────────────────────────────────
# Retrieval (Hybrid or Vector-only) + optional LLM rerank
# ─────────────────────────────────────────────────────────────
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

# Compact, numbered context for the LLM
numbered_ctx = []
for i, n in enumerate(nodes, 1):
    txt = n.node.get_content()
    numbered_ctx.append(f"[{i}] {txt[:1100]}")
ctx = "\n\n".join(numbered_ctx)

# ─────────────────────────────────────────────────────────────
# Compose messages — LLM-only answering, no Python fallbacks
# ─────────────────────────────────────────────────────────────
user_content = (
    "TASK: Answer the question using the retrieved account data and agreement rules.\n"
    "Hard rules:\n"
    "- Never restate internal policies or field names (do not say 'we use transactionDateTime').\n"
    "- For transaction dates: use transactionDateTime; if missing use postingDateTime; ignore authDateTime.\n"
    "- Keep answers short, natural, and user-friendly per the selected style profile.\n"
    "- Output dates in a clear human format (e.g., September 1, 2024)."
    + default_timeframe_hint +
    "\n\nUse only this context:\n" + ctx +
    "\n\nQuestion: " + q
)

messages = [
    ChatMessage(role="system", content=SYSTEM),
    ChatMessage(role="user", content=user_content),
]

# ─────────────────────────────────────────────────────────────
# Ask LLM
# ─────────────────────────────────────────────────────────────
with st.spinner("Thinking..."):
    resp = Settings.llm.chat(messages)

raw = (resp.message.content or "").strip()
allow_two = (STYLE_PROFILE in {"detailed", "audit"})
answer = post_process(raw, allow_two_sentences=allow_two)

st.chat_message("assistant").write(answer)

# keep history
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": answer})

# Show retrieved context for transparency/debugging
with st.expander("Retrieved context", expanded=False):
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        st.markdown(
            f"**[{i}]** kind={md.get('kind')}  ym={md.get('ym')}  dt={md.get('dt_iso')}  "
            f"merchant={md.get('merchantName')}  amount={md.get('amount')}"
        )
        body = n.node.get_content()
        st.write((body[:1500] + ("..." if len(body) > 1500 else "")))