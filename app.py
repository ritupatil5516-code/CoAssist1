# app.py — LLM-only RAG with Hybrid Retrieval, Freshness, Optional Reranker
# Layout: sidebar controls (like your working version) + stacked chat bubbles.
# Defaults: spend/top-merchant questions use the CURRENT CALENDAR MONTH.
# Date policy: use transactionDateTime; fallback postingDateTime; ignore authDateTime.

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# ==== your project modules (unchanged names) ====
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes          # returns Built(nodes, vector_index, bm25)
from core.retrieve import hybrid_with_freshness, filter_spend_current_month
from core.reranker import rerank_nodes          # optional LLM reranker

# >>> NEW: short answers (your current file API)
from core.short_answers import (
    detect_intent,
    build_extraction_prompt,
    format_answer,
)

# ─────────────────────────────────────────────────────────────
# Page & Setup
# ─────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Agent desktop co-pilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Sidebar controls (same knobs you used before)
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
        # Minimal safe fallback if file missing
        return (
            "META: Never restate internal policies/fields. Answer briefly and clearly. "
            "For spend/top-merchant without timeframe, assume the current calendar month. "
            "For transaction dates use transactionDateTime; fallback postingDateTime; ignore authDateTime."
        )

def _strip_code_fences(s: str) -> str:
    if s.startswith("```"):
        s = "\n".join(ln for ln in s.splitlines() if not ln.strip().startswith("```"))
    return s.strip()

def _coerce_json(s: str):
    """
    Be lenient with model output: strip fences, find first {...} block.
    """
    s = _strip_code_fences(s)
    # grab first top-level JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    return json.loads(s)

def post_process(text: str, allow_two_sentences: bool = False) -> str:
    """(Kept for compatibility) Keep answers short if we ever fall back to raw text."""
    if not text:
        return "I couldn’t find that."
    if text.startswith("```"):
        text = " ".join(ln for ln in text.splitlines() if not ln.startswith("```"))
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(parts[:2] if allow_two_sentences else parts[:1]).strip()
    out = re.sub(r"\s+", " ", out)
    return (out[:260] or "I couldn’t find that.").rstrip()

# ─────────────────────────────────────────────────────────────
# Build (cached) — same pattern as your working file
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build(style_profile: str):
    # Wire LLM + embeddings into LlamaIndex Settings
    Settings.llm = make_llm()             # configure temp/max_tokens in core/llm.py
    Settings.embed_model = make_embed_model()

    built = build_indexes(str(DATA_DIR))  # builds vector index + bm25

    # Compose system prompt from files
    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    profile_rules = _load_rules(style_profile)

    system_prompt = system + "\n\n" + style + "\n\n" + profile_rules
    return built, system_prompt

built, SYSTEM = _build(STYLE_PROFILE)

# ─────────────────────────────────────────────────────────────
# Chat state (stacked bubbles)
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

# show user msg
st.chat_message("user").write(q)

# Decide spend-style query and normalize timeframe to CURRENT CALENDAR MONTH if unspecified
is_spend_query = any(
    k in q.lower() for k in ["spend", "top merchant", "most", "highest spend"]
)

default_timeframe_hint = ""
if is_spend_query and not _mentions_timeframe(q):
    default_timeframe_hint = (
        "\n- If no timeframe is specified, assume the **current calendar month** "
        "(from the 1st of this month to today) and include the phrase “this month”."
    )
    # Normalize the user text to reduce variance
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

# For spend queries, drop non-spend/outflow and keep current month only (no payments/credits/interest)
if is_spend_query:
    candidates = filter_spend_current_month(candidates)

# Optional LLM reranker
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
# NEW: Intent + JSON extraction + final formatting
# ─────────────────────────────────────────────────────────────
intent = detect_intent(q)
extraction_prompt = build_extraction_prompt(intent, q, ctx)

# Compose messages for JSON-only extraction (keeps your SYSTEM prompt active)
messages = [
    ChatMessage(role="system", content=SYSTEM),
    ChatMessage(role="user", content=extraction_prompt),
]

with st.spinner("Thinking..."):
    resp = Settings.llm.chat(messages)

raw = (resp.message.content or "").strip()

# Try to parse JSON
final_answer = None
try:
    data = _coerce_json(raw)
    final_answer = format_answer(intent, data)
except Exception:
    # If the model didn't return JSON, fall back to your previous single-sentence post_processor
    allow_two = (STYLE_PROFILE in {"detailed", "audit"})
    final_answer = post_process(raw, allow_two_sentences=allow_two)

st.chat_message("assistant").write(final_answer)

# Keep history
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": final_answer})

# Optional: show retrieved context for transparency
with st.expander("Retrieved context", expanded=False):
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        st.markdown(
            f"**[{i}]** kind={md.get('kind')}  ym={md.get('ym')}  dt={md.get('dt_iso')}  "
            f"merchant={md.get('merchantName')}  amount={md.get('amount')}"
        )
        body = n.node.get_content()
        st.write((body[:1500] + ('...' if len(body) > 1500 else '')))