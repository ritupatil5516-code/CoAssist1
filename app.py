# app.py — simplified stacked style, concise answers

from __future__ import annotations
import os, re, json
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

# Project modules
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness
from core.reranker import rerank_nodes

# ───────────── Setup ─────────────
load_dotenv()
st.set_page_config(page_title="Banking Copilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Sidebar options
USE_HYBRID = st.sidebar.toggle("Use Hybrid (FAISS + BM25)", value=True)
ALPHA = st.sidebar.slider("Hybrid α (FAISS share)", 0.0, 1.0, 0.6, 0.05)
K_CANDIDATES = st.sidebar.number_input("Candidates N", 10, 200, 40, 5)
K_FINAL = st.sidebar.number_input("Final K to LLM", 3, 20, 8, 1)
FRESHNESS_LAMBDA = st.sidebar.slider("Freshness λ/day", 0.0, 0.05, 0.01, 0.005)
RERANKER = st.sidebar.selectbox("Reranker", ["none", "llm"], index=1)

# Show loaded data files
st.sidebar.subheader("Data files")
st.sidebar.write(sorted(p.name for p in DATA_DIR.glob("*")))

# ───────────── Build indexes ─────────────
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

# ───────────── Concise Answer Helpers ─────────────
def _ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def _pretty_date(dt_iso: str) -> str:
    if not dt_iso: return ""
    try:
        s = dt_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        base = dt.strftime(f"%B {_ordinal(dt.day)} %Y")
        if dt.hour or dt.minute:
            base += dt.strftime(", %H:%M")
        return base
    except: return dt_iso

def _fmt_money(v):
    try: return f"${float(v):,.2f}"
    except: return str(v)

def detect_intent(q: str) -> str:
    s = q.lower()
    if "current balance" in s: return "current_balance"
    if "statement balance" in s: return "statement_balance"
    if "last payment" in s and "when" in s: return "last_payment_date"
    if "last payment" in s and ("amount" in s or "what was" in s): return "last_payment_amount"
    if "account status" in s: return "account_status"
    if "interest" in s: return "interest_total_month"
    return "generic"

def build_extraction_prompt(intent, question, ctx):
    fields = {
        "current_balance": ["current_balance"],
        "statement_balance": ["statement_balance"],
        "last_payment_amount": ["payment_amount", "payment_date"],
        "last_payment_date": ["payment_date"],
        "account_status": ["account_status"],
        "interest_total_month": ["interest_total", "ym"],
        "generic": ["answer_text"],
    }
    wanted = ", ".join(f'"{k}"' for k in fields[intent])
    return f"""
You are a banking assistant. Read context and return JSON with keys: {wanted}.
Context:
{ctx}
User question: {question}
Return JSON only:
""".strip()

def format_answer(intent, data: dict) -> str:
    if intent == "current_balance" and "current_balance" in data:
        return f"Your current balance is {_fmt_money(data['current_balance'])}."
    if intent == "statement_balance" and "statement_balance" in data:
        return f"Your statement balance is {_fmt_money(data['statement_balance'])}."
    if intent == "last_payment_amount" and "payment_amount" in data:
        amt = _fmt_money(data["payment_amount"])
        dt = _pretty_date(data.get("payment_date",""))
        return f"Your last payment was {amt}" + (f" on {dt}." if dt else ".")
    if intent == "last_payment_date" and "payment_date" in data:
        return f"Your last payment date was {_pretty_date(data['payment_date'])}."
    if intent == "account_status" and "account_status" in data:
        return f"Your account status is {data['account_status']}."
    if intent == "interest_total_month" and "interest_total" in data:
        return f"Your total interest is {_fmt_money(data['interest_total'])}."
    return "I couldn't find that."

# ───────────── Chat state ─────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": (
            "Hi! Ask about: **interest this month/total**, **last payment** (date/amount), "
            "**statement balance**, **account status**, etc."
        ),
    }]

# Render history stacked
for m in st.session_state.history:
    if m["role"] == "user":
        st.chat_message("user").write(m["content"])
    else:
        st.chat_message("assistant").write(m["content"])

# ───────────── Handle new input ─────────────
q = st.chat_input("Type your question…")
if q:
    st.chat_message("user").write(q)
    intent = detect_intent(q)

    # Retrieval
    if USE_HYBRID:
        candidates = hybrid_with_freshness(built, q, alpha=ALPHA, lam=FRESHNESS_LAMBDA, kN=K_CANDIDATES)
    else:
        candidates = built.vector_index.as_retriever(similarity_top_k=K_CANDIDATES).retrieve(q)

    nodes = candidates[:K_FINAL]
    if RERANKER == "llm":
        nodes = rerank_nodes(candidates, q, k=K_FINAL)

    ctx_str = "\n\n".join(f"[{i}] {n.node.get_content()[:800]}" for i,n in enumerate(nodes,1))
    extraction_prompt = build_extraction_prompt(intent, q, ctx_str)

    messages = [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(role="user", content=extraction_prompt),
    ]

    with st.spinner("Thinking..."):
        resp = Settings.llm.chat(messages)

    raw = resp.message.content.strip()
    data = {}
    try:
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
        data = json.loads(raw)
    except:
        pass

    answer = format_answer(intent, data)
    st.chat_message("assistant").write(answer)

    st.session_state.history.append({"role": "user", "content": q})
    st.session_state.history.append({"role": "assistant", "content": answer})