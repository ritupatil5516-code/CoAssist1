# app.py — Chat-only UI; all knobs come from config/app.yaml

from __future__ import annotations
import os, re
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

from core.config import load_config
from core.llm import make_llm, make_embed_model
from core.indexes import build_indexes
from core.retrieve import hybrid_with_freshness, filter_spend_current_month
from core.reranker import rerank_nodes

# ─────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Agent desktop co-pilot — LlamaIndex", layout="wide")
st.title("Agent desktop co-pilot — LlamaIndex")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load config (cached by Streamlit)
@st.cache_data(show_spinner=False)
def _load_cfg():
    return load_config("config/app.yaml")

CFG = _load_cfg()

# Compose system prompt from files + style profile rules
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
        return (
            "META: Keep answers short and clear. For spend/top-merchant without timeframe, "
            "assume current month. Use transactionDateTime; fallback postingDateTime; ignore authDateTime."
        )

@st.cache_resource(show_spinner=False)
def _build(style_profile: str):
    Settings.llm = make_llm()
    Settings.embed_model = make_embed_model()

    built = build_indexes(str(DATA_DIR))

    system = Path("prompts/system.md").read_text(encoding="utf-8")
    style  = Path("prompts/assistant_style.md").read_text(encoding="utf-8")
    rules  = _load_rules(style_profile)
    system_prompt = f"{system}\n\n{style}\n\n{rules}"
    return built, system_prompt

built, SYSTEM = _build(CFG["llm"]["style_profile"])

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _mentions_timeframe(text: str) -> bool:
    s = text.lower()
    return any(w in s for w in ["month", "year", "week", "day", "between", "since", "from", "to"])

def post_process(text: str, allow_two_sentences: bool = False) -> str:
    if not text:
        return "I couldn’t find that."
    if text.startswith("```"):
        text = " ".join(ln for ln in text.splitlines() if not ln.startswith("```"))
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(parts[:2] if allow_two_sentences else parts[:1]).strip()
    out = re.sub(r"\s+", " ", out)
    return (out[:260] or "I couldn’t find that.").rstrip()

# ─────────────────────────────────────────────────────────────
# Chat state
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

st.chat_message("user").write(q)

# Spend / interest total defaults (current month), but never for “last time” interest
s = q.lower()
is_spend_query = any(k in s for k in ["spend", "top merchant", "most", "highest spend"])
is_interest_last = ("interest" in s) and any(w in s for w in ["last", "previous", "most recent", "last time"])
is_interest_total_query = (
    ("interest this" in s) or ("total interest" in s) or ("this month" in s) or ("this year" in s)
    or (("how much interest" in s) and not is_interest_last)
)

default_hint = ""
if (is_spend_query or is_interest_total_query) and not _mentions_timeframe(q):
    default_hint = (
        "\n- If no timeframe is specified for spend/interest totals, assume the **current calendar month** "
        "(from the 1st to today) and include the phrase “this month”."
    )
    q += " (assume current calendar month)"

# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────
if CFG["retrieval"]["use_hybrid"]:
    candidates = hybrid_with_freshness(
        built,
        q,
        alpha=CFG["retrieval"]["alpha"],
        lam=CFG["retrieval"]["freshness_lambda_per_day"],
        kN=CFG["retrieval"]["candidates_n"],
    )
else:
    candidates = built.vector_index.as_retriever(
        similarity_top_k=CFG["retrieval"]["candidates_n"]
    ).retrieve(q)

if is_spend_query:
    candidates = filter_spend_current_month(candidates)

nodes = candidates[: CFG["retrieval"]["final_k"]]
if CFG["reranker"]["name"] == "llm":
    nodes = rerank_nodes(candidates, q, k=CFG["retrieval"]["final_k"])

# Prepare numbered context
numbered = []
for i, n in enumerate(nodes, 1):
    txt = n.node.get_content()
    numbered.append(f"[{i}] {txt[:1100]}")
ctx = "\n\n".join(numbered)

# ─────────────────────────────────────────────────────────────
# Ask LLM
# ─────────────────────────────────────────────────────────────
user_content = (
    "TASK: Answer the question using the retrieved account data and agreement rules.\n"
    "Hard rules:\n"
    "- Never restate internal policies or field names.\n"
    "- For transaction dates: use transactionDateTime; if missing use postingDateTime; ignore authDateTime.\n"
    "- Consider only spend/outflow transactions when asked about “spend”/“where did I spend most”; "
    "exclude payment, refund, credit, interest.\n"
    "- Keep answers short, natural, and user-friendly.\n"
    "- Output dates like “September 1, 2024”."
    + default_hint +
    "\n\nUse only this context:\n" + ctx +
    "\n\nQuestion: " + q
)

messages = [
    ChatMessage(role="system", content=SYSTEM),
    ChatMessage(role="user", content=user_content),
]

with st.spinner("Thinking..."):
    resp = Settings.llm.chat(messages)

raw = (resp.message.content or "").strip()
allow_two = (CFG["llm"]["style_profile"] in {"detailed", "audit"})
answer = post_process(raw, allow_two_sentences=allow_two)

st.chat_message("assistant").write(answer)

# Save history
st.session_state.history.append({"role": "user", "content": q})
st.session_state.history.append({"role": "assistant", "content": answer})

# Optional transparency
if CFG["ui"]["show_retrieved_context"]:
    with st.expander("Retrieved context", expanded=False):
        for i, n in enumerate(nodes, 1):
            md = n.node.metadata or {}
            st.markdown(
                f"**[{i}]** kind={md.get('kind')}  ym={md.get('ym')}  dt={md.get('dt_iso')}  "
                f"merchant={md.get('merchantName')}  amount={md.get('amount')}"
            )
            body = n.node.get_content()
            st.write((body[:1500] + ("..." if len(body) > 1500 else "")))