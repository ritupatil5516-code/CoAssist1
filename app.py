from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from backend.utils.text import dir_mtime_fingerprint
from backend.prompting.loader import load_prompts
from backend.prompting.composer import make_system, make_user
from backend.context.builder import build_context

from backend.rag.corpus import build_corpus
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store

from backend.llm.client import LLMClient

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "data")).resolve()
PROMPTS_DIR = APP_DIR / "prompts"

load_dotenv(override=False)

REQUIRED = ["account_summary.json", "payments.json", "statements.json", "transactions.json", "agreement.pdf"]

st.set_page_config(page_title="Banking Copilot — No-HF v3", layout="wide")
st.title("Banking Copilot (No-HuggingFace) — v3")
st.caption("FAISS + BM25 + optional LLM reranker. Context-engineered prompts.")

pack = load_prompts(PROMPTS_DIR)

with st.sidebar:
    st.subheader("Data directory")
    st.code(str(DATA_DIR))
    DATA_DIR.mkdir(exist_ok=True)
    files = sorted(p.name for p in DATA_DIR.glob("*"))
    st.write("Files found:", files)
    missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
    if missing:
        st.warning(f"Missing files: {missing}. The app will run, but answers need these to be grounded.")

    st.subheader("Retrieval settings")
    use_hybrid = st.toggle("Use Hybrid (FAISS + BM25)", value=bool(pack.instructions["retrieval"]["use_hybrid"]))
    alpha = st.slider("Hybrid weight α (FAISS share)", 0.0, 1.0, float(pack.instructions["retrieval"]["alpha"]), 0.05)
    k_candidates = st.number_input("Candidates (top-N)", 10, 200, int(pack.instructions["retrieval"]["k_candidates"]), 5)
    k_final = st.number_input("Final top-K to Llama", 3, 20, int(pack.instructions["retrieval"]["k_final"]), 1)
    reranker = st.selectbox("Reranker", options=["none", "llm"], index=0)

fingerprint = dir_mtime_fingerprint(str(DATA_DIR))

@st.cache_resource(show_spinner=False)
def _build(_fp: str):
    chunks = build_corpus(str(DATA_DIR))
    return chunks, FAISSStore(chunks), BM25Store(chunks)

chunks, vec_store, bm25_store = _build(fingerprint)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hi! Ask about statements, transactions, payments, balances, or interest rules. I’ll answer from your data + agreement.pdf and cite sources."}]
if "last_results" not in st.session_state:
    st.session_state.last_results = []

def _recent(limit_chars: int):
    txts = []
    for m in st.session_state.messages:
        role = "User" if m["role"] == "user" else "Assistant"
        txts.append(f"{role}: {m['content']}")
    s = "\n".join(txts)
    return s[-limit_chars:]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Type your question…")
if not q:
    st.stop()

st.session_state.messages.append({"role": "user", "content": q})
with st.chat_message("user"):
    st.markdown(q)

results = build_context(
    query=q, vec=vec_store, bm25=bm25_store,
    use_hybrid=use_hybrid, alpha=float(alpha),
    kN=int(k_candidates), kK=int(k_final),
    reranker=reranker,
)
# ensure a rule/schema hint is always present in final context
from backend.rag.types import Chunk
rules_chunk = Chunk(text="RULE: Use AGGREGATE > STATEMENT.interestCharged > TRANSACTION[interestFlag=true] for monthly interest.",
                    source="rule", meta={"kind":"rule"})
schema_chunk = Chunk(text="SCHEMA: STATEMENT{ym,interestCharged}; TRANSACTION{ym,amount,interestFlag}; PAYMENT{ym,amount}.",
                     source="schema", meta={"kind":"schema"})

# prepend if missing
current = [c for c,_ in results]
prepend = []
if not any(getattr(c, "source", "") == "rule" for c in current):
    prepend.append((rules_chunk, 1.0))
if not any(getattr(c, "source", "") == "schema" for c in current):
    prepend.append((schema_chunk, 1.0))
results = prepend + results
results = results[:int(k_final)]  # keep final K cap

st.session_state.last_results = results

system_prompt = make_system(pack)
user_prompt = make_user(
    pack=pack,
    conversation_tail=_recent(int(pack.instructions["safety"]["limit_history_chars"])),
    numbered_context=results,
    question=q,
)

llm = LLMClient()
answer = llm.chat(system_prompt, user_prompt)

with st.chat_message("assistant"):
    st.markdown(answer)
st.session_state.messages.append({"role":"assistant","content":answer})

with st.expander("Retrieved context used for the last answer", expanded=False):
    if st.session_state.last_results:
        for i,(chunk,score) in enumerate(st.session_state.last_results,1):
            st.markdown(f"**[{i}]** `{chunk.source}` score={score:.3f}  \nmeta={chunk.meta}")
            st.write(chunk.text[:1000] + ("..." if len(chunk.text) > 1000 else ""))
    else:
        st.info("Ask a question to see the retrieved context here.")
