from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

# --- Backend modules (your existing ones) ---
from backend.utils.text import dir_mtime_fingerprint
from backend.rag.corpus import build_corpus
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store
from backend.context.builder import build_context  # builds top-k with hybrid
from backend.llm.client import LLMClient
from backend.rag.types import Chunk

# --- Prompt pack (context-engineering files) ---
from backend.prompting.loader import load_prompts
from backend.prompting.composer import make_system, make_user

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "data")).resolve()
PROMPTS_DIR = APP_DIR / "prompts"

REQUIRED = [
    "account_summary.json",
    "payments.json",
    "statements.json",
    "transactions.json",
    "agreement.pdf",
]

# ------------------------ UI CONFIG ------------------------
st.set_page_config(page_title="Agent desktop assist — Context Engineered", layout="wide")
st.title("Agent desktop assist")
st.caption("LLM-only answers grounded in retrieved context (agreement.pdf + JSON). Prompts & settings come from /prompts.")

# ------------------------ PROMPTS PACK ------------------------
# Fail-soft loading with a minimal fallback so app still runs if files are missing.
try:
    pack = load_prompts(PROMPTS_DIR)
except Exception as e:
    st.warning(f"Could not load prompt pack from {PROMPTS_DIR}: {e}")
    class _Fallback:
        system = (
            "You are a precise banking assistant. Answer only from context. "
            "Cite sources using bracket numbers like [1]. If info is missing, say so."
        )
        answer_style = ""
        refusal = "I don’t have enough data to answer precisely. Please specify the missing items."
        retrieval = "Prefer statements over transactions for interest; prefer latest data."
        glossary = "DPR=APR/365; Grace Period applies if prior balance paid in full."
        instructions = {
            "retrieval": {"use_hybrid": True, "alpha": 0.6, "k_candidates": 40, "k_final": 8},
            "safety": {"limit_history_chars": 1600},
        }
    pack = _Fallback()  # type: ignore

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.subheader("Data directory")
    st.code(str(DATA_DIR))
    if not DATA_DIR.exists():
        st.error("Data directory not found. Set DATA_DIR or create ./data next to app.py.")
        st.stop()

    files = sorted(p.name for p in DATA_DIR.glob("*"))
    st.write("Files found:", files)
    missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
    if missing:
        st.error(f"Missing required files: {missing}")
        st.stop()

    st.subheader("Retrieval settings (from prompts/instructions.yaml)")
    use_hybrid = st.toggle(
        "Use Hybrid (FAISS + BM25)",
        value=bool(pack.instructions["retrieval"]["use_hybrid"]),
    )
    alpha = st.slider(
        "Hybrid weight α (FAISS share)",
        min_value=0.0, max_value=1.0,
        value=float(pack.instructions["retrieval"]["alpha"]), step=0.05
    )
    k_candidates = st.number_input(
        "Candidates (top-N)",
        min_value=10, max_value=100,
        value=int(pack.instructions["retrieval"]["k_candidates"]), step=5
    )
    k_final = st.number_input(
        "Final top-K to Llama",
        min_value=3, max_value=20,
        value=int(pack.instructions["retrieval"]["k_final"]), step=1
    )

# ------------------------ BUILD INDEXES (FRESHNESS) ------------------------
fingerprint = dir_mtime_fingerprint(str(DATA_DIR))

@st.cache_resource(show_spinner=False)
def _build_resources(_fp: str):
    # Build the chunk corpus (JSON + agreement.pdf) and indexes
    chunks = build_corpus(str(DATA_DIR))              # -> List[Chunk]
    vec = FAISSStore(chunks)                          # semantic index
    bm25 = BM25Store(chunks)                          # lexical index
    return chunks, vec, bm25

chunks, vec_store, bm25_store = _build_resources(fingerprint)

# ------------------------ CONVERSATION STATE ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! Ask about transactions, statements, balances, payments, or interest rules. "
                    "I’ll answer strictly from your data + agreement.pdf and show my work if I calculate anything."}
    ]
if "last_results" not in st.session_state:
    st.session_state.last_results = []  # type: list[tuple[Chunk, float]]

def _recent_chat(limit_chars: int | None = None) -> str:
    txts = []
    for m in st.session_state.messages:
        role = "User" if m["role"] == "user" else "Assistant"
        txts.append(f"{role}: {m['content']}")
    s = "\n".join(txts)
    if limit_chars is None:
        return s
    return s[-limit_chars:]

# Render existing history as chat bubbles
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ------------------------ CHAT INPUT ------------------------
q = st.chat_input("Type your question…")
if not q:
    st.stop()

# Show user bubble
st.session_state.messages.append({"role": "user", "content": q})
with st.chat_message("user"):
    st.markdown(q)

# ------------------------ BUILD CONTEXT (RAG) ------------------------
results = build_context(
    query=q,
    vec=vec_store,
    bm25=bm25_store,
    use_hybrid=use_hybrid,
    alpha=float(alpha),
    kN=int(k_candidates),
    kK=int(k_final),
)
st.session_state.last_results = results

# Numbered context to encourage grounded citations
numbered_rows = []
for i, (chunk, score) in enumerate(results, 1):
    numbered_rows.append(f"[{i}] source={chunk.source} meta={chunk.meta} score={score:.3f}\n{chunk.text}")
context_block = "\n\n".join(numbered_rows)

# ------------------------ PROMPT COMPOSITION ------------------------
try:
    system_prompt = make_system(pack)
    hist_limit = int(pack.instructions["safety"]["limit_history_chars"])
except Exception:
    system_prompt = pack.system  # fallback
    hist_limit = 1600

user_prompt = make_user(
    pack=pack,
    conversation_tail=_recent_chat(hist_limit),
    numbered_context=results,
    question=q,
)

# ------------------------ LLM CALL ------------------------
llm = LLMClient()
answer = llm.chat(system_prompt, user_prompt)

# Assistant bubble
with st.chat_message("assistant"):
    st.markdown(answer)
st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------------ CONTEXT INSPECTOR ------------------------
with st.expander("Retrieved context used for the last answer", expanded=False):
    if st.session_state.last_results:
        for i, (chunk, score) in enumerate(st.session_state.last_results, 1):
            st.markdown(f"**[{i}]** `{chunk.source}` score={score:.3f}  \nmeta={chunk.meta}")
            st.write(chunk.text[:1000] + ("..." if len(chunk.text) > 1000 else ""))
    else:
        st.info("Ask a question to see the retrieved context here.")