from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

from backend.utils.text import dir_mtime_fingerprint
from backend.rag.corpus import build_corpus
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store
from backend.rag.hybrid import hybrid_merge
from backend.llm.client import LLMClient
from backend.rag.types import Chunk

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "data")).resolve()

REQUIRED = [
    "account_summary.json",
    "payments.json",
    "statements.json",
    "transactions.json",
    "agreement.pdf",
]

st.set_page_config(page_title="Banking Co-Pilot — Chat", layout="wide")
st.title("Banking Co-Pilot")
st.caption("Chat with your account data + agreement.pdf. Answers are grounded in retrieved context.")

# ---------- Sidebar: data checks + knobs ----------
with st.sidebar:
    st.subheader("Data path & files")
    st.code(str(DATA_DIR))
    exists = DATA_DIR.exists()
    st.write("📁 Data directory exists:", exists)
    if exists:
        files = sorted(p.name for p in DATA_DIR.glob("*"))
        st.write("Files found:", files)
    else:
        st.error("Data directory not found. Set DATA_DIR or create ./data.")
    missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
    if missing:
        st.error(f"Missing required files: {missing}")
        st.stop()

    st.divider()
    st.subheader("Retrieval settings")
    use_hybrid = st.toggle("Use Hybrid (FAISS + BM25)", value=True)
    alpha = st.slider("Hybrid weight α (FAISS share)", 0.0, 1.0, 0.6, 0.05)
    k_candidates = st.number_input("Candidates (top-N)", min_value=10, max_value=100, value=40, step=5)
    k_final = st.number_input("Final top-K to Llama", min_value=3, max_value=20, value=8, step=1)

    st.divider()
    st.subheader("Embedding mode")
    st.write("Default uses your internal API (`goldmansachs.openai`).")
    st.caption("Set env `USE_LOCAL_EMBEDS=true` to force local Sentence-Transformers fallback.")

# ---------- Build / cache resources with freshness ----------
fingerprint = dir_mtime_fingerprint(str(DATA_DIR))

@st.cache_resource(show_spinner=False)
def _build_resources(_fingerprint: str):
    chunks = build_corpus(str(DATA_DIR))       # List[Chunk] (robust)
    vec = FAISSStore(chunks)                   # FAISS semantic index
    bm25 = BM25Store(chunks)                   # BM25 lexical index
    return chunks, vec, bm25

chunks, vec_store, bm25_store = _build_resources(fingerprint)

# ---------- Conversation state ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about your transactions, statements, accounts, or how interest/minimum due is calculated. I’ll cite the context I used."}
    ]

if "last_results" not in st.session_state:
    st.session_state.last_results = []  # store (Chunk, score) for the last turn

# Render chat history (bubbles)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Helper: build short transcript for the LLM ----------
def build_transcript(max_chars: int = 1500) -> str:
    txts = []
    for m in st.session_state.messages:
        role = "User" if m["role"] == "user" else "Assistant"
        txts.append(f"{role}: {m['content']}")
    transcript = "\n".join(txts)[-max_chars:]  # tail-trim to keep it short
    return transcript

# ---------- Retrieval function ----------
def retrieve(query: str) -> list[tuple[Chunk, float]]:
    vec = vec_store.search(query, k=int(k_candidates))
    if use_hybrid:
        lex = bm25_store.search(query, k=int(k_candidates))
        merged = hybrid_merge(vec, lex, alpha=float(alpha), k=int(k_candidates))
    else:
        merged = vec
    return merged[:int(k_final)]

# ---------- Chat input ----------
prompt = st.chat_input("Type your question…")
if prompt:
    # Show user bubble immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve supporting context for THIS turn
    results = retrieve(prompt)
    st.session_state.last_results = results

    # Build “context with markers” so the LLM can cite [#]
    numbered_chunks = []
    for i, (chunk, score) in enumerate(results, 1):
        header = f"[{i}] source={chunk.source} meta={chunk.meta} score={score:.3f}"
        text = chunk.text
        numbered_chunks.append(f"{header}\n{text}")
    context_block = "\n\n".join(numbered_chunks)

    # Build short conversation transcript
    transcript = build_transcript()

    # Compose LLM messages
    system = (
        "You are a precise banking assistant. "
        "Always ground your answer ONLY in the provided context. "
        "If the context lacks the information, say you don’t have data. "
        "Show calculations step-by-step when applicable. "
        "Cite the chunks by their bracket numbers, e.g., [2], [4]."
    )

    user_msg = (
        f"Conversation (recent):\n{transcript}\n\n"
        f"Retrieved Context (numbered chunks):\n{context_block}\n\n"
        f"User question: {prompt}\n\n"
        "Answer concisely. Include citations like [#] that map to the chunk markers."
    )

    # Call Llama
    llm = LLMClient()
    answer = llm.chat(system, user_msg)

    # Assistant bubble
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------- Context inspector for the latest turn ----------
with st.expander("Retrieved context used for the last answer", expanded=False):
    if st.session_state.last_results:
        for i, (chunk, score) in enumerate(st.session_state.last_results, 1):
            st.markdown(f"**[{i}]** `{chunk.source}` score={score:.3f}  \nmeta={chunk.meta}")
            st.write(chunk.text[:1000] + ("..." if len(chunk.text) > 1000 else ""))
    else:
        st.info("Ask a question to see the retrieved context here.")