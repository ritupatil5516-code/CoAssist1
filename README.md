# Banking Copilot — No-HuggingFace Build (v3)

- No Hugging Face deps (`sentence-transformers`, `FlagEmbedding`, `transformers`) — removed.
- Retrieval: **FAISS** (API embeddings or TF‑IDF) + **BM25** (lexical).
- Optional reranker: **LLM-based** only.
- Context engineering via `/prompts` (system, retrieval, style, glossary).

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

# Put your data files in ./data:
# account_summary.json, statements.json, transactions.json, payments.json, agreement.pdf

# Embeddings:
# - API (default): goldmansachs.openai + EMBED_MODEL
# - Local TF‑IDF:   export USE_TFIDF=true
streamlit run app.py
```
