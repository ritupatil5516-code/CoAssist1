# Banking Copilot — LlamaIndex (v2)

- **Data**: `account-summary.json` (or `account_summary.json`), `statements.json`, `transactions.json`, `payments.json`, `agreement.pdf` in `./data/`
- **Retrieval**: VectorStoreIndex (FAISS) + **LangChain BM25** (with local fallback) + Freshness
- **Reranker**: LLMRerank (optional)
- **LLM**: Llama‑3.3‑70B (chat), **Embeddings**: Qwen3‑8B (OpenAI-compatible)

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY and (optionally) OPENAI_BASE_URL
streamlit run app.py
```

On startup, the sidebar shows a **BM25 health check** and logs which backend is active (LangChain or local rank-bm25).
