# Banking Copilot — LlamaIndex build

- **Data**: `account-summary.json` (or `account_summary.json`), `statements.json`, `transactions.json`, `payments.json`, `agreement.pdf` in `./data/`
- **Retrieval**: LlamaIndex VectorStoreIndex (FAISS) + BM25Retriever (hybrid) + **freshness** postprocessor
- **Reranker**: LLM-based reranker (LlamaIndex `LLMRerank`)
- **LLM**: Llama‑3.3‑70B Instruct (OpenAI-compatible), **Embeddings**: Qwen3 8B (OpenAI-compatible)

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY (and OPENAI_BASE_URL if applicable)
streamlit run app.py
```

This version keeps your **context engineering** rules in `prompts/` and lets LlamaIndex handle indexing & retrieval.
