# Banking Copilot — Production-Ready RAG (FAISS + BM25) with Context Engineering

This app is a Streamlit chat that answers questions about **statements, transactions, payments, accounts** and rules from **agreement.pdf** using **LLM-only** reasoning, grounded in **retrieved context** (no tools required).

## Features
- Chat bubbles + conversation memory
- Context-engineering via `/prompts` (system, style, retrieval, glossary, refusal, YAML switches)
- FAISS (semantic) + BM25 (lexical) hybrid retrieval, optional **re-ranking** (BGE or LLM JSON)
- **Pydantic**-validated JSON models (v1/v2 compatible)
- Agreement PDF ingestion
- Sentence-Transformers fallback for embeddings `USE_LOCAL_EMBEDS=true`
- Aggregated **AGGREGATE** chunks (e.g., interest totals by month) to make numeric answers trivial for the LLM
- Env-driven config + `.env.example`
- Optional Dockerfile

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put your files under ./data next to app.py:
#   account_summary.json, statements.json, transactions.json, payments.json, agreement.pdf

# optional: local embeddings
export USE_LOCAL_EMBEDS=true

streamlit run app.py
```

## Env Vars
See `.env.example` for all variables. The app uses `goldmansachs.openai` if available, else falls back to Sentence-Transformers.

## Docker
```bash
docker build -t banking-copilot .
docker run -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data banking-copilot
```
