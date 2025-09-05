import json
from typing import List, Tuple
from backend.llm.client import LLMClient
from backend.rag.types import Chunk

_SYSTEM = "Score relevance between 0.0 and 1.0. Return ONLY JSON: {\"scores\":[number,...]}."

def rerank_with_llm(query: str, candidates: List[Chunk], batch: int = 20) -> List[Tuple[Chunk, float]]:
    llm = LLMClient()
    results: List[Tuple[Chunk, float]] = []
    for i in range(0, len(candidates), batch):
        batch_items = candidates[i:i+batch]
        payload = {"query": query, "candidates": [c.text[:1200] for c in batch_items]}
        user = "Score each candidate for relevance to the query (0.0-1.0). Respond ONLY as JSON with 'scores'.\n" + json.dumps(payload)
        raw = llm.chat(_SYSTEM, user, temperature=0.0)
        try:
            data = json.loads(raw)
            scores = data.get("scores", [])
        except Exception:
            scores = [0.0] * len(batch_items)
        for c, s in zip(batch_items, scores):
            try: results.append((c, float(s)))
            except Exception: results.append((c, 0.0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
