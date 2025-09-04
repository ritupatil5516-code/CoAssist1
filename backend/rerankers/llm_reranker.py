from __future__ import annotations
from typing import List, Tuple
import json
from backend.llm.client import LLMClient
from backend.rag.types import Chunk

SYSTEM = "You score relevance between 0.0 and 1.0. Return JSON: {\"scores\":[number,...]} only."

def rerank_with_llm(query: str, candidates: List[Chunk], batch: int = 20) -> List[Tuple[Chunk, float]]:
    llm = LLMClient()
    out: List[Tuple[Chunk, float]] = []
    for i in range(0, len(candidates), batch):
        batch_items = candidates[i:i+batch]
        payload = {"query": query, "candidates": [c.text[:1200] for c in batch_items]}
        user = "Score each candidate for relevance to the query (0.0-1.0). "                "Respond ONLY as JSON object with 'scores' list in same order.\n" + json.dumps(payload)
        resp = llm.chat(SYSTEM, user, temperature=0.0)
        try:
            data = json.loads(resp)
            scores = data.get("scores", [])
            for c, s in zip(batch_items, scores):
                try:
                    out.append((c, float(s)))
                except Exception:
                    out.append((c, 0.0))
        except Exception:
            out.extend((c, 0.0) for c in batch_items)
    out.sort(key=lambda x: x[1], reverse=True)
    return out
