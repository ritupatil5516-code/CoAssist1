from __future__ import annotations
from typing import List, Tuple
from backend.rag.types import Chunk

def rerank_with_bge(query: str, candidates: List[Chunk]) -> List[Tuple[Chunk, float]]:
    try:
        from FlagEmbedding import FlagReranker
    except Exception:
        return [(c, 0.0) for c in candidates]
    try:
        reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        pairs = [[query, c.text] for c in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        out = list(zip(candidates, [float(s) for s in scores]))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception:
        return [(c, 0.0) for c in candidates]
