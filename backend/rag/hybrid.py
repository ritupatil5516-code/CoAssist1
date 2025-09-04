from __future__ import annotations
from typing import List, Tuple, Dict
import math
from backend.rag.types import Chunk

def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if math.isclose(hi, lo):
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def hybrid_merge(vec_results: List[Tuple[Chunk, float]],
                 lex_results: List[Tuple[Chunk, float]],
                 alpha: float = 0.6,
                 k: int = 10) -> List[Tuple[Chunk, float]]:
    v_scores = _normalize([s for _, s in vec_results])
    l_scores = _normalize([s for _, s in lex_results])

    combined: Dict[int, float] = {}
    id_map: Dict[int, Chunk] = {}

    for i, (c, _) in enumerate(vec_results):
        combined[id(c)] = combined.get(id(c), 0.0) + alpha * v_scores[i]
        id_map[id(c)] = c
    for i, (c, _) in enumerate(lex_results):
        combined[id(c)] = combined.get(id(c), 0.0) + (1.0 - alpha) * l_scores[i]
        id_map[id(c)] = c

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(id_map[i], score) for i, score in ranked]
