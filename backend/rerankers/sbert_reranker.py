from __future__ import annotations
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from backend.rag.types import Chunk

# Cache the model at module scope so it's loaded once
# You can switch to "cross-encoder/ms-marco-MiniLM-L-12-v2" for a bit more accuracy
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce: CrossEncoder | None = None

def _get_model() -> CrossEncoder:
    global _ce
    if _ce is None:
        _ce = CrossEncoder(_MODEL_NAME)
    return _ce

def rerank_with_sbert(query: str, candidates: List[Chunk]) -> List[Tuple[Chunk, float]]:
    if not candidates:
        return []
    ce = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = ce.predict(pairs)  # higher = more relevant
    ranked = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    return ranked