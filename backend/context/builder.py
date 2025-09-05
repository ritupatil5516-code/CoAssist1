from __future__ import annotations
from typing import List, Tuple
from backend.rag.types import Chunk
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store
from backend.rag.hybrid import hybrid_merge

def build_context(query: str, vec: FAISSStore, bm25: BM25Store | None,
                  use_hybrid: bool, alpha: float, kN: int, kK: int) -> List[Tuple[Chunk, float]]:
    vec_res = vec.search(query, k=kN)
    if use_hybrid and bm25 is not None:
        lex_res = bm25.search(query, k=kN)
        merged = hybrid_merge(vec_res, lex_res, alpha=alpha, k=kN)
    else:
        merged = vec_res
    return merged[:kK]