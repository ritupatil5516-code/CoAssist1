from __future__ import annotations
from typing import List, Tuple, Optional, Literal
from backend.rag.types import Chunk
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store
from backend.rag.hybrid import hybrid_merge

RerankerName = Literal["none", "bge", "llm"]

def build_context(query: str, vec: FAISSStore, bm25: Optional[BM25Store],
                  use_hybrid: bool, alpha: float, kN: int, kK: int,
                  reranker: RerankerName = "none") -> List[Tuple[Chunk, float]]:
    vec_res = vec.search(query, k=kN)
    if use_hybrid and bm25 is not None:
        lex_res = bm25.search(query, k=kN)
        merged = hybrid_merge(vec_res, lex_res, alpha=alpha, k=kN)
    else:
        merged = vec_res

    candidates = [c for c, _ in merged]
    if reranker == "bge":
        from backend.rerankers.bge_reranker import rerank_with_bge
        rr = rerank_with_bge(query, candidates)
        return rr[:kK]
    if reranker == "llm":
        from backend.rerankers.llm_reranker import rerank_with_llm
        rr = rerank_with_llm(query, candidates)
        return rr[:kK]
    return merged[:kK]
