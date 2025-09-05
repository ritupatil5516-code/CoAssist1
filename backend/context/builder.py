from __future__ import annotations
from typing import List, Tuple, Optional, Literal
import re

# ✅ required imports (avoids NameError)
from backend.rag.types import Chunk
from backend.rag.faiss_store import FAISSStore
from backend.rag.lexical import BM25Store
from backend.rag.hybrid import hybrid_merge

RerankerName = Literal["none", "llm"]

def _extract_ym(q: str) -> Optional[str]:
    """
    Try to pull a year-month from the user query.
    Supports '2025-08' or 'Aug 2025' / 'August 2025'.
    """
    # 1) YYYY-MM
    m = re.search(r"(\d{4})-(\d{2})", q)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # 2) Month YYYY
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s\-/_,]+(20\d{2})", q, re.I)
    if m:
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
            "jul": "07", "aug": "08", "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12"
        }
        return f"{m.group(2)}-{month_map[m.group(1)[:3].lower()]}"
    return None


def build_context(
    query: str,
    vec: FAISSStore,
    bm25: Optional[BM25Store],
    use_hybrid: bool,
    alpha: float,
    kN: int,
    kK: int,
    reranker: RerankerName = "none",
) -> List[Tuple[Chunk, float]]:
    """
    Retrieve candidates with FAISS (and BM25 if enabled), softly bias to a detected ym,
    then optionally re-rank with the LLM reranker. Return top-K (Chunk, score).
    """
    # Vector retrieval
    vec_res = vec.search(query, k=kN)
    merged = vec_res

    # Optional hybrid (vector + lexical)
    if use_hybrid and bm25 is not None:
        lex_res = bm25.search(query, k=kN)
        merged = hybrid_merge(vec_res, lex_res, alpha=alpha, k=kN)

    # Soft bias by ym if present (small bonus to keep recall but nudge precision)
    ym = _extract_ym(query)
    if ym:
        boosted = []
        for c, s in merged:
            bonus = 0.15 if (c.meta.get("ym") == ym) else 0.0
            boosted.append((c, s + bonus))
        boosted.sort(key=lambda x: x[1], reverse=True)
        merged = boosted

    # Optional LLM-based reranking
    if reranker == "llm":
        from backend.rerankers.llm_reranker import rerank_with_llm
        reranked = rerank_with_llm(query, [c for c, _ in merged])
        return reranked[:kK]

    # No reranker: return merged scores
    return merged[:kK]