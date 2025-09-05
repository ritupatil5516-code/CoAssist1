import re

def _extract_ym(q: str) -> str | None:
    # matches 2025-08 or Aug 2025 / August 2025
    m = re.search(r"(\d{4})-(\d{2})", q)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s\-_/]+(20\d{2})", q, re.I)
    if m:
        month_map = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06","jul":"07",
                     "aug":"08","sep":"09","sept":"09","oct":"10","nov":"11","dec":"12"}
        return f"{m.group(2)}-{month_map[m.group(1)[:3].lower()]}"
    return None

def build_context(query: str, vec: FAISSStore, bm25: Optional[BM25Store],
                  use_hybrid: bool, alpha: float, kN: int, kK: int,
                  reranker: RerankerName = "none"):
    vec_res = vec.search(query, k=kN)
    merged = vec_res
    if use_hybrid and bm25 is not None:
        lex_res = bm25.search(query, k=kN)
        from backend.rag.hybrid import hybrid_merge
        merged = hybrid_merge(vec_res, lex_res, alpha=alpha, k=kN)

    # soft-bias by ym if present
    ym = _extract_ym(query)
    if ym:
        boosted = []
        for c, s in merged:
            bonus = 0.15 if (c.meta.get("ym") == ym) else 0.0
            boosted.append((c, s + bonus))
        boosted.sort(key=lambda x: x[1], reverse=True)
        merged = boosted

    # optional reranker
    candidates = [c for c, _ in merged]
    if reranker == "llm":
        from backend.rerankers.llm_reranker import rerank_with_llm
        rr = rerank_with_llm(query, candidates)
        return rr[:kK]
    return merged[:kK]