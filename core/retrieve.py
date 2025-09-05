from __future__ import annotations
from typing import List, Tuple
from math import exp
from datetime import datetime, timezone
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from core.indexes import Built

def _parse_dt(s: str | None):
    if not s: return None
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None

def _freshness_weight(dt_iso: str | None, lam: float) -> float:
    if not dt_iso or lam <= 0: return 1.0
    dt = _parse_dt(dt_iso)
    if not dt: return 1.0
    age_days = (datetime.now(timezone.utc) - dt).total_seconds()/86400.0
    return max(0.2, float(exp(-lam * age_days)))

def hybrid_with_freshness(built: Built, query: str, alpha: float, lam: float, kN: int) -> List[NodeWithScore]:
    vec_retriever = VectorIndexRetriever(index=built.vector_index, similarity_top_k=kN)
    vec_nodes = vec_retriever.retrieve(query)
    bm_nodes = built.bm25.retrieve(query)

    scores = {}
    for n in vec_nodes:
        scores[id(n.node)] = scores.get(id(n.node), 0.0) + alpha * n.score
    for n in bm_nodes:
        scores[id(n.node)] = scores.get(id(n.node), 0.0) + (1-alpha) * n.score

    # apply freshness
    id2node = {id(n.node): n for n in vec_nodes + bm_nodes}
    out = []
    for key, s in scores.items():
        meta = id2node[key].node.metadata or {}
        wt = _freshness_weight(meta.get("dt_iso"), lam)
        out.append(NodeWithScore(node=id2node[key].node, score=s*wt))
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:kN]
