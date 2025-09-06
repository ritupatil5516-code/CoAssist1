from __future__ import annotations
from typing import List
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
    vec_nodes = VectorIndexRetriever(index=built.vector_index, similarity_top_k=kN).retrieve(query)
    bm_nodes = built.bm25.retrieve(query)

    scores = {}
    for n in vec_nodes:
        scores[id(n.node)] = scores.get(id(n.node), 0.0) + alpha * n.score
    for n in bm_nodes:
        scores[id(n.node)] = scores.get(id(n.node), 0.0) + (1-alpha) * n.score

    id2node = {id(n.node): n for n in vec_nodes + bm_nodes}
    out = []
    for key, s in scores.items():
        meta = id2node[key].node.metadata or {}
        wt = _freshness_weight(meta.get("dt_iso"), lam)
        out.append(NodeWithScore(node=id2node[key].node, score=s*wt))
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:kN]

def filter_spend_current_month(nodes, now_utc=None):
    if not nodes:
        return nodes
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    out = []
    for n in nodes:
        md = getattr(n.node, "metadata", {}) or {}
        if md.get("kind") != "transaction":
            out.append(n)  # allow non-transaction docs through
            continue
        if not md.get("spend_candidate", False):
            continue
        dt_iso = md.get("dt_iso")
        if not dt_iso:
            continue
        try:
            dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if month_start <= dt <= now_utc:
            out.append(n)
    return out
