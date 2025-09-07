# core/interest_explain.py
from __future__ import annotations
from typing import Optional, Tuple, List
from datetime import datetime, timezone

def _to_dt(iso: Optional[str]) -> Optional[datetime]:
    if not iso: return None
    try: return datetime.fromisoformat(iso.replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception: return None

def is_why_interest_intent(q: str) -> bool:
    s = q.lower()
    keys = [
        "why was my interest", "why was i charged interest",
        "what caused the interest", "why interest",
        "transactions responsible for interest"
    ]
    return any(k in s for k in keys)

def _pick_last_interest_statement(nodes) -> Optional[dict]:
    best, best_dt = None, None
    for n in nodes:
        md = getattr(n, "node", n).metadata or {}
        if md.get("kind") != "statement":
            continue
        try:
            amt = float(md.get("interestCharged") or 0.0)
        except Exception:
            amt = 0.0
        if amt <= 0: continue
        dt = _to_dt(md.get("closingDateTime") or md.get("dt_iso"))
        if dt and (best_dt is None or dt > best_dt):
            best, best_dt = md, dt
    return best

def _within(dt: Optional[datetime], start: Optional[datetime], end: Optional[datetime]) -> bool:
    return bool(dt and start and end and start <= dt <= end)

def _txns_in_window(nodes, start_iso: str, end_iso: str):
    sdt, edt = _to_dt(start_iso), _to_dt(end_iso)
    out = []
    for n in nodes:
        node_obj = getattr(n, "node", n)
        md = node_obj.metadata or {}
        if md.get("kind") != "transaction": continue
        dt = _to_dt(md.get("dt_iso"))
        if _within(dt, sdt, edt):
            out.append(n)
    return out

def build_interest_context(built, q: str, kN: int = 80):
    # Broad recall
    wide = list(built.vector_index.as_retriever(similarity_top_k=kN).retrieve(q))
    try:
        wide += list(built.bm25.retrieve(q))
    except Exception:
        pass

    stmt_md = _pick_last_interest_statement(wide)
    if not stmt_md:
        return None, []

    open_iso = stmt_md.get("openingDateTime")
    close_iso = stmt_md.get("closingDateTime") or stmt_md.get("dt_iso")
    if not (open_iso and close_iso):
        return stmt_md, []

    txns = _txns_in_window(wide, open_iso, close_iso)

    # dedupe + cap
    seen = set()
    ordered = sorted(txns, key=lambda x: (getattr(x, "node", x).metadata or {}).get("dt_iso") or "", reverse=True)
    out = []
    for n in ordered:
        md = getattr(n, "node", n).metadata or {}
        key = md.get("transactionId") or (md.get("merchantName"), md.get("amount"), md.get("dt_iso"))
        if key in seen: continue
        seen.add(key)
        out.append(n)
        if len(out) >= 20: break

    return stmt_md, out