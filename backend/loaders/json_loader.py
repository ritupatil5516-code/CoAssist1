from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import json

from backend.models.banking import BankingData, AccountSummary, Statement, Transaction, Payment

def _read(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_all(data_dir: str) -> dict:
    d = Path(data_dir)
    raw = {
        "account_summary": _read(d/"account_summary.json") if (d/"account_summary.json").exists() else [],
        "statements": _read(d/"statements.json") if (d/"statements.json").exists() else [],
        "transactions": _read(d/"transactions.json") if (d/"transactions.json").exists() else [],
        "payments": _read(d/"payments.json") if (d/"payments.json").exists() else [],
    }
    bd = BankingData(
        account_summary=[AccountSummary(**x) for x in raw["account_summary"]],
        statements=[Statement(**x) for x in raw["statements"]],
        transactions=[Transaction(**x) for x in raw["transactions"]],
        payments=[Payment(**x) for x in raw["payments"]],
    )
    # pydantic v2 uses model_dump, v1 uses dict()
    return bd.model_dump() if hasattr(bd, "model_dump") else bd.dict()

def _pd_to_dict(obj) -> dict:
    """Pydantic v2/v1 safe to-dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj  # already a dict

def _short_header(kind: str, d: dict) -> str:
    """Human-readable header to help BM25/semantic retrieval."""
    # try to extract common bits if present
    id_ = d.get("statementId") or d.get("transactionId") or d.get("paymentId") or d.get("scheduledPaymentId") or d.get("accountId")
    ym = d.get("ym")
    extra = []
    if kind == "statement":
        if d.get("interestCharged") is not None: extra.append(f"interestCharged={d['interestCharged']}")
        if d.get("endingBalance") is not None: extra.append(f"endingBalance={d['endingBalance']}")
    if kind == "transaction":
        if d.get("interestFlag"): extra.append("interestFlag=true")
        if d.get("amount") is not None: extra.append(f"amount={d['amount']}")
    if kind == "payment":
        if d.get("amount") is not None: extra.append(f"amount={d['amount']}")
    hdr = f"{kind.upper()} id={id_} ym={ym}" if id_ or ym else kind.upper()
    return hdr + ((" " + " ".join(extra)) if extra else "")

def flatten_for_rag(data: Dict[str, Any]) -> List[dict]:
    """
    Generic flattener:
      - text:  "<KIND> <short-header>\nJSON::<compact json>"
      - source: "statement" | "transaction" | ...
      - meta:   {"id": ..., "ym": ..., "raw": <full dict>}
    """
    docs: List[dict] = []

    # 1) Account summary
    for acc in data.get("account_summary", []):
        a = acc if isinstance(acc, dict) else _pd_to_dict(acc)
        item = {
            "text": _short_header("account_summary", a) + "\nJSON::" + json.dumps(a, separators=(",", ":"), ensure_ascii=False),
            "source": "account_summary",
            "meta": {"id": a.get("accountId"), "ym": a.get("ym"), "raw": a},
        }
        docs.append(item)

    # 2) Statements
    for st in data.get("statements", []):
        s = st if isinstance(st, dict) else _pd_to_dict(st)
        item = {
            "text": _short_header("statement", s) + "\nJSON::" + json.dumps(s, separators=(",", ":"), ensure_ascii=False),
            "source": "statement",
            "meta": {"id": s.get("statementId"), "ym": s.get("ym"), "raw": s},
        }
        docs.append(item)

    # 3) Transactions
    for tr in data.get("transactions", []):
        t = tr if isinstance(tr, dict) else _pd_to_dict(tr)
        item = {
            "text": _short_header("transaction", t) + "\nJSON::" + json.dumps(t, separators=(",", ":"), ensure_ascii=False),
            "source": "transaction",
            "meta": {"id": t.get("transactionId"), "ym": t.get("ym"), "raw": t},
        }
        docs.append(item)

    # 4) Payments
    for py in data.get("payments", []):
        p = py if isinstance(py, dict) else _pd_to_dict(py)
        pid = p.get("paymentId") or p.get("scheduledPaymentId")
        item = {
            "text": _short_header("payment", p) + "\nJSON::" + json.dumps(p, separators=(",", ":"), ensure_ascii=False),
            "source": "payment",
            "meta": {"id": pid, "ym": p.get("ym"), "raw": p},
        }
        docs.append(item)

    return docs