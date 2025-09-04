from __future__ import annotations
from typing import Dict, Any, List
import json, os

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def load_all(data_dir: str) -> Dict[str, Any]:
    files = {
        "account_summary": os.path.join(data_dir, "account_summary.json"),
        "payments": os.path.join(data_dir, "payments.json"),
        "statements": os.path.join(data_dir, "statements.json"),
        "transactions": os.path.join(data_dir, "transactions.json"),
    }
    out: Dict[str, Any] = {}
    for k, p in files.items():
        out[k] = load_json(p) if os.path.exists(p) else []
    return out

def flatten_for_rag(data: Dict[str, Any]) -> List[dict | str]:
    docs: List[dict | str] = []
    for acc in data.get("account_summary", []):
        docs.append({"text": f"ACCOUNT {acc}", "source": "account_summary", "meta": {"id": acc.get("accountId")}})
    for st in data.get("statements", []):
        docs.append({"text": f"STATEMENT {st}", "source": "statement", "meta": {"id": st.get("statementId"), "dueDate": st.get("dueDate")}})
    for t in data.get("transactions", []):
        docs.append({"text": f"TRANSACTION {t}", "source": "transaction", "meta": {"id": t.get("transactionId"), "date": t.get("transactionDateTime")}})
    for p in data.get("payments", []):
        docs.append({"text": f"PAYMENT {p}", "source": "payment", "meta": {"id": p.get("paymentId") or p.get("scheduledPaymentId"), "date": p.get("paymentDateTime")}})
    return docs