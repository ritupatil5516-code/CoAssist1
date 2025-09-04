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

def flatten_for_rag(data: Dict[str, Any]) -> List[str]:
    docs: List[str] = []
    for acc in data.get("account_summary", []):
        docs.append(f"ACCOUNT {acc}")
    for st in data.get("statements", []):
        docs.append(f"STATEMENT {st}")
    for t in data.get("transactions", []):
        docs.append(f"TRANSACTION {t}")
    for p in data.get("payments", []):
        docs.append(f"PAYMENT {p}")
    return docs
