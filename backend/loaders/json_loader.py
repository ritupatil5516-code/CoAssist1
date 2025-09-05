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
    return bd.model_dump() if hasattr(bd, "model_dump") else bd.dict()

def flatten_for_rag(data: Dict[str, Any]) -> List[dict]:
    docs: List[dict] = []
    for acc in data.get("account_summary", []):
        a = AccountSummary(**acc) if not isinstance(acc, AccountSummary) else acc
        docs.append({"text": f"ACCOUNT id={a.accountId} product={a.product or a.accountType} currentBalance={a.currentBalance} outstandingBalance={a.outstandingBalance} creditLimit={a.creditLimit} apr={a.apr or a.purchaseApr} openDate={a.openDate}", "source":"account_summary", "meta":{"id":a.accountId}})
    for st in data.get("statements", []):
        s = Statement(**st) if not isinstance(st, Statement) else st
        docs.append({"text": f"STATEMENT id={s.statementId} ym={s.ym} closingDate={s.closingDateTime} dueDate={s.dueDate} interestCharged={s.interestCharged} minimumAmountDue={s.minimumAmountDue} endingBalance={s.endingBalance} totalAmountDue={s.totalAmountDue}", "source":"statement", "meta":{"id":s.statementId,"ym":s.ym}})
    for tr in data.get("transactions", []):
        t = Transaction(**tr) if not isinstance(tr, Transaction) else tr
        docs.append({"text": f"TRANSACTION id={t.transactionId} ym={t.ym} date={t.transactionDateTime or t.postingDateTime} type={t.transactionType or t.displayTransactionType} amount={t.amount} description={t.description or t.merchantName} interestFlag={str(t.interestFlag)}", "source":"transaction", "meta":{"id":t.transactionId,"ym":t.ym,"interest":t.interestFlag}})
    for py in data.get("payments", []):
        p = Payment(**py) if not isinstance(py, Payment) else py
        pid = p.paymentId or p.scheduledPaymentId
        docs.append({"text": f"PAYMENT id={pid} ym={p.ym} date={p.paymentDateTime or p.scheduledPaymentDateTime} amount={p.amount} status={p.status or p.paymentStatus}", "source":"payment", "meta":{"id":pid,"ym":p.ym}})
    return docs
