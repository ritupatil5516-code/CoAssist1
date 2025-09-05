from __future__ import annotations
from pydantic import BaseModel, field_validator
from typing import Optional, List
from pathlib import Path
import json, PyPDF2
from core.utils import parse_iso, ym_from_dt

class AccountSummary(BaseModel):
    accountId: str
    accountStatus: Optional[str] = None
    accountNumberLast4: Optional[str] = None
    creditLimit: Optional[float] = None
    currentBalance: Optional[float] = None

class Statement(BaseModel):
    statementId: str
    closingDateTime: Optional[str] = None
    openingDateTime: Optional[str] = None
    dueDate: Optional[str] = None
    minimumAmountDue: Optional[float] = None
    totalAmountDue: Optional[float] = None
    endingBalance: Optional[float] = None
    interestCharged: Optional[float] = None
    ym: Optional[str] = None
    @field_validator("ym", mode="before")
    @classmethod
    def _ym(cls, v, info):
        dt = parse_iso(info.data.get("closingDateTime") or info.data.get("openingDateTime") or info.data.get("dueDate"))
        return ym_from_dt(dt)

class Transaction(BaseModel):
    transactionId: str
    transactionDateTime: Optional[str] = None
    postingDateTime: Optional[str] = None
    transactionType: Optional[str] = None
    displayTransactionType: Optional[str] = None
    merchantName: Optional[str] = None
    description: Optional[str] = None
    amount: float
    ym: Optional[str] = None
    interestFlag: bool = False
    @field_validator("ym", mode="before")
    @classmethod
    def _ym(cls, v, info):
        dt = parse_iso(info.data.get("transactionDateTime") or info.data.get("postingDateTime"))
        return ym_from_dt(dt)
    @field_validator("interestFlag", mode="before")
    @classmethod
    def _interest(cls, v, info):
        t = (info.data.get("transactionType") or "").upper()
        dt = (info.data.get("displayTransactionType") or "").lower()
        m = (info.data.get("merchantName") or "").lower()
        return ("INTEREST" in t) or (dt == "interest_charged") or ("interest" in m)

class Payment(BaseModel):
    paymentId: Optional[str] = None
    scheduledPaymentId: Optional[str] = None
    paymentDateTime: Optional[str] = None
    scheduledPaymentDateTime: Optional[str] = None
    amount: Optional[float] = None
    ym: Optional[str] = None
    @field_validator("ym", mode="before")
    @classmethod
    def _ym(cls, v, info):
        dt = parse_iso(info.data.get("paymentDateTime") or info.data.get("scheduledPaymentDateTime"))
        return ym_from_dt(dt)

class Bundle(BaseModel):
    account_summary: List[AccountSummary] = []
    statements: List[Statement] = []
    transactions: List[Transaction] = []
    payments: List[Payment] = []

def _read_json(p: Path):
    if not p.exists(): return []
    return json.loads(p.read_text(encoding="utf-8"))

def load_bundle(data_dir: str) -> Bundle:
    d = Path(data_dir)
    a = d/"account-summary.json"
    if not a.exists():
        a = d/"account_summary.json"
    return Bundle(
        account_summary=[AccountSummary(**x) for x in _read_json(a)],
        statements=[Statement(**x) for x in _read_json(d/"statements.json")],
        transactions=[Transaction(**x) for x in _read_json(d/"transactions.json")],
        payments=[Payment(**x) for x in _read_json(d/"payments.json")],
    )

def load_agreement_text(data_dir: str) -> str:
    p = Path(data_dir)/"agreement.pdf"
    if not p.exists(): return ""
    with p.open("rb") as f:
        r = PyPDF2.PdfReader(f)
        return "\n".join((pg.extract_text() or "") for pg in r.pages)
