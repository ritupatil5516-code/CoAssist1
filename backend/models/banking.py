from __future__ import annotations
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

IsoStr = str  # keep raw strings for round-trip while parsing to datetime fields too

def _parse_dt(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None

class AccountSummary(BaseModel):
    accountId: str = Field(..., description="Unique account id")
    product: Optional[str] = None
    accountType: Optional[str] = None
    currentBalance: Optional[float] = None
    outstandingBalance: Optional[float] = None
    creditLimit: Optional[float] = None
    apr: Optional[float] = None
    purchaseApr: Optional[float] = None
    openDate: Optional[IsoStr] = None

class Statement(BaseModel):
    statementId: str
    openingDateTime: Optional[IsoStr] = None
    closingDateTime: Optional[IsoStr] = None
    dueDate: Optional[IsoStr] = None
    minimumAmountDue: Optional[float] = None
    totalAmountDue: Optional[float] = None
    endingBalance: Optional[float] = None
    interestCharged: Optional[float] = None

    # parsed dates
    opening_dt: Optional[datetime] = None
    closing_dt: Optional[datetime] = None
    due_dt: Optional[datetime] = None
    ym: Optional[str] = None

    @field_validator("opening_dt", mode="before")
    @classmethod
    def _set_opening_dt(cls, v, info):
        return _parse_dt(info.data.get("openingDateTime"))

    @field_validator("closing_dt", mode="before")
    @classmethod
    def _set_closing_dt(cls, v, info):
        return _parse_dt(info.data.get("closingDateTime"))

    @field_validator("due_dt", mode="before")
    @classmethod
    def _set_due_dt(cls, v, info):
        return _parse_dt(info.data.get("dueDate"))

    @field_validator("ym", mode="before")
    @classmethod
    def _set_ym(cls, v, info):
        dt = _parse_dt(info.data.get("closingDateTime") or info.data.get("openingDateTime") or info.data.get("dueDate"))
        return f"{dt.year:04d}-{dt.month:02d}" if dt else None

class Transaction(BaseModel):
    transactionId: str
    transactionDateTime: Optional[IsoStr] = None
    postingDateTime: Optional[IsoStr] = None
    transactionType: Optional[str] = None
    displayTransactionType: Optional[str] = None
    description: Optional[str] = None
    merchantName: Optional[str] = None
    amount: float

    # derived
    date_dt: Optional[datetime] = None
    ym: Optional[str] = None
    interestFlag: bool = False

    @field_validator("date_dt", mode="before")
    @classmethod
    def _set_date_dt(cls, v, info):
        return _parse_dt(info.data.get("transactionDateTime") or info.data.get("postingDateTime"))

    @field_validator("ym", mode="before")
    @classmethod
    def _set_ym(cls, v, info):
        dt = _parse_dt(info.data.get("transactionDateTime") or info.data.get("postingDateTime"))
        return f"{dt.year:04d}-{dt.month:02d}" if dt else None

    @field_validator("interestFlag", mode="before")
    @classmethod
    def _set_interest(cls, v, info):
        ttype = (info.data.get("transactionType") or info.data.get("displayTransactionType") or "").upper()
        desc = (info.data.get("description") or info.data.get("merchantName") or "").upper()
        return ("INTEREST" in ttype) or ("INTEREST" in desc)

class Payment(BaseModel):
    paymentId: Optional[str] = None
    scheduledPaymentId: Optional[str] = None
    paymentDateTime: Optional[IsoStr] = None
    scheduledPaymentDateTime: Optional[IsoStr] = None
    amount: Optional[float] = None
    status: Optional[str] = None
    paymentStatus: Optional[str] = None

    date_dt: Optional[datetime] = None
    ym: Optional[str] = None

    @field_validator("date_dt", mode="before")
    @classmethod
    def _set_date_dt(cls, v, info):
        return _parse_dt(info.data.get("paymentDateTime") or info.data.get("scheduledPaymentDateTime"))

    @field_validator("ym", mode="before")
    @classmethod
    def _set_ym(cls, v, info):
        dt = _parse_dt(info.data.get("paymentDateTime") or info.data.get("scheduledPaymentDateTime"))
        return f"{dt.year:04d}-{dt.month:02d}" if dt else None

class BankingData(BaseModel):
    account_summary: list[AccountSummary] = []
    statements: list[Statement] = []
    transactions: list[Transaction] = []
    payments: list[Payment] = []