# core/short_answers.py
from __future__ import annotations
from typing import Dict, Optional, List
import re
from datetime import datetime
from dateutil import parser as dateparser

CURRENCY = "USD"

# ---------- small format helpers ----------
def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def pretty_date(dt_iso: str) -> str:
    if not dt_iso:
        return ""
    try:
        dt = dateparser.parse(dt_iso)
        # Month Day(th) Year[, HH:MM]
        base = f"{dt.strftime('%B')} {ordinal(dt.day)} {dt.year}" if getattr(dt, "day", None) else dt.strftime("%B %Y")
        if dt.hour or dt.minute:
            hhmm = dt.strftime("%H:%M")
            if hhmm != "00:00":
                base += f", {hhmm}"
        return base
    except Exception:
        return dt_iso

def fmt_money(x: Optional[float]) -> str:
    if x is None: return ""
    try: return f"${float(x):,.2f}"
    except Exception: return str(x)

# ---------- intent detection ----------
MONTH_WORDS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"

def detect_intent(q: str) -> str:
    s = q.lower().strip()

    # balances + account
    if "current balance" in s or ("balance" in s and "statement" not in s):
        return "current_balance"
    if "statement balance" in s:
        return "statement_balance"
    if "account status" in s or "status of my account" in s:
        return "account_status"

    # payments
    if ("when" in s and "last payment" in s) or ("last payment" in s and "when" in s):
        return "last_payment_date"
    if "last payment amount" in s or ("what" in s and "last payment" in s and "amount" in s):
        return "last_payment_amount"
    if "what was my last payment" in s and "amount" not in s and "when" not in s:
        return "last_payment_amount"

    # last posted transaction
    if "last posted transaction" in s:
        return "last_posted_transaction"

    # interest — last date/amount
    if ("when" in s and "interest" in s and "last" in s) or ("last interest" in s and "when" in s):
        return "interest_last_date"
    if ("how much" in s and "last interest" in s) or ("last interest amount" in s):
        return "interest_last_amount"

    # totals this month/year/statement cycle
    if ("how much" in s and "interest" in s and "this month" in s) or ("interest this month" in s):
        return "interest_total_month"
    if ("how much" in s and "interest" in s and "this year" in s) or ("interest this year" in s):
        return "interest_total_year"
    if "statement cycle" in s and "interest" in s:
        return "interest_total_cycle"

    # month-specific WHY (e.g., why interest in March?)
    if "why" in s and "interest" in s and re.search(MONTH_WORDS, s):
        return "interest_why_for_month"

    # general WHY (e.g., why was I charged interest?)
    if "why" in s and "interest" in s:
        return "interest_why_generic"

    # spend / top merchants are handled elsewhere in your app,
    # but keep generic fallback for anything else:
    return "generic"

# ---------- JSON-extraction prompts ----------
def build_extraction_prompt(intent: str, question: str, numbered_context: str) -> str:
    """
    We ask the LLM to return a very small JSON object keyed by intent.
    Keep keys minimal so we can format concise answers.
    """
    keys = {
        # balances / account
        "current_balance": ["current_balance"],
        "statement_balance": ["statement_balance", "statement_ym"],
        "account_status": ["account_status"],

        # payments / transactions
        "last_payment_amount": ["payment_amount", "payment_date"],
        "last_payment_date": ["payment_date"],
        "last_posted_transaction": ["txn_date", "txn_amount", "merchant", "txn_type"],

        # interest: last date/amount
        "interest_last_date": ["interest_date"],
        "interest_last_amount": ["interest_amount", "interest_date"],

        # interest totals
        "interest_total_month": ["interest_total", "ym"],
        "interest_total_year":  ["interest_total", "year"],
        "interest_total_cycle": ["interest_total", "cycle_open", "cycle_close"],

        # interest WHY
        "interest_why_generic": [
            "cause",                       # residual/trailing | carried balance | cash advance | late fee interest | etc.
            "statement_open", "statement_close",
            "interest_amount", "interest_date",
            "transactions"                 # [{merchant, amount, date_iso}]
        ],
        "interest_why_for_month": [
            "ym",                          # YYYY-MM you’re explaining
            "cause",
            "statement_open", "statement_close",
            "interest_amount", "interest_date",
            "transactions"
        ],

        # generic fallback
        "generic": ["answer_text"],
    }
    wanted = ", ".join(f'"{k}"' for k in keys.get(intent, ["answer_text"]))

    # guardrails that matter for your data:
    return f"""
You are a banking assistant. Use only the context below.
Return a MINIMAL JSON object having exactly these keys: {wanted}

Formatting rules:
- If a value is unknown, omit that key.
- Money as NUMBER (no $ or commas). Dates as ISO (YYYY-MM-DD or full ISO).
- For "transactions", return an array of objects with keys: merchant, amount, date_iso.
- IMPORTANT date policy for transactions: use "transactionDateTime"; if missing use "postingDateTime"; ignore "authDateTime".
- Prefer statement fields for interest totals/amounts when available (e.g., "interestCharged"); otherwise use interest transactions.

Context:
{numbered_context}

User question: {question}

Return JSON only:
"""

# ---------- final formatting ----------
def _one_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    if not s:
        return ""
    first = re.split(r"(?<=[.!?])\s+", s)[0]
    return first if first.endswith((".", "!", "?")) else first + "."

def format_answer(intent: str, data: Dict) -> str:
    # balances / account
    if intent == "current_balance" and "current_balance" in data:
        return f"Your current balance is {fmt_money(data['current_balance'])}."
    if intent == "statement_balance":
        bal = data.get("statement_balance")
        ym  = data.get("statement_ym")
        if bal is not None and ym:  return f"Your statement balance for {ym} is {fmt_money(bal)}."
        if bal is not None:         return f"Your statement balance is {fmt_money(bal)}."

    if intent == "account_status" and "account_status" in data:
        return f"Your account status is {data['account_status']}."

    # payments
    if intent == "last_payment_amount":
        amt = data.get("payment_amount")
        dt  = pretty_date(data.get("payment_date", ""))
        if amt is not None and dt:  return f"Your last payment was {fmt_money(amt)} on {dt}."
        if amt is not None:         return f"Your last payment was {fmt_money(amt)}."
    if intent == "last_payment_date":
        dt = pretty_date(data.get("payment_date", ""))
        if dt: return f"Your last payment date was {dt}."

    # last posted txn
    if intent == "last_posted_transaction":
        dt  = pretty_date(data.get("txn_date",""))
        amt = data.get("txn_amount")
        merch = data.get("merchant")
        typ = data.get("txn_type")
        bits = []
        if amt is not None: bits.append(fmt_money(amt))
        if merch: bits.append(str(merch))
        if typ:   bits.append(str(typ).lower())
        core = " • ".join(bits) if bits else "No details available"
        return f"Your last posted transaction was on {dt}: {core}." if dt else f"Your last posted transaction: {core}."

    # interest: last date/amount
    if intent == "interest_last_date":
        dt = pretty_date(data.get("interest_date",""))
        if dt: return f"Last interest was applied on {dt}."
    if intent == "interest_last_amount":
        amt = data.get("interest_amount")
        dt  = pretty_date(data.get("interest_date",""))
        if amt is not None and dt:  return f"Your last interest charge was {fmt_money(amt)} on {dt}."
        if amt is not None:         return f"Your last interest charge was {fmt_money(amt)}."

    # interest totals
    if intent == "interest_total_month":
        tot = data.get("interest_total")
        ym  = data.get("ym")
        if tot is not None and ym:  return f"Your interest for {ym} is {fmt_money(tot)}."
        if tot is not None:         return f"Your interest this month is {fmt_money(tot)}."
    if intent == "interest_total_year":
        tot = data.get("interest_total")
        yr  = data.get("year")
        if tot is not None and yr:  return f"Your interest for {yr} is {fmt_money(tot)}."
        if tot is not None:         return f"Your interest this year is {fmt_money(tot)}."
    if intent == "interest_total_cycle":
        tot = data.get("interest_total")
        op  = pretty_date(data.get("cycle_open",""))
        cl  = pretty_date(data.get("cycle_close",""))
        if tot is not None and (op or cl):
            win = f"{op}–{cl}".strip("–")
            return f"Your interest for the statement cycle {win} is {fmt_money(tot)}."
        if tot is not None:
            return f"Your interest for the statement cycle is {fmt_money(tot)}."

    # interest WHY (generic or month-specific)
    if intent in {"interest_why_generic", "interest_why_for_month"}:
        cause = data.get("cause")
        op = pretty_date(data.get("statement_open",""))
        cl = pretty_date(data.get("statement_close",""))
        win = f" during {op}–{cl}" if (op or cl) else ""
        amt = data.get("interest_amount")
        dt  = pretty_date(data.get("interest_date",""))
        lead = []
        if amt is not None and dt:
            lead.append(f"Interest of {fmt_money(amt)} was applied on {dt}")
        elif amt is not None:
            lead.append(f"Interest of {fmt_money(amt)} was applied")
        if cause:
            lead.append(f"due to {cause}")
        msg = " ".join(lead) + (win if lead else "")
        # include up to two responsible transactions if present
        txns: List[Dict] = data.get("transactions") or []
        if txns:
            items = []
            for t in txns[:2]:
                m = str(t.get("merchant") or "").strip()
                a = fmt_money(t.get("amount"))
                d = pretty_date(t.get("date_iso",""))
                if m and a and d: items.append(f"{m} {a} on {d}")
                elif m and a:     items.append(f"{m} {a}")
            if items:
                msg += f". Examples: " + "; ".join(items)
        return _one_sentence(msg) or "Interest was charged due to trailing balance."

    # generic fallback
    if "answer_text" in data and isinstance(data["answer_text"], str):
        return _one_sentence(data["answer_text"]) or "I couldn’t find that."
    return "I couldn’t find that."