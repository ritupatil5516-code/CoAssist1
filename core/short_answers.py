# core/short_answers.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import re
from datetime import datetime
from dateutil import parser as dateparser

CURRENCY = "USD"

# -------------------------
# Intent detection
# -------------------------
def detect_intent(q: str) -> str:
    s = q.lower().strip()

    # order matters for some overlaps
    if "current balance" in s or ("balance" in s and "statement" not in s):
        return "current_balance"
    if "statement balance" in s:
        return "statement_balance"

    # payments
    if ("when" in s and "last payment" in s) or ("last payment" in s and "when" in s):
        return "last_payment_date"
    if ("what" in s and "last payment" in s and "amount" in s) or ("last payment amount" in s):
        return "last_payment_amount"
    if "what was my last payment" in s or ("last payment" in s and "amount" not in s and "when" not in s):
        return "last_payment_amount"

    # transactions
    if "last posted transaction" in s:
        return "last_posted_transaction"

    # account
    if "account status" in s or "status of my account" in s:
        return "account_status"

    # interest totals
    if "total interest" in s or ("interest" in s and "total" in s):
        return "interest_total_month"
    if "interest this month" in s or "my interest this month" in s:
        return "interest_total_month"

    # interest date & explanation
    if ("when" in s and "interest" in s) or ("last interest" in s and "when" in s):
        return "interest_last_date"
    if ("why" in s and "interest" in s) or ("what caused the interest" in s) or ("responsible for interest" in s):
        return "interest_why"

    # spend
    if "where did i spend most" in s or "top merchant" in s:
        return "top_merchants"
    if ("how much did i spend" in s and "year" in s):
        return "spend_total_year"
    if ("how much did i spend" in s and "month" in s) or ("how much did i spend this month" in s):
        return "spend_total_month"

    return "generic"


# -------------------------
# Formatting helpers
# -------------------------
def ordinal(n: int) -> str:
    # 1st, 2nd, 3rd, 4th ...
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def pretty_date(dt_iso: str) -> str:
    if not dt_iso:
        return ""
    try:
        dt = dateparser.parse(dt_iso)
        base = dt.strftime("%B %Y")
        # day with ordinal if present
        if hasattr(dt, "day") and dt.day:
            base = f"{dt.strftime('%B')} {ordinal(dt.day)} {dt.year}"
        # include time if present and non-midnight
        if dt.hour or dt.minute:
            hhmm = dt.strftime("%H:%M")
            if hhmm != "00:00":
                base += f", {hhmm}"
        return base
    except Exception:
        return dt_iso

def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

# -------------------------
# Prompt builder for extraction
# -------------------------
def build_extraction_prompt(intent: str, question: str, numbered_context: str) -> str:
    """
    Ask the LLM to return ONLY tiny JSON we can format.
    """
    fields = {
        "current_balance": ["current_balance"],
        "statement_balance": ["statement_balance", "statement_ym"],
        "last_payment_amount": ["payment_amount", "payment_date"],
        "last_payment_date": ["payment_date"],
        "last_posted_transaction": ["txn_date", "txn_amount", "merchant", "txn_type"],
        "account_status": ["account_status"],
        "interest_total_month": ["interest_total", "ym"],
        "interest_last_date": ["interest_date"],
        "interest_why": [
            "cause",                    # short phrase: trailing interest | cash advance interest | carried balance | late fee interest | etc.
            "statement_open",           # ISO date
            "statement_close",          # ISO date
            "transactions"              # list of {merchant, amount, date_iso} for the statement window
        ],
        "top_merchants": ["merchant", "amount", "ym"],
        "spend_total_month": ["spend_total", "ym"],
        "spend_total_year": ["spend_total", "year"],
        "generic": ["answer_text"],
    }
    wanted = ", ".join(f'"{k}"' for k in fields.get(intent, ["answer_text"]))

    # Keep the schema extremely strict to avoid verbose answers
    return f"""
You are a banking assistant. Read only the context and return a MINIMAL JSON object with exactly these keys: {wanted}.
Rules:
- If unknown, omit the key.
- Money as NUMBER (no $). Dates as ISO (YYYY-MM-DD or full ISO).
- For "transactions", return a JSON array of objects with keys: merchant, amount, date_iso.
- Do not add commentary, do not add extra keys, do not wrap in code fences.

Context:
{numbered_context}

User question: {question}

Return JSON only:
"""

# -------------------------
# Final answer formatter
# -------------------------
def format_answer(intent: str, data: Dict) -> str:
    # Ensure minimal answers
    if intent == "current_balance" and "current_balance" in data:
        return f"Your current balance is {fmt_money(data['current_balance'])}."

    if intent == "statement_balance":
        bal = data.get("statement_balance")
        ym = data.get("statement_ym")
        if bal is not None and ym:
            return f"Your statement balance for {ym} is {fmt_money(bal)}."
        if bal is not None:
            return f"Your statement balance is {fmt_money(bal)}."

    if intent == "last_payment_amount":
        amt = data.get("payment_amount")
        dt = pretty_date(data.get("payment_date", ""))
        if amt is not None and dt:
            return f"Your last payment was {fmt_money(amt)} on {dt}."
        if amt is not None:
            return f"Your last payment was {fmt_money(amt)}."

    if intent == "last_payment_date":
        dt = pretty_date(data.get("payment_date", ""))
        if dt:
            return f"Your last payment date was {dt}."

    if intent == "last_posted_transaction":
        dt = pretty_date(data.get("txn_date", ""))
        amt = data.get("txn_amount")
        merch = data.get("merchant")
        typ = data.get("txn_type")
        parts = []
        if amt is not None: parts.append(fmt_money(amt))
        if merch: parts.append(str(merch))
        if typ: parts.append(str(typ).lower())
        core = " • ".join(parts) if parts else "No details available"
        if dt:
            return f"Your last posted transaction was on {dt}: {core}."
        return f"Your last posted transaction: {core}."

    if intent == "account_status" and "account_status" in data:
        return f"Your account status is {data['account_status']}."

    if intent == "interest_total_month":
        tot = data.get("interest_total")
        ym = data.get("ym")
        if tot is not None and ym:
            return f"Your total interest for {ym} is {fmt_money(tot)}."
        if tot is not None:
            return f"Your total interest is {fmt_money(tot)}."

    if intent == "interest_last_date":
        dt = pretty_date(data.get("interest_date", ""))
        if dt:
            return f"Last interest was applied on {dt}."

    if intent == "interest_why":
        cause = data.get("cause")
        open_iso = data.get("statement_open")
        close_iso = data.get("statement_close")
        window = ""
        if open_iso or close_iso:
            o = pretty_date(open_iso or "")
            c = pretty_date(close_iso or "")
            if o and c:
                window = f" during {o}–{c}"
        txns: List[Dict] = data.get("transactions") or []
        if txns:
            # show up to 2 illustrative transactions
            items = []
            for t in txns[:2]:
                merch = str(t.get("merchant") or "").strip()
                amt = fmt_money(t.get("amount"))
                d = pretty_date(t.get("date_iso", ""))
                if merch and amt and d:
                    items.append(f"{merch} {amt} on {d}")
                elif merch and amt:
                    items.append(f"{merch} {amt}")
            txn_str = "; ".join(items)
            if cause:
                return f"Interest was charged due to {cause}{window}: {txn_str}."
            return f"Interest was charged{window}: {txn_str}."
        # No transactions → likely trailing/residual interest
        if cause:
            return f"Interest was charged due to {cause}{window}."
        return f"Interest was charged as trailing interest{window}."

    if intent == "spend_total_month":
        tot = data.get("spend_total")
        ym = data.get("ym")
        if tot is not None and ym:
            return f"You spent {fmt_money(tot)} in {ym}."
        if tot is not None:
            return f"You spent {fmt_money(tot)}."

    if intent == "spend_total_year":
        tot = data.get("spend_total")
        yr = data.get("year")
        if tot is not None and yr:
            return f"You spent {fmt_money(tot)} in {yr}."
        if tot is not None:
            return f"You spent {fmt_money(tot)}."

    # Generic or fallback
    if "answer_text" in data and isinstance(data["answer_text"], str):
        # Keep it short: first sentence only
        s = re.split(r"[.!?]", data["answer_text"].strip())[0]
        return (s + ".") if s else "I couldn't find that."
    return "I couldn't find that."