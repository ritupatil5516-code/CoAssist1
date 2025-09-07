# core/short_answers.py
from __future__ import annotations
from typing import Dict, List, Optional
import re
from datetime import datetime
from dateutil import parser as dateparser

CURRENCY = "USD"

# ----------------------------
# Intent detection
# ----------------------------
def detect_intent(q: str) -> str:
    s = (q or "").lower().strip()

    # Explanations first (more specific)
    if ("why" in s and "interest" in s) or ("reason" in s and "interest" in s):
        return "interest_reason"

    # Interest facts
    if "how much" in s and "interest" in s and ("last" in s or "previous" in s):
        return "interest_amount_last"
    if "how much" in s and "interest" in s:
        # allow generic "how much interest did i get charged"
        return "interest_amount"
    if "interest this month" in s or ("interest" in s and "this month" in s):
        return "interest_total_month"
    if "interest this year" in s or ("interest" in s and "this year" in s):
        return "interest_total_year"
    if ("how much" in s and "interest" in s and "statement" in s) or ("statement cycle" in s and "interest" in s):
        return "interest_total_statement"

    # Dates
    if ("when" in s and "last interest" in s) or ("when" in s and "interest" in s and "last" in s):
        return "interest_date_last"

    # Payments
    if ("when" in s and "last payment" in s) or ("last payment" in s and "when" in s):
        return "last_payment_date"
    if ("what" in s and "last payment" in s and "amount" in s) or ("last payment amount" in s):
        return "last_payment_amount"
    if "what was my last payment" in s and "amount" not in s and "when" not in s:
        return "last_payment_amount"

    # Balances / status
    if "statement balance" in s:
        return "statement_balance"
    if "current balance" in s or ("balance" in s and "statement" not in s):
        return "current_balance"
    if "account status" in s or "status of my account" in s:
        return "account_status"

    # Transactions
    if "last posted transaction" in s:
        return "last_posted_transaction"

    # Spend
    if "where did i spend most" in s or "top merchant" in s:
        return "top_merchants"
    if "how much did i spend" in s and "year" in s:
        return "spend_total_year"
    if "how much did i spend" in s and "month" in s:
        return "spend_total_month"

    return "generic"

def is_explanatory_intent(intent: str) -> bool:
    """Return True if the intent should produce multi-sentence explanations."""
    return intent in {
        "interest_reason",
    }

# ----------------------------
# Formatting helpers
# ----------------------------
def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def pretty_date(dt_iso: str) -> str:
    if not dt_iso:
        return ""
    try:
        dt = dateparser.parse(dt_iso)
        # e.g., "September 1, 2024" and include time if present (HH:MM)
        base = f"{dt.strftime('%B')} {ordinal(dt.day)}, {dt.year}"
        if dt.hour or dt.minute:
            base += dt.strftime(", %H:%M")
        return base
    except Exception:
        return dt_iso

def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return f"${x}"

# ----------------------------
# LLM extraction prompt
# ----------------------------
def build_extraction_prompt(intent: str, question: str, numbered_context: str) -> str:
    """
    Instruct the LLM to return a *minimal* JSON object with only the keys for this intent.
    All money must be numeric (no $); dates must be ISO strings.
    """
    fields = {
        # Facts (one-liners)
        "current_balance": ["current_balance"],
        "statement_balance": ["statement_balance", "statement_ym"],
        "last_payment_amount": ["payment_amount", "payment_date"],
        "last_payment_date": ["payment_date"],
        "last_posted_transaction": ["txn_date", "txn_amount", "merchant", "txn_type"],
        "account_status": ["account_status"],
        "interest_amount": ["interest_total", "interest_date"],
        "interest_amount_last": ["interest_total", "interest_date"],
        "interest_total_month": ["interest_total", "ym"],
        "interest_total_year": ["interest_total", "year"],
        "interest_total_statement": ["interest_total", "statement_id", "ym"],
        "interest_date_last": ["interest_date"],

        # Explanations (2–3 sentences)
        # reason_text: brief, human explanation (e.g., trailing interest because previous cycle wasn't paid in full)
        # driver_txns: optional list of merchant or txn ids/names that contributed
        "interest_reason": [
            "interest_total", "interest_date",
            "reason_text", "period_start", "period_end",
            "driver_txns"
        ],

        # Spend
        "top_merchants": ["merchant", "amount", "ym"],
        "spend_total_month": ["spend_total", "ym"],
        "spend_total_year": ["spend_total", "year"],

        "generic": ["answer_text"],
    }

    wanted = ", ".join(f'"{k}"' for k in fields.get(intent, ["answer_text"]))
    return f"""
You are a banking assistant. Read ONLY the provided context and return a MINIMAL JSON object
with exactly these keys (omit any you cannot determine): {wanted}.

Rules:
- If a value is unknown, omit the key.
- Money values must be numbers (no currency symbol).
- Dates must be ISO (YYYY-MM-DD or full ISO).
- Do not add any extra keys or explanations.

Context:
{numbered_context}

User question: {question}

Return JSON only:
""".strip()

# ----------------------------
# Final textual answer formatting
# ----------------------------
def format_answer(intent: str, data: Dict) -> str:
    # FACT one-liners
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
        parts: List[str] = []
        if amt is not None: parts.append(fmt_money(amt))
        if merch: parts.append(str(merch))
        if typ: parts.append(str(typ).lower())
        core = " • ".join(parts) if parts else "No details available"
        return f"Your last posted transaction was on {dt}: {core}." if dt else f"Your last posted transaction: {core}."

    if intent in {"interest_amount", "interest_amount_last"}:
        tot = data.get("interest_total")
        dt = pretty_date(data.get("interest_date", ""))
        if tot is not None and dt:
            return f"Your last interest charge was {fmt_money(tot)} on {dt}."
        if tot is not None:
            return f"Your last interest charge was {fmt_money(tot)}."

    if intent == "interest_total_month":
        tot = data.get("interest_total")
        ym = data.get("ym")
        if tot is not None and ym:
            return f"Your total interest for {ym} is {fmt_money(tot)}."
        if tot is not None:
            return f"Your total interest is {fmt_money(tot)}."

    if intent == "interest_total_year":
        tot = data.get("interest_total")
        yr = data.get("year")
        if tot is not None and yr:
            return f"Your total interest in {yr} is {fmt_money(tot)}."
        if tot is not None:
            return f"Your total interest is {fmt_money(tot)}."

    if intent == "interest_total_statement":
        tot = data.get("interest_total")
        ym = data.get("ym")
        if tot is not None and ym:
            return f"Your statement-cycle interest for {ym} is {fmt_money(tot)}."
        if tot is not None:
            return f"Your statement-cycle interest is {fmt_money(tot)}."

    if intent == "interest_date_last":
        dt = pretty_date(data.get("interest_date", ""))
        if dt:
            return f"Last interest was applied on {dt}."

    if intent == "account_status" and "account_status" in data:
        return f"Your account status is {data['account_status']}."

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

    if intent == "top_merchants":
        merch = data.get("merchant")
        amt = data.get("amount")
        ym = data.get("ym")
        if merch and amt is not None and ym:
            return f"You spent most at {merch} in {ym}, totaling {fmt_money(amt)}."
        if merch and amt is not None:
            return f"You spent most at {merch}, totaling {fmt_money(amt)}."

    # EXPLANATIONS (2–3 short sentences)
    if intent == "interest_reason":
        amt = data.get("interest_total")
        dt = pretty_date(data.get("interest_date", ""))
        reason = (data.get("reason_text") or "").strip()
        start = pretty_date(data.get("period_start", ""))
        end = pretty_date(data.get("period_end", ""))
        drivers = data.get("driver_txns") or []
        pieces: List[str] = []

        # Sentence 1: headline
        if amt is not None and dt:
            pieces.append(f"Interest of {fmt_money(amt)} was charged on {dt}.")
        elif amt is not None:
            pieces.append(f"Interest of {fmt_money(amt)} was charged.")
        elif dt:
            pieces.append(f"Interest was charged on {dt}.")
        else:
            pieces.append("Interest was charged.")

        # Sentence 2: reason
        if reason:
            pieces.append(reason)
        elif start or end:
            # generic trailing interest explanation with period window
            if start and end:
                pieces.append(f"This reflects trailing interest accruing between {start} and {end}.")
            else:
                pieces.append("This reflects trailing interest that accrued after the prior statement.")

        # Sentence 3: drivers (optional)
        if isinstance(drivers, list) and drivers:
            # Show up to two drivers to stay concise
            show = [str(x) for x in drivers[:2]]
            tail = " and others" if len(drivers) > 2 else ""
            pieces.append(f"Contributing transactions include {', '.join(show)}{tail}.")

        return " ".join(pieces).strip()

    # GENERIC fallback: first sentence only
    if "answer_text" in data and isinstance(data["answer_text"], str):
        s = re.split(r"[.!?]", data["answer_text"].strip())[0]
        return (s + ".") if s else "I couldn't find that."

    return "I couldn't find that."