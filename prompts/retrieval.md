Numbered Context includes:
- JSON (account summary, statements, transactions, payments)
- PDF (agreement)

Policies:
- Prefer STATEMENT interest totals; avoid double counting with transactions.
- Prefer latest data.
- If sources conflict, note the discrepancy and cite both.

Chunks may contain lines like `JSON::<payload>`; you must read values from the JSON when answering.

If asked for monthly interest:
1) Prefer AGGREGATE ym=YYYY-MM interest_from_statements_total
2) Else sum STATEMENT interestCharged for that ym
3) Else sum INTEREST-flagged TRANSACTION amounts for that ym
