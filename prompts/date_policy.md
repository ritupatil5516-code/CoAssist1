DATE SELECTION POLICY (FOR TRANSACTIONS)

- Use **transactionDateTime** as the canonical date.
- If missing, fallback to **postingDateTime**.
- Do NOT use **authDateTime** for time windows (“latest”, “this month”, “last month”, ranges) or spend aggregation.