## Calculation rules (updated with transaction interest lines)

1) **Monthly total interest**  
   Priority order:
   a. If `AGGREGATE ym=YYYY-MM interest_from_statements_total` exists → use it.  
   b. Else sum `STATEMENT.interestCharged` for that `ym`.  
   c. Else sum `TRANSACTION.amount` where:
      - `transactionType = INTEREST` OR
      - `displayTransactionType = interest_charged` OR
      - `merchantName` contains "Interest" (case-insensitive)
      AND `transactionStatus = POSTED`.  
   ⚠️ Do not double count statement interest and transactions — use one path only.

2) **Minimum payment for a month**  
   - Use `STATEMENT.minimumAmountDue` for that `ym`.

3) **Ending balance / current balance**  
   - Use `STATEMENT.endingBalance` where `ym` matches.  
   - If not found, fall back to `TRANSACTION.endingBalance` of the last transaction in that month.

4) **Payments**  
   - Use `PAYMENT.amount` grouped by `ym`.

5) **If data is missing**  
   - Explicitly list the missing `ym`/fields (e.g., “need STATEMENT.interestCharged or INTEREST transactions for 2024-08”).