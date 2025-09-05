## Data schema (field→meaning)
- STATEMENT: 
  - `ym` = YYYY-MM for the closing period
  - `interestCharged` = total interest billed for that statement cycle
  - `endingBalance`, `totalAmountDue`, `minimumAmountDue` = as printed on the statement
- TRANSACTION:
  - `interestFlag` = true when the transaction is an interest charge
  - `amount` = signed amount; interest charges are usually positive fees
  - `ym` = month of the posting/transaction date
- PAYMENT:
  - `amount` = amount of a payment (may be scheduled or posted)
  - `ym` = month of payment posting/schedule

## Calculation rules
1) **Monthly total interest** (preferred order)
   a. If an `AGGREGATE ym=YYYY-MM interest_from_statements_total` exists → use it.  
   b. Else sum `STATEMENT.interestCharged` for that `ym`.  
   c. Else sum `TRANSACTION.amount` where `interestFlag=true` for that `ym`.  
   - *Do not* add statement totals and interest transactions together (to avoid double counting).

2) **Minimum payment** for a cycle → use `STATEMENT.minimumAmountDue` for that `ym`.

3) **Balance questions**  
   - Current cycle: use the latest `STATEMENT.endingBalance`.
   - Historical by month: use the `STATEMENT.endingBalance` whose `ym` matches the question.

4) **Payments in a month** → sum `PAYMENT.amount` by the `ym`.

5) **If data is missing** → say exactly which `ym`/field you need and stop.

## Examples
Q: "What’s my total interest for 2025-08?"  
A: Use AGGREGATE (if present) → else sum STATEMENT.interestCharged where ym=2025-08. Show the sum and cite the lines.