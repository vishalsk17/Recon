"""
Generates synthetic overdue-receivable events with counterfactual outcomes.

Surface 3 of 3: a B2B invoice is past its due date. Different again from
the other two — the customer is a business, the amounts are an order of
magnitude larger, and the wrong intervention damages a commercial
relationship rather than just wasting a rupee of gateway fee.

The interesting economics here
------------------------------
Collections escalation is expensive (agency handling, plus real
relationship damage if it turns out the invoice was simply wrong). The
response curves make it a genuinely close call rather than an obvious one:

  * A **cash-flow-constrained** payer responds best to a payment plan, not
    to pressure — a plan converts materially better than a bare reminder.
  * A **chronic late payer** pays reliably in the end; a reminder is
    usually enough and escalation mostly burns goodwill.
  * A **disputed** invoice will not be paid until the dispute is resolved,
    and chasing it makes things worse. So will an **erroneous** invoice —
    the fix is a corrected invoice from finance, not a chase.

Because escalation costs hundreds of rupees to attempt and carries a large
penalty when unwarranted, expected net recovery only justifies it on large,
long-overdue, undisputed invoices. That threshold falls out of the
economics rather than being hardcoded — and it is then *also* fenced by a
hard guardrail requiring human sign-off, because a relationship-damaging
action should never rest on a model's arithmetic alone.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _simlib import OutcomeSampler, sample_logged_action, write_csv  # noqa: E402

SEED = 2121
N_ROWS = 1200

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "overdue_receivables.csv")
CUSTOMERS_PATH = os.path.join(_HERE, "customers.csv")

CAUSES = ["cash_flow_issue", "dispute_pending", "invoice_error", "chronic_late_payer"]
CAUSE_WEIGHTS = [0.38, 0.16, 0.11, 0.35]

ACTION_KEYS = [
    "do_nothing",
    "automated_reminder",
    "automated_reminder_with_payment_plan_offer",
    "escalate_to_collections",
]


def load_customers() -> tuple[list[str], dict[str, dict]]:
    if not os.path.exists(CUSTOMERS_PATH):
        raise SystemExit("customers.csv not found — run `python data/generate_all.py`.")
    with open(CUSTOMERS_PATH, newline="", encoding="utf-8") as fh:
        idx = {row["customer_id"]: row for row in csv.DictReader(fh)}
    # Receivables are a B2B surface, so draw only from business accounts.
    business = [cid for cid, r in idx.items() if r["segment"] in ("smb", "enterprise")]
    return business, idx


def recovery_probabilities(
    cause: str,
    days_overdue: int,
    prior_late_payments: int,
    invoice_amount: float,
) -> dict[str, float]:
    """Marginal probability the invoice gets paid within the horizon."""

    # --- No action: many invoices are eventually paid regardless. This is
    # the single most important baseline on this surface — it is why
    # "chase everything" wastes so much. ---
    organic = {
        "cash_flow_issue": 0.30,
        "dispute_pending": 0.05,
        "invoice_error": 0.03,
        "chronic_late_payer": 0.37,   # always late, always pays
    }[cause]

    reminder = {
        "cash_flow_issue": 0.55,
        "dispute_pending": 0.08,
        "invoice_error": 0.06,
        "chronic_late_payer": 0.63,
    }[cause]

    # A payment plan is what actually unblocks a cash-flow problem.
    plan = {
        "cash_flow_issue": 0.69,
        "dispute_pending": 0.11,
        "invoice_error": 0.07,
        "chronic_late_payer": 0.66,   # barely better than a reminder
    }[cause]

    # Escalation extracts payment from some hard cases but is blunt.
    collections = {
        "cash_flow_issue": 0.47,
        "dispute_pending": 0.22,
        "invoice_error": 0.12,
        "chronic_late_payer": 0.58,
    }[cause]

    # --- Ageing. A reminder goes stale fast: after four months, an email
    # into the same AP inbox that has already ignored three is worth very
    # little. A collections agency does not decay the same way — that is
    # the entire reason the channel exists and costs what it costs.
    #
    # The consequence is a genuine crossover: on a fresh invoice a reminder
    # beats escalation comfortably, and somewhere past the 60-90 day mark
    # escalation overtakes it. The agent has to find that crossover *and*
    # weigh it against escalation's cost, which is only worth paying on a
    # large invoice. Neither a fixed day-threshold rule nor a pure
    # recovery-probability model gets this right on its own.
    def decay(p: float, slow: bool = False) -> float:
        per_day = 0.0008 if slow else 0.0022
        return p - per_day * days_overdue - 0.012 * prior_late_payments

    # Very large invoices need sign-off cycles on the customer's side, so
    # they move more slowly whatever we do.
    size_drag = 0.05 if invoice_amount > 500_000 else 0.0

    return {
        "do_nothing": decay(organic) - size_drag,
        "automated_reminder": decay(reminder) - size_drag,
        "automated_reminder_with_payment_plan_offer": decay(plan) - size_drag,
        "escalate_to_collections": decay(collections, slow=True) - size_drag,
    }


def document_state(rng: random.Random, cause: str) -> dict:
    """Invoice document state as an AR/ERP system actually records it.

    Added because without it `invoice_error` is not identifiable: amount,
    ageing and late history cannot distinguish "this invoice is wrong" from
    "this customer is slow", and a classifier trained without these fields
    never predicts the class at all (0.00 recall on a 33-row test slice).

    That is not an acceptable place to leave it, because `invoice_error` is
    one of the two causes on the `never_auto_chase` list. A cause the model
    can never detect is a guardrail that never fires — and chasing a
    customer over an invoice that is genuinely wrong is precisely the
    relationship damage this system exists to avoid.

    All four fields are standard AR data. Purchase-order mismatch in
    particular is the single most common reason a B2B invoice sits unpaid
    in accounts payable: no PO match, no payment run, and often no
    notification to the supplier either.

    Naming note: these are deliberately `purchase_order_*` and not
    `po_number_*`. The `po_` prefix is reserved throughout this project for
    potential-outcome oracle columns, which src/dataio.py strips before the
    agent ever sees a row — a field named `po_number_required` would be
    silently deleted as an outcome and, worse, would trip the leakage
    assertion in src/ml/features.py.
    """
    # Larger, more process-heavy buyers raise POs; small ones often do not.
    purchase_order_required = rng.random() < 0.62

    if cause == "invoice_error":
        # Wrong PO reference, a reissued document, or a figure that does not
        # match what was agreed. AP parks it and frequently says nothing.
        purchase_order_on_invoice = not purchase_order_required or rng.random() < 0.45
        invoice_revision_count = rng.choices([0, 1, 2, 3], weights=[0.34, 0.34, 0.21, 0.11])[0]
        amount_variance_vs_contract_pct = round(abs(rng.gauss(9.0, 7.0)), 2)
        acknowledged_in_portal = rng.random() < 0.22
    elif cause == "dispute_pending":
        # They read it and objected, so it is acknowledged; the disagreement
        # is often about the figure.
        purchase_order_on_invoice = not purchase_order_required or rng.random() < 0.88
        invoice_revision_count = rng.choices([0, 1], weights=[0.78, 0.22])[0]
        amount_variance_vs_contract_pct = round(abs(rng.gauss(5.0, 5.0)), 2)
        acknowledged_in_portal = rng.random() < 0.86
    else:
        # cash_flow_issue / chronic_late_payer: the document is fine. They
        # have seen it. They simply have not paid it.
        purchase_order_on_invoice = not purchase_order_required or rng.random() < 0.96
        invoice_revision_count = 0 if rng.random() < 0.93 else 1
        amount_variance_vs_contract_pct = round(abs(rng.gauss(0.4, 0.9)), 2)
        acknowledged_in_portal = rng.random() < (0.80 if cause == "cash_flow_issue" else 0.66)

    return {
        "purchase_order_required": purchase_order_required,
        "purchase_order_on_invoice": purchase_order_on_invoice,
        "invoice_revision_count": invoice_revision_count,
        "amount_variance_vs_contract_pct": amount_variance_vs_contract_pct,
        "acknowledged_in_portal": acknowledged_in_portal,
    }


def payment_history_norm(rng: random.Random, cause: str, prior_late: int) -> float:
    """Average days late across this account's last six settled invoices.

    This is the field that separates the two causes that otherwise blur
    together. A chronic late payer looks exactly as they always look — 90
    days overdue against a history of paying 20 days late is business as
    usual. A cash-flow-constrained account has a *clean* history and has
    suddenly stopped, and that deviation from its own norm is the signal.

    It is deliberately not a clean separator: the distributions overlap, so
    the model has to combine it with ageing rather than read the answer off
    one column.
    """
    if cause == "chronic_late_payer":
        base = rng.gauss(17.0, 7.0)
    elif cause == "cash_flow_issue":
        base = rng.gauss(5.0, 4.0)
    else:
        base = rng.gauss(4.0, 3.5)
    # A long recorded late history drags the average up whatever the cause.
    base += 0.9 * prior_late
    return round(max(0.0, base), 1)


def generate_row(rng: random.Random, sampler: OutcomeSampler, idx: int,
                 customer_ids: list[str], customers: dict[str, dict]) -> dict:
    customer_id = rng.choice(customer_ids)
    cust = customers[customer_id]

    cause = rng.choices(CAUSES, weights=CAUSE_WEIGHTS, k=1)[0]

    # Invoice size tracks the account's annual value, so enterprise
    # accounts carry the large invoices — which is what makes the
    # collections trade-off segment-dependent.
    annual = float(cust["estimated_annual_value_inr"])
    invoice_amount = round(max(3_000.0, min(2_000_000.0, annual * rng.uniform(0.04, 0.30))), 2)

    days_overdue = rng.choices(
        [1, 5, 15, 30, 45, 60, 90, 120],
        weights=[0.14, 0.16, 0.18, 0.16, 0.10, 0.10, 0.09, 0.07],
    )[0]
    prior_late = int(cust["prior_late_payments"])
    if cause == "chronic_late_payer":
        # Only *most* chronic late payers show a long late history in AR —
        # some are newly-turned-bad accounts whose record has not caught up.
        # Left at 100% this would make the label definitionally recoverable
        # from `prior_late_payments`, the classifier would score ~1.0, and
        # the confidence gate would never do any work. The residual
        # ambiguity is the point.
        if rng.random() < 0.72:
            prior_late = max(prior_late, rng.randint(3, 8))
    elif rng.random() < 0.12:
        # And some accounts have a messy history for reasons unrelated to
        # this invoice, so a long late record is suggestive, not decisive.
        prior_late = max(prior_late, rng.randint(3, 6))

    # Disputes are flagged in the AR system. This is decision-time
    # information, not a hidden label — and the agent is required to
    # respect it (see the never_auto_chase guardrail).
    dispute_flagged_in_ar = cause == "dispute_pending" and rng.random() < 0.80

    due_date = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 200))

    doc = document_state(rng, cause)
    avg_days_late = payment_history_norm(rng, cause, prior_late)

    probs = recovery_probabilities(cause, days_overdue, prior_late, invoice_amount)
    outcomes = sampler.draw(probs)
    logged_action = sample_logged_action(rng, ACTION_KEYS)

    row = {
        "invoice_id": f"inv_{idx:06d}",
        "customer_id": customer_id,
        "due_date": due_date.isoformat(),
        "invoice_amount": invoice_amount,
        "days_overdue": days_overdue,
        "prior_late_payments": prior_late,
        "dispute_flagged_in_ar": dispute_flagged_in_ar,
        "avg_days_late_last_6_invoices": avg_days_late,
        **doc,
        "hour_of_day": rng.randint(9, 18),   # B2B invoices are raised in business hours
        "true_root_cause": cause,
        "logged_action": logged_action,
        "logged_recovered": outcomes[logged_action],
    }
    for key in ACTION_KEYS:
        row[f"po_{key}"] = outcomes[key]
    return row


def main() -> None:
    rng = random.Random(SEED)
    sampler = OutcomeSampler(rng)
    customer_ids, customers = load_customers()

    print("Generating overdue-receivable events...")
    rows = [generate_row(rng, sampler, i, customer_ids, customers) for i in range(N_ROWS)]
    write_csv(OUT_PATH, rows)

    total = sum(r["invoice_amount"] for r in rows)
    self_healing = sum(r["po_do_nothing"] for r in rows)
    print(f"  {total:,.0f} INR at risk; {self_healing} invoices "
          f"({self_healing / len(rows):.0%}) get paid with no intervention at all")


if __name__ == "__main__":
    main()
