"""
Generates synthetic failed-payment events with full counterfactual outcomes.

Surface 1 of 3: a payment was attempted and declined.

What each row contains
----------------------
* Decision-time features only: payment method, issuer, amount, decline
  code, prior attempts, timing. No instrument data — see src/schemas.py
  for why that is a hard rule rather than an omission.
* `true_root_cause`: the latent reason, used as the training label for the
  root-cause classifier.
* `po_*` columns: the potential outcome under each action variant, i.e.
  would this payment have recovered had we taken that action. Oracle
  knowledge, used only by src/benchmark.py to score policies.
* `logged_action` / `logged_recovered`: a randomised exploration log. This
  is the *only* outcome data the uplift model is allowed to train on.

Why the optimal action genuinely varies per event
-------------------------------------------------
This matters, because if one action were always best then ranking by
expected value would be pointless and a lookup table would do. The
response curves below encode real, learnable structure:

  * A transient bank-side failure recovers best on an *immediate* retry —
    waiting adds nothing.
  * Insufficient funds recovers best on a *delayed* retry, and much better
    if the delay crosses a payday. `day_of_month` carries that signal, so
    a 48-hour wait beats a 12-hour one late in the month and loses to it
    early in the month.
  * An expired card or bad details will never recover on any retry; the
    only thing that works is asking the customer to update the method —
    and that works far better on an engaged, long-tenured customer.

So the best action depends on cause *and* timing *and* relationship, which
is exactly the structure an expected-net-recovery ranker can exploit.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta

import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _simlib import OutcomeSampler, sample_logged_action, write_csv  # noqa: E402

SEED = 202
N_ROWS = 4000

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "failed_payments.csv")
CUSTOMERS_PATH = os.path.join(_HERE, "customers.csv")

# ---------------------------------------------------------------------
# Decline-code taxonomy. Codes overlap across causes on purpose: in real
# card networks "05" is a generic decline that several distinct causes all
# produce. A clean 1:1 code->cause map would let the classifier hit ~100%
# accuracy and the whole ML layer would be theatre.
# ---------------------------------------------------------------------
DECLINE_CODE_WEIGHTS = {
    "insufficient_funds":  {"51": 0.58, "61": 0.18, "65": 0.09, "05": 0.15},
    "expired_card":        {"54": 0.66, "33": 0.19, "05": 0.15},
    "technical_bank_side": {"91": 0.38, "96": 0.27, "05": 0.22, "51": 0.13},
    "fraud_suspected":     {"59": 0.47, "63": 0.28, "34": 0.10, "05": 0.15},
    "invalid_details":     {"14": 0.46, "82": 0.19, "12": 0.20, "05": 0.15},
}

CAUSES = list(DECLINE_CODE_WEIGHTS)
CAUSE_WEIGHTS = [0.35, 0.15, 0.25, 0.08, 0.17]

# Fraction of rows where the recorded cause is wrong. Ops teams mis-tag
# declines; without this the classifier looks implausibly good and its
# confidence scores stop meaning anything.
LABEL_NOISE_RATE = 0.07

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
METHOD_WEIGHTS = [0.46, 0.34, 0.12, 0.08]
BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "IDFC"]

# The action variants whose outcomes we simulate. `@n` is a parameter:
# hours of delay for retries. Note that `request_human_review` and
# `stop_and_flag_fraud` are absent — they are not interventions with a
# recovery probability to learn, and are handled separately (see the
# module docstring in src/economics.py).
ACTION_KEYS = [
    "do_nothing",
    "immediate_retry",
    "delayed_retry@12",
    "delayed_retry@48",
    "prompt_new_payment_method",
]

# Actions a randomised exploration period would plausibly have covered.
# Retries on *suspected fraud* are excluded: no real merchant randomises
# retries against transactions their fraud system flagged, so that data
# would never exist. This has a consequence worth being explicit about —
# the uplift model cannot learn "retrying fraud fails" because it never
# observes it. That is precisely why fraud handling is a hard guardrail
# and an explicit chargeback term in the economics, not a learned pattern.
LOGGABLE_ACTION_KEYS = ACTION_KEYS


def load_customer_ids() -> list[str]:
    if not os.path.exists(CUSTOMERS_PATH):
        raise SystemExit(
            "customers.csv not found — run `python data/generate_customers.py` first "
            "(or just run `python data/generate_all.py`)."
        )
    with open(CUSTOMERS_PATH, newline="", encoding="utf-8") as fh:
        return [row["customer_id"] for row in csv.DictReader(fh)]


def load_customer_index() -> dict[str, dict]:
    with open(CUSTOMERS_PATH, newline="", encoding="utf-8") as fh:
        return {row["customer_id"]: row for row in csv.DictReader(fh)}


def sample_decline_code(rng: random.Random, cause: str) -> str:
    weights = DECLINE_CODE_WEIGHTS[cause]
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def recovery_probabilities(
    cause: str,
    retry_count: int,
    day_of_month: int,
    engagement: float,
) -> dict[str, float]:
    """Marginal recovery probability for each action variant.

    `engagement` is a 0-1 summary of how responsive this customer is,
    derived from tenure and payment history. It matters most for actions
    that require the customer to *do* something (updating a payment
    method) and not at all for a silent gateway retry.
    """
    # --- Organic self-recovery: the customer retries unprompted. ---
    organic = {
        "insufficient_funds": 0.08,
        "expired_card": 0.04,
        "technical_bank_side": 0.14,   # people naturally retry a glitch
        "fraud_suspected": 0.01,
        "invalid_details": 0.06,
    }[cause]
    organic += 0.06 * engagement

    # --- Immediate retry: only genuinely useful for transient faults. ---
    immediate = {
        "insufficient_funds": 0.22,
        "expired_card": 0.02,
        "technical_bank_side": 0.72,
        "fraud_suspected": 0.015,
        "invalid_details": 0.05,
    }[cause]

    # --- Delayed retry: the payday effect lives here. ---
    delayed_base = {
        "insufficient_funds": 0.44,
        "expired_card": 0.02,
        "technical_bank_side": 0.56,   # blip may have cleared anyway
        "fraud_suspected": 0.012,
        "invalid_details": 0.05,
    }[cause]

    def payday_bonus(delay_hours: int) -> float:
        """Extra probability from a retry landing after salary credit.

        Only insufficient_funds benefits. A 48-hour wait crosses into the
        new month from the 29th onward; a 12-hour wait only from the 31st.
        """
        if cause != "insufficient_funds":
            return 0.0
        landing_day = day_of_month + (delay_hours / 24.0)
        crossed_month = landing_day > 30.5
        near_payday = 28 <= day_of_month <= 31
        if crossed_month:
            return 0.20
        if near_payday:
            return 0.08
        return 0.0

    delayed_12 = delayed_base + payday_bonus(12)
    delayed_48 = delayed_base + payday_bonus(48) - 0.04  # staleness cost of waiting longer

    # --- Prompt for a new payment method: the only fix for a dead card. ---
    prompt = {
        "insufficient_funds": 0.14,
        "expired_card": 0.46,
        "technical_bank_side": 0.20,
        "fraud_suspected": 0.02,
        "invalid_details": 0.40,
    }[cause]
    # Requires customer effort, so engagement matters a lot here.
    prompt *= 0.55 + 0.9 * engagement

    # --- Retry fatigue: each prior attempt lowers the odds and raises
    # issuer suspicion. Applies to gateway retries only. ---
    fatigue = 1.0 - 0.22 * retry_count

    return {
        "do_nothing": organic,
        "immediate_retry": immediate * fatigue,
        "delayed_retry@12": delayed_12 * fatigue,
        "delayed_retry@48": delayed_48 * fatigue,
        "prompt_new_payment_method": prompt,
    }


def engagement_score(cust: dict) -> float:
    """0-1 summary of customer responsiveness."""
    tenure = int(cust["tenure_months"])
    successes = int(cust["prior_successful_payments"])
    rp = float(cust["repeat_purchase_probability"])
    raw = 0.35 * min(1.0, tenure / 60.0) + 0.35 * min(1.0, successes / 40.0) + 0.30 * rp
    return max(0.0, min(1.0, raw))


def generate_row(rng: random.Random, sampler: OutcomeSampler, idx: int,
                 customer_ids: list[str], customers: dict[str, dict]) -> dict:
    customer_id = rng.choice(customer_ids)
    cust = customers[customer_id]

    true_cause = rng.choices(CAUSES, weights=CAUSE_WEIGHTS, k=1)[0]
    decline_code = sample_decline_code(rng, true_cause)

    # Recorded cause may be mis-tagged even though the underlying physics
    # (and therefore the outcomes) follow the true cause. This is the
    # honest version of label noise: it degrades the classifier without
    # secretly degrading the simulation.
    recorded_cause = true_cause
    if rng.random() < LABEL_NOISE_RATE:
        recorded_cause = rng.choice(CAUSES)

    payment_method = rng.choices(PAYMENT_METHODS, weights=METHOD_WEIGHTS, k=1)[0]
    bank = rng.choice(BANKS)
    amount = round(rng.lognormvariate(7.1, 1.05), 2)   # heavy right tail, like real ticket sizes
    amount = min(amount, 400_000.0)
    retry_count = rng.choices([0, 1, 2, 3], weights=[0.52, 0.29, 0.14, 0.05])[0]
    hour_of_day = rng.randint(0, 23)
    day_of_month = rng.randint(1, 31)

    failed_at = datetime(2026, 1, 1) + timedelta(
        days=rng.randint(0, 200), hours=hour_of_day, minutes=rng.randint(0, 59)
    )

    eng = engagement_score(cust)
    probs = recovery_probabilities(true_cause, retry_count, day_of_month, eng)
    outcomes = sampler.draw(probs)

    logged_action = sample_logged_action(rng, LOGGABLE_ACTION_KEYS)

    row = {
        "txn_id": f"txn_{idx:06d}",
        "customer_id": customer_id,
        "failed_at": failed_at.isoformat(),
        "payment_method": payment_method,
        "bank": bank,
        "amount": amount,
        "decline_code": decline_code,
        "retry_count": retry_count,
        "hour_of_day": hour_of_day,
        "day_of_month": day_of_month,
        # --- label for the root-cause classifier ---
        "true_root_cause": recorded_cause,
        # Whether this transaction is actually fraudulent. Used by the
        # benchmark to charge back any "recovered" fraud, which is what
        # makes a retry-everything policy genuinely expensive rather than
        # merely ineffective.
        "is_fraudulent": true_cause == "fraud_suspected",
        # --- randomised exploration log (the only outcome data models see) ---
        "logged_action": logged_action,
        "logged_recovered": outcomes[logged_action],
    }
    # --- oracle counterfactuals, for offline policy scoring only ---
    for key in ACTION_KEYS:
        row[f"po_{key}"] = outcomes[key]
    return row


def main() -> None:
    rng = random.Random(SEED)
    sampler = OutcomeSampler(rng)
    customer_ids = load_customer_ids()
    customers = load_customer_index()

    print("Generating failed-payment events...")
    rows = [generate_row(rng, sampler, i, customer_ids, customers) for i in range(N_ROWS)]
    write_csv(OUT_PATH, rows)

    total = sum(r["amount"] for r in rows)
    fraud = sum(r["is_fraudulent"] for r in rows)
    print(f"  {total:,.0f} INR at risk; {fraud} genuinely fraudulent "
          f"({fraud / len(rows):.1%}) — these must never be retried")


if __name__ == "__main__":
    main()
