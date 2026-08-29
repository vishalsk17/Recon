"""
Generates synthetic checkout-abandonment events with counterfactual outcomes.

Surface 2 of 3: a cart was built but no payment was ever attempted. A
distinct problem from a failed payment — nothing went wrong technically,
the customer simply did not complete.

The interesting economics here
------------------------------
This surface is where a naive "always discount to recover the cart" policy
loses the most money, and the response curves below are tuned so that shows
up honestly rather than by construction:

  * A cart abandoned because of **technical friction** recovers at roughly
    the same rate from a plain reminder as from a 10%-off code — the
    customer already wanted to buy. Discounting it hands away margin for
    nothing. Spotting this is worth real money and requires distinguishing
    cause, not just predicting recovery.
  * A **price-sensitive** cart genuinely responds to a discount, and
    responds *more* if the customer has redeemed coupons before. But the
    step from 5% to 10% buys far less conversion than the first 5% did, so
    the higher tier is only justified on some carts.
  * **High-value hesitation** carts barely respond to automation at all.
    The honest answer is often a human follow-up or nothing.

So the right action varies by cause, by coupon history and by cart size,
and the margin cost of a discount has to be weighed against the conversion
it actually buys. That is the case for ranking by expected net recovery
instead of mapping cause to a fixed action.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _simlib import OutcomeSampler, sample_logged_action, write_csv  # noqa: E402

SEED = 707
N_ROWS = 2500

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "checkout_abandonment.csv")
CUSTOMERS_PATH = os.path.join(_HERE, "customers.csv")

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.63, 0.30, 0.07]

CAUSES = ["price_sensitivity", "high_cart_hesitation", "technical_friction", "comparison_shopping"]
CAUSE_WEIGHTS = [0.34, 0.24, 0.17, 0.25]

ACTION_KEYS = [
    "do_nothing",
    "send_reminder_email",
    "send_reminder_whatsapp",
    "offer_bounded_discount@5",
    "offer_bounded_discount@10",
]


def load_customers() -> tuple[list[str], dict[str, dict]]:
    if not os.path.exists(CUSTOMERS_PATH):
        raise SystemExit("customers.csv not found — run `python data/generate_all.py`.")
    with open(CUSTOMERS_PATH, newline="", encoding="utf-8") as fh:
        idx = {row["customer_id"]: row for row in csv.DictReader(fh)}
    return list(idx), idx


def recovery_probabilities(
    cause: str,
    cart_value: float,
    minutes_since_abandon: int,
    is_returning: bool,
    coupon_affinity: float,
    device: str,
) -> dict[str, float]:
    """Marginal recovery probability per action variant.

    `coupon_affinity` is 0-1, from the customer's prior redemptions. It
    only modulates the discount actions — a reminder does not care whether
    you like coupons.
    """
    # --- Organic return without any nudge. ---
    organic = {
        "price_sensitivity": 0.06,
        "high_cart_hesitation": 0.07,
        "technical_friction": 0.16,   # they meant to buy; many come back
        "comparison_shopping": 0.09,
    }[cause]
    if is_returning:
        organic += 0.06

    # --- Plain reminder email. ---
    email = {
        "price_sensitivity": 0.17,
        "high_cart_hesitation": 0.13,
        "technical_friction": 0.52,   # a link back fixes it
        "comparison_shopping": 0.22,
    }[cause]

    # --- Discount at two tiers. Diminishing returns are explicit: the
    # 10% tier adds only about half of what the 5% tier added. ---
    disc5 = {
        "price_sensitivity": 0.36,
        "high_cart_hesitation": 0.19,
        "technical_friction": 0.54,   # barely better than a free reminder
        "comparison_shopping": 0.30,
    }[cause]
    increment = {
        "price_sensitivity": 0.09,
        "high_cart_hesitation": 0.05,
        "technical_friction": 0.02,
        "comparison_shopping": 0.06,
    }[cause]
    disc10 = disc5 + increment

    # Coupon-trained customers respond harder to discounts.
    coupon_mult = 0.85 + 0.45 * coupon_affinity
    disc5 *= coupon_mult
    disc10 *= coupon_mult

    # --- Shared modifiers. ---
    def modify(p: float, is_outreach: bool) -> float:
        if is_returning:
            p += 0.10
        # Recency: catching someone within the hour works much better.
        if minutes_since_abandon <= 60:
            p += 0.09
        elif minutes_since_abandon >= 720:
            p -= 0.06
        # Larger baskets are harder to close regardless of channel.
        if cart_value > 15_000:
            p -= 0.09
        elif cart_value > 8_000:
            p -= 0.04
        # Mobile technical friction is more likely to be a genuine
        # checkout bug, so a link back helps more.
        if is_outreach and device == "mobile" and cause == "technical_friction":
            p += 0.05
        return p

    email = modify(email, True)
    disc5 = modify(disc5, True)
    disc10 = modify(disc10, True)

    # --- WhatsApp: higher open rate than email, so a genuine lift, but it
    # needs consent and carries a bigger goodwill cost when unwanted. ---
    whatsapp = email * 1.22

    return {
        "do_nothing": organic,
        "send_reminder_email": email,
        "send_reminder_whatsapp": whatsapp,
        "offer_bounded_discount@5": disc5,
        "offer_bounded_discount@10": disc10,
    }


def session_telemetry(rng: random.Random, cause: str, cart_value: float) -> dict:
    """Front-end session signals a real checkout genuinely logs.

    Added because without them `technical_friction` is not identifiable at
    all: cart value, item count and device simply do not distinguish "the
    payment widget threw an error" from "I decided it was too expensive",
    and a classifier trained without them recovers that class at about 5%.
    That is not a modelling failure, it is missing instrumentation — and it
    matters, because friction carts are the ones a free reminder recovers
    and a discount wastes margin on.

    Any real merchant has all four of these in their analytics already:
    session length, whether the payment page was reached, client-side
    errors, and whether a payment was actually initiated.
    """
    if cause == "technical_friction":
        # Got to the payment step, tried, and hit errors.
        reached_payment_page = rng.random() < 0.92
        checkout_errors_logged = rng.choices([0, 1, 2, 3], weights=[0.08, 0.36, 0.34, 0.22])[0]
        payment_attempts_started = rng.choices([0, 1, 2], weights=[0.12, 0.52, 0.36])[0]
        session_seconds = int(rng.gauss(420, 160))
    elif cause == "price_sensitivity":
        # Often bails at the point totals appear (shipping, tax).
        reached_payment_page = rng.random() < 0.38
        checkout_errors_logged = rng.choices([0, 1], weights=[0.94, 0.06])[0]
        payment_attempts_started = 0
        session_seconds = int(rng.gauss(210, 90))
    elif cause == "high_cart_hesitation":
        # Long deliberation on an expensive basket, rarely commits.
        reached_payment_page = rng.random() < 0.45
        checkout_errors_logged = rng.choices([0, 1], weights=[0.93, 0.07])[0]
        payment_attempts_started = 1 if rng.random() < 0.14 else 0
        session_seconds = int(rng.gauss(680, 240))
    else:  # comparison_shopping
        # Browses a lot, leaves to check a competitor, never starts paying.
        reached_payment_page = rng.random() < 0.22
        checkout_errors_logged = rng.choices([0, 1], weights=[0.95, 0.05])[0]
        payment_attempts_started = 0
        session_seconds = int(rng.gauss(500, 220))

    # A little cross-class overlap so the signal is strong but not decisive.
    if rng.random() < 0.07:
        checkout_errors_logged = max(checkout_errors_logged, 1)

    return {
        "session_seconds": max(20, session_seconds),
        "reached_payment_page": reached_payment_page,
        "checkout_errors_logged": checkout_errors_logged,
        "payment_attempts_started": payment_attempts_started,
    }


def generate_row(rng: random.Random, sampler: OutcomeSampler, idx: int,
                 customer_ids: list[str], customers: dict[str, dict]) -> dict:
    customer_id = rng.choice(customer_ids)
    cust = customers[customer_id]

    cause = rng.choices(CAUSES, weights=CAUSE_WEIGHTS, k=1)[0]

    # Cart value correlates with cause: hesitation carts are big by
    # definition, price-sensitivity carts skew small.
    if cause == "high_cart_hesitation":
        cart_value = round(rng.lognormvariate(9.4, 0.6), 2)
    elif cause == "price_sensitivity":
        cart_value = round(rng.lognormvariate(7.3, 0.8), 2)
    else:
        cart_value = round(rng.lognormvariate(8.2, 0.9), 2)
    cart_value = float(min(max(cart_value, 150.0), 250_000.0))

    items_count = max(1, int(rng.gauss(3.2, 1.8)))
    if cause == "high_cart_hesitation":
        items_count = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]

    device = rng.choices(DEVICES, weights=DEVICE_WEIGHTS, k=1)[0]
    is_returning = int(cust["prior_successful_payments"]) > 0 and rng.random() < 0.75
    minutes_since_abandon = rng.choices(
        [10, 30, 60, 180, 720, 1440], weights=[0.16, 0.18, 0.18, 0.2, 0.16, 0.12]
    )[0]
    hour_of_day = rng.randint(0, 23)
    abandoned_at = datetime(2026, 1, 1) + timedelta(
        days=rng.randint(0, 200), hours=hour_of_day, minutes=rng.randint(0, 59)
    )

    coupon_affinity = min(1.0, int(cust["prior_coupon_redemptions"]) / 4.0)

    telemetry = session_telemetry(rng, cause, cart_value)

    probs = recovery_probabilities(
        cause, cart_value, minutes_since_abandon, is_returning, coupon_affinity, device
    )
    outcomes = sampler.draw(probs)
    logged_action = sample_logged_action(rng, ACTION_KEYS)

    row = {
        "event_id": f"cart_{idx:06d}",
        "customer_id": customer_id,
        "abandoned_at": abandoned_at.isoformat(),
        "cart_value": cart_value,
        "items_count": items_count,
        "device": device,
        "is_returning_customer": is_returning,
        "minutes_since_abandon": minutes_since_abandon,
        "hour_of_day": hour_of_day,
        **telemetry,
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

    print("Generating checkout-abandonment events...")
    rows = [generate_row(rng, sampler, i, customer_ids, customers) for i in range(N_ROWS)]
    write_csv(OUT_PATH, rows)

    total = sum(r["cart_value"] for r in rows)
    # How often a discount buys nothing a free reminder would not have won
    # anyway. This is the waste the agent is meant to avoid.
    pointless = sum(
        1 for r in rows if r["po_send_reminder_email"] and r["po_offer_bounded_discount@10"]
    )
    print(f"  {total:,.0f} INR at risk; on {pointless} carts "
          f"({pointless / len(rows):.0%}) a free reminder recovers it just as well as 10% off")


if __name__ == "__main__":
    main()
