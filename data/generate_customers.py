"""
Generates the customer dimension.

Every revenue-at-risk event joins to a customer here. This is the table
that lets the agent tell apart a five-year enterprise account from a
first-time low-margin buyer, which is improvement #8 in improvements.md:
interventions should reflect relationship value, not just ticket size.

Note what is deliberately *absent*: no email address, no phone number, no
name. The decision layer works from relationship attributes and consent
flags only. Resolving a customer to an actual contact address is the
messaging adapter's job, so the agent can decide "email this person"
without ever holding a directly contactable identifier.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _simlib import write_csv  # noqa: E402

SEED = 42
N_CUSTOMERS = 1500

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customers.csv")

SEGMENTS = ["consumer", "smb", "enterprise"]
SEGMENT_WEIGHTS = [0.70, 0.24, 0.06]

# Segment shapes the whole relationship: enterprise accounts are older,
# worth far more annually, carry thinner margins (negotiated pricing) and
# are much more likely to have given explicit contact consent.
SEGMENT_PROFILE = {
    "consumer":   {"tenure": (0, 48),   "annual": (500, 40_000),      "margin": (28, 55)},
    "smb":        {"tenure": (2, 84),   "annual": (20_000, 400_000),  "margin": (22, 42)},
    "enterprise": {"tenure": (6, 140),  "annual": (300_000, 6_000_000), "margin": (14, 30)},
}


def generate_customer(rng: random.Random, idx: int) -> dict:
    segment = rng.choices(SEGMENTS, weights=SEGMENT_WEIGHTS, k=1)[0]
    prof = SEGMENT_PROFILE[segment]

    tenure_months = rng.randint(*prof["tenure"])
    estimated_annual_value = round(rng.uniform(*prof["annual"]), 2)
    gross_margin_pct = round(rng.uniform(*prof["margin"]), 1)

    # Payment history scales with tenure, with noise. A long-tenured
    # customer with many successful payments is the strongest signal that
    # an intervention is worth spending on.
    prior_successful = max(0, int(rng.gauss(tenure_months * 0.8, 4)))
    prior_late = min(
        prior_successful,
        max(0, int(rng.gauss(tenure_months * 0.06, 1.2))),
    )

    # Repeat-purchase probability rises with tenure and falls with a poor
    # payment record.
    rp = 0.15 + 0.004 * tenure_months - 0.03 * prior_late + rng.gauss(0, 0.08)
    repeat_purchase_probability = round(max(0.02, min(0.95, rp)), 3)

    prior_coupon_redemptions = rng.choices([0, 1, 2, 3, 6], weights=[0.5, 0.22, 0.14, 0.09, 0.05])[0]

    # Contact history. Most customers have not been contacted recently;
    # a minority are already close to the frequency cap, which is what
    # makes the fatigue guardrail bite on real rows rather than never.
    contacts_last_7d = rng.choices([0, 1, 2, 3], weights=[0.62, 0.24, 0.10, 0.04])[0]
    hours_since_last_contact = (
        round(rng.uniform(1, 168), 1) if contacts_last_7d else round(rng.uniform(200, 3000), 1)
    )

    # Consent. Enterprise/SMB relationships are contractual so consent is
    # near-universal; consumer opt-in is much patchier, especially for
    # WhatsApp and SMS.
    if segment == "consumer":
        email_consent = rng.random() < 0.88
        whatsapp_consent = rng.random() < 0.41
        sms_consent = rng.random() < 0.33
    else:
        email_consent = rng.random() < 0.98
        whatsapp_consent = rng.random() < 0.62
        sms_consent = rng.random() < 0.48

    dnd_flagged = rng.random() < 0.05

    return {
        "customer_id": f"cust_{idx:06d}",
        "segment": segment,
        "tenure_months": tenure_months,
        "prior_successful_payments": prior_successful,
        "prior_late_payments": prior_late,
        "estimated_annual_value_inr": estimated_annual_value,
        "gross_margin_pct": gross_margin_pct,
        "repeat_purchase_probability": repeat_purchase_probability,
        "prior_coupon_redemptions": prior_coupon_redemptions,
        "contacts_last_7d": contacts_last_7d,
        "hours_since_last_contact": hours_since_last_contact,
        "email_consent": email_consent,
        "whatsapp_consent": whatsapp_consent,
        "sms_consent": sms_consent,
        "dnd_flagged": dnd_flagged,
    }


def main() -> None:
    rng = random.Random(SEED)
    rows = [generate_customer(rng, i) for i in range(N_CUSTOMERS)]
    print("Generating customer dimension...")
    write_csv(OUT_PATH, rows)

    n_dnd = sum(r["dnd_flagged"] for r in rows)
    n_ent = sum(r["segment"] == "enterprise" for r in rows)
    print(f"  {n_ent} enterprise, {n_dnd} DND-flagged (these can never be contacted)")


if __name__ == "__main__":
    main()
