"""
Core data shapes passed between the agent's tools.

Two deliberate design choices worth flagging:

1. **No cardholder data anywhere.** There is no field for a PAN, CVV,
   expiry, cardholder name, or full bank account number — not even an
   optional one. The agent reasons about a payment from its *metadata*
   (method, issuer name, decline code, timing, retry history) and never
   needs the instrument itself. A test in tests/test_defensive_posture.py
   asserts this stays true, because "we'll just add a card_number field
   for convenience" is how PCI scope creeps into a codebase.

2. **Ground truth is quarantined.** Simulated outcome labels live in a
   separate `SimulatedOutcomes` object that the decision path structurally
   cannot read. The agent receives a `RiskEvent`; the evaluation harness
   receives the outcomes separately and joins them by id afterwards. This
   makes outcome leakage a type error rather than a code-review question.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------
# Event surfaces
# ---------------------------------------------------------------------

PAYMENT_FAILURE = "payment_failure"
CHECKOUT_ABANDONMENT = "checkout_abandonment"
OVERDUE_RECEIVABLE = "overdue_receivable"

EVENT_TYPES = (PAYMENT_FAILURE, CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE)

# ---------------------------------------------------------------------
# The action space. Every action the agent can ever take is named here.
# A closed action vocabulary is the whole point: it is not possible for
# the system to invent a new kind of intervention at runtime, which is
# what makes "bounded" a structural property rather than a promise.
# ---------------------------------------------------------------------

DO_NOTHING = "do_nothing"
IMMEDIATE_RETRY = "immediate_retry"
DELAYED_RETRY = "delayed_retry"
PROMPT_NEW_PAYMENT_METHOD = "prompt_new_payment_method"
SEND_REMINDER_EMAIL = "send_reminder_email"
SEND_REMINDER_WHATSAPP = "send_reminder_whatsapp"
OFFER_BOUNDED_DISCOUNT = "offer_bounded_discount"
AUTOMATED_REMINDER = "automated_reminder"
AUTOMATED_REMINDER_WITH_PLAN = "automated_reminder_with_payment_plan_offer"
ESCALATE_TO_COLLECTIONS = "escalate_to_collections"
STOP_AND_FLAG_FRAUD = "stop_and_flag_fraud"
REQUEST_HUMAN_REVIEW = "request_human_review"

ALL_ACTIONS = (
    DO_NOTHING,
    IMMEDIATE_RETRY,
    DELAYED_RETRY,
    PROMPT_NEW_PAYMENT_METHOD,
    SEND_REMINDER_EMAIL,
    SEND_REMINDER_WHATSAPP,
    OFFER_BOUNDED_DISCOUNT,
    AUTOMATED_REMINDER,
    AUTOMATED_REMINDER_WITH_PLAN,
    ESCALATE_TO_COLLECTIONS,
    STOP_AND_FLAG_FRAUD,
    REQUEST_HUMAN_REVIEW,
)

# Which actions are available on which surface. Used by the candidate
# generator, and asserted in tests so a receivables invoice can never be
# handed a "retry the card" action.
ACTIONS_BY_SURFACE: dict[str, tuple[str, ...]] = {
    PAYMENT_FAILURE: (
        DO_NOTHING,
        IMMEDIATE_RETRY,
        DELAYED_RETRY,
        PROMPT_NEW_PAYMENT_METHOD,
        STOP_AND_FLAG_FRAUD,
        REQUEST_HUMAN_REVIEW,
    ),
    CHECKOUT_ABANDONMENT: (
        DO_NOTHING,
        SEND_REMINDER_EMAIL,
        SEND_REMINDER_WHATSAPP,
        OFFER_BOUNDED_DISCOUNT,
        REQUEST_HUMAN_REVIEW,
    ),
    OVERDUE_RECEIVABLE: (
        DO_NOTHING,
        AUTOMATED_REMINDER,
        AUTOMATED_REMINDER_WITH_PLAN,
        ESCALATE_TO_COLLECTIONS,
        REQUEST_HUMAN_REVIEW,
    ),
}

# Actions that contact the customer directly. Consent, DND, quiet hours
# and frequency caps apply to exactly this set.
OUTREACH_ACTIONS = frozenset({
    PROMPT_NEW_PAYMENT_METHOD,
    SEND_REMINDER_EMAIL,
    SEND_REMINDER_WHATSAPP,
    OFFER_BOUNDED_DISCOUNT,
    AUTOMATED_REMINDER,
    AUTOMATED_REMINDER_WITH_PLAN,
})

# Actions that attempt to move money through the gateway. Retry
# discipline applies to exactly this set.
RETRY_ACTIONS = frozenset({IMMEDIATE_RETRY, DELAYED_RETRY})

# Actions that always require a human to sign off before anything real
# happens, regardless of expected value.
HUMAN_GATED_ACTIONS = frozenset({ESCALATE_TO_COLLECTIONS, REQUEST_HUMAN_REVIEW})

# Channel each outreach action uses, for consent checking.
ACTION_CHANNEL: dict[str, str] = {
    PROMPT_NEW_PAYMENT_METHOD: "email",
    SEND_REMINDER_EMAIL: "email",
    SEND_REMINDER_WHATSAPP: "whatsapp",
    OFFER_BOUNDED_DISCOUNT: "email",
    AUTOMATED_REMINDER: "email",
    AUTOMATED_REMINDER_WITH_PLAN: "email",
}


# ---------------------------------------------------------------------
# Customer dimension
# ---------------------------------------------------------------------

# The closed vocabulary for `CustomerProfile.segment`.
#
# Not enforced in `__post_init__`, deliberately. A segment arriving from an
# upstream system with a value nobody has seen before is a data question, and
# raising on it would take the whole sweep down over a label that affects
# nothing the agent decides. What it must not do is flow into a prompt, so
# `narrator.build_fact_sheet` renders through this set and falls back to
# "unspecified" — the one place an unrecognised string would otherwise become
# instruction-shaped text a model reads.
SEGMENTS: frozenset[str] = frozenset({"enterprise", "smb", "consumer"})


@dataclass(frozen=True)
class CustomerProfile:
    """Relationship context for the customer behind an event.

    This is what lets the agent distinguish "chase a one-off low-margin
    buyer" from "handle a five-year account carefully". Note there is no
    contact detail here (no email address, no phone number) — the agent
    decides *whether and how* to contact someone; resolving that to an
    actual address is the messaging adapter's job, and keeping the two
    apart means the decision layer holds no directly contactable PII.
    """
    customer_id: str
    tenure_months: int
    prior_successful_payments: int
    prior_late_payments: int
    estimated_annual_value_inr: float
    gross_margin_pct: float
    repeat_purchase_probability: float
    prior_coupon_redemptions: int
    contacts_last_7d: int
    hours_since_last_contact: float
    email_consent: bool
    whatsapp_consent: bool
    sms_consent: bool
    dnd_flagged: bool
    segment: str  # one of SEGMENTS; see the note there on why it is not validated here

    def has_consent(self, channel: str) -> bool:
        if self.dnd_flagged:
            return False
        return {
            "email": self.email_consent,
            "whatsapp": self.whatsapp_consent,
            "sms": self.sms_consent,
        }.get(channel, False)


def default_customer(customer_id: str = "unknown", margin_pct: float = 35.0) -> CustomerProfile:
    """Conservative stand-in when no customer record is available.

    Deliberately pessimistic: no consent for any channel, DND assumed
    off but nothing opted in, zero relationship value. An unknown
    customer therefore cannot be messaged, which is the safe default —
    a missing record must never widen what the agent is allowed to do.
    """
    return CustomerProfile(
        customer_id=customer_id,
        tenure_months=0,
        prior_successful_payments=0,
        prior_late_payments=0,
        estimated_annual_value_inr=0.0,
        gross_margin_pct=margin_pct,
        repeat_purchase_probability=0.0,
        prior_coupon_redemptions=0,
        contacts_last_7d=0,
        hours_since_last_contact=9999.0,
        email_consent=False,
        whatsapp_consent=False,
        sms_consent=False,
        dnd_flagged=False,
        segment="consumer",
    )


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RiskEvent:
    """One unit of revenue at risk.

    `features` holds only surface-specific signals available at decision
    time. `occurred_at_hour` is carried separately because quiet-hours
    logic needs it on every surface.
    """
    event_id: str
    event_type: str
    amount_inr: float
    customer: CustomerProfile
    features: dict[str, Any]
    occurred_at: str = ""          # ISO8601
    occurred_at_hour: int = 12     # 0-23 local (IST)

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {self.event_type!r}")
        if self.amount_inr < 0:
            raise ValueError("amount_inr cannot be negative")


@dataclass(frozen=True)
class SimulatedOutcomes:
    """Counterfactual outcomes for one event, from the data simulator.

    Maps action name -> whether the event would have been recovered had
    that action been taken. This is oracle knowledge that exists only
    because the data is synthetic; it is used **exclusively** by
    src/benchmark.py to score policies offline. It is never attached to a
    RiskEvent and never reachable from the decision path.
    """
    event_id: str
    outcomes: dict[str, bool]

    def recovered_under(self, action: str) -> bool:
        return bool(self.outcomes.get(action, False))


# ---------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------

@dataclass
class CandidateAction:
    """One option under consideration, before economics or guardrails."""
    action: str
    # Discount offered, as a percentage. Only non-zero for discount actions.
    discount_pct: float = 0.0
    # Delay before execution, in hours. Used by delayed retry and by
    # quiet-hours deferral.
    delay_hours: int = 0
    channel: Optional[str] = None
    rationale: str = ""


@dataclass
class ScoredAction:
    """A candidate with its economics attached, and its verdict.

    Every money field is *incremental against doing nothing*, which is why
    `do_nothing` always scores exactly zero. That makes
    `expected_net_recovery_inr` directly readable as "rupees better than
    inaction", and any action scoring below zero is literally worse than
    leaving the event alone.
    """
    candidate: CandidateAction
    p_recover: float                # P(recover | event, this action)
    p_recover_baseline: float       # P(recover | event, do nothing)
    uplift: float                   # p_recover - p_recover_baseline
    gross_value_inr: float          # incremental value captured vs. doing nothing
    action_cost_inr: float
    expected_failure_cost_inr: float
    expected_chargeback_cost_inr: float
    cx_penalty_inr: float
    ltv_component_inr: float
    expected_net_recovery_inr: float
    # Guardrail verdict
    allowed: bool = True
    blocked_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # True when p_recover is an assumption rather than a fitted estimate.
    # See src/economics.py — two actions cannot be learned from an
    # exploration log and are modelled explicitly instead.
    probability_is_assumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.candidate.action
        d["discount_pct"] = self.candidate.discount_pct
        d["delay_hours"] = self.candidate.delay_hours
        d["channel"] = self.candidate.channel
        d.pop("candidate", None)
        # Round money and probabilities so audit lines stay readable
        for k in ["p_recover", "p_recover_baseline", "uplift"]:
            d[k] = round(d[k], 4)
        for k in ["gross_value_inr", "action_cost_inr", "expected_failure_cost_inr",
                  "expected_chargeback_cost_inr", "cx_penalty_inr",
                  "ltv_component_inr", "expected_net_recovery_inr"]:
            d[k] = round(d[k], 2)
        return d


@dataclass
class Decision:
    """The agent's final, executable determination for one event."""
    event_id: str
    event_type: str
    amount_inr: float
    customer_id: str

    root_cause: str
    root_cause_confidence: float
    root_cause_distribution: dict[str, float]

    chosen: ScoredAction
    considered: list[ScoredAction]

    requires_human_approval: bool
    approval_reason: str = ""
    guardrails_applied: list[str] = field(default_factory=list)

    # Explanation of why each alternative lost, keyed by action name.
    rejected_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def action(self) -> str:
        return self.chosen.candidate.action

    @property
    def expected_net_recovery_inr(self) -> float:
        return self.chosen.expected_net_recovery_inr
