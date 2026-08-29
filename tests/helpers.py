"""Shared fixtures and skip conditions.

Four things live here, and nothing else: a way to make a synthetic event
without repeating fourteen customer fields in every test, a way to build the
ranked candidate set and guardrail context those events get screened against,
a way to give a test its own audit trail so nothing writes to the shipped one,
and the skip predicates for tests that need generated data or trained weights.

The synthetic customer is *permissive* — consent on every channel, no DND, no
recent contact — which is the opposite of `schemas.default_customer`. That is
deliberate and the two must not be confused. `default_customer` is the
production stand-in for a missing record and is pessimistic so that ignorance
can never widen what the agent may do. This one exists so a guardrail test can
start from "everything is allowed" and switch off exactly one thing, which is
the only way to prove that the guardrail under test is the one doing the
blocking.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Any, Optional

from src import audit as A
from src import config as C
from src.schemas import CustomerProfile, RiskEvent

# ---------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------

DATA_PRESENT = all(os.path.exists(p) for p in
                   (C.CUSTOMERS_CSV, C.PAYMENTS_CSV, C.CHECKOUT_CSV, C.RECEIVABLES_CSV))
MODELS_PRESENT = all(os.path.exists(p) for p in
                     (C.ROOT_CAUSE_MODEL_PATH, C.UPLIFT_MODEL_PATH))
AUDIT_PRESENT = os.path.exists(C.AUDIT_LOG_PATH) and os.path.getsize(C.AUDIT_LOG_PATH) > 0

needs_data = unittest.skipUnless(
    DATA_PRESENT, "generated data missing — run `python data/generate_all.py`")
needs_models = unittest.skipUnless(
    MODELS_PRESENT, "model artifacts missing — run `python -m src.train`")
needs_audit = unittest.skipUnless(
    AUDIT_PRESENT, "no audit trail — run `python -m src.agent run --split test`")


# ---------------------------------------------------------------------
# Synthetic events
# ---------------------------------------------------------------------

def permissive_customer(**overrides: Any) -> CustomerProfile:
    """A customer for whom nothing is blocked, before overrides.

    See the module docstring: this is a test scaffold, not a production
    default, and it is only safe because a guardrail test needs to isolate one
    variable.
    """
    fields: dict[str, Any] = dict(
        customer_id="cust_test",
        tenure_months=24,
        prior_successful_payments=8,
        prior_late_payments=0,
        estimated_annual_value_inr=120_000.0,
        gross_margin_pct=35.0,
        repeat_purchase_probability=0.6,
        prior_coupon_redemptions=0,
        contacts_last_7d=0,
        hours_since_last_contact=9999.0,
        email_consent=True,
        whatsapp_consent=True,
        sms_consent=True,
        dnd_flagged=False,
        segment="smb",
    )
    fields.update(overrides)
    return CustomerProfile(**fields)


def make_event(event_type: str, amount_inr: float = 10_000.0, *,
               event_id: str = "evt_test_0001",
               features: Optional[dict[str, Any]] = None,
               occurred_at_hour: int = 12,
               customer: Optional[CustomerProfile] = None) -> RiskEvent:
    """One synthetic event whose feature dict matches the real data contract.

    The features are assembled from `dataio.SURFACE_SPEC` and
    `dataio.CUSTOMER_COLUMNS` rather than written out by hand, so a fixture
    cannot quietly drift from the columns the pipeline actually produces. If
    the contract grows a column with no value in FEATURE_VALUES below, this
    raises — see `test_fixtures.py`, which asserts exactly that, because a
    fixture that silently omits a field tests a shape nothing else has.

    `overrides` go in last, so a test can set `retry_count` or
    `dispute_flagged_in_ar` to aim at one guardrail.
    """
    from src import dataio

    spec = dataio.SURFACE_SPEC[event_type]
    built: dict[str, Any] = {}
    for column in list(spec["feature_cols"]) + list(dataio.CUSTOMER_COLUMNS):
        if column == spec["amount_col"]:
            built[column] = float(amount_inr)
        elif column == "hour_of_day":
            built[column] = int(occurred_at_hour)
        elif column in FEATURE_VALUES:
            built[column] = FEATURE_VALUES[column]
        else:
            raise KeyError(
                f"{event_type} feature {column!r} has no test value. Add one to "
                f"FEATURE_VALUES in tests/helpers.py — do not drop the column, "
                f"or this fixture stops resembling the real thing."
            )
    built.update(features or {})

    return RiskEvent(
        event_id=event_id,
        event_type=event_type,
        amount_inr=amount_inr,
        customer=customer or permissive_customer(),
        features=built,
        occurred_at="2026-08-28T12:00:00+05:30",
        occurred_at_hour=occurred_at_hour,
    )


# Plausible mid-distribution values, one per column the surfaces declare.
# Nothing here is extreme: a test that wants an extreme is expected to say so
# explicitly, so that reading the test tells you what it is actually varying.
FEATURE_VALUES: dict[str, Any] = {
    # payment_failure
    "payment_method": "card",
    "bank": "HDFC",
    "decline_code": "insufficient_funds",
    "retry_count": 0,
    "day_of_month": 15,
    # checkout_abandonment
    "items_count": 2,
    "device": "mobile",
    "is_returning_customer": True,
    "minutes_since_abandon": 180.0,
    "session_seconds": 420.0,
    "reached_payment_page": True,
    "checkout_errors_logged": 0,
    "payment_attempts_started": 1,
    # overdue_receivable
    "days_overdue": 21,
    "dispute_flagged_in_ar": False,
    "avg_days_late_last_6_invoices": 4.0,
    "purchase_order_required": False,
    "purchase_order_on_invoice": False,
    "invoice_revision_count": 0,
    "amount_variance_vs_contract_pct": 0.0,
    "acknowledged_in_portal": True,
    # customer columns, kept consistent with permissive_customer()
    "segment": "smb",
    "tenure_months": 24,
    "prior_successful_payments": 8,
    "prior_late_payments": 0,
    "estimated_annual_value_inr": 120_000.0,
    "gross_margin_pct": 35.0,
    "repeat_purchase_probability": 0.6,
    "prior_coupon_redemptions": 0,
    "contacts_last_7d": 0,
    "hours_since_last_contact": 9999.0,
    "email_consent": True,
    "whatsapp_consent": True,
    "sms_consent": True,
    "dnd_flagged": False,
}


# ---------------------------------------------------------------------
# Ranked candidates and guardrail context
# ---------------------------------------------------------------------

def clear_context(**overrides: Any):
    """A context in which no root-cause-driven guardrail fires.

    Confident, retryable, non-fraudulent. A test that wants a block passes the
    one field that causes it — `clear_context(root_cause="expired_card")` —
    which keeps the reason for the block visible in the test body instead of
    buried in a fixture.
    """
    from src.guardrails import GuardrailContext

    fields: dict[str, Any] = dict(
        root_cause="insufficient_funds",
        root_cause_confidence=0.90,
        root_cause_distribution={"insufficient_funds": 0.90, "fraud_suspected": 0.02},
    )
    fields.update(overrides)
    return GuardrailContext(**fields)


def ranked_options(event, p: float = 0.55, p_baseline: float = 0.30,
                   p_fraud: float = 0.0, cfg: Optional[Any] = None,
                   economics: Optional[Any] = None) -> list:
    """Every candidate for an event, priced and sorted, with no model involved.

    The probability map is filled in with one flat number for every arm, which
    is exactly what a test wants: with the model held constant, whatever
    differences appear between options are the work of the pricing and the
    guardrails, which are the things under test. A test that needs a specific
    option to win says so by passing a different `p` for that arm.

    `p_baseline` is written last and deliberately so. `uplift.BASELINE_ACTION`
    is the string `"do_nothing"`, which is also the key of a real candidate, so
    filling the map in the other order sets the baseline to `p` as well and
    every uplift in the list comes out as zero — a whole ranked set worth
    nothing, with no error anywhere. That is not hypothetical; it is what this
    helper did when it was first written, and the symptom was a selection test
    reporting that the agent had declined to act.
    """
    from src.economics import Economics, build_candidates
    from src.ml.uplift import BASELINE_ACTION, make_action_key

    economics = economics or Economics(cfg)
    candidates = build_candidates(event, cfg)
    probabilities = {}
    for candidate in candidates:
        probabilities[make_action_key(
            candidate.action,
            candidate.discount_pct or candidate.delay_hours or 0.0)] = p
    probabilities[BASELINE_ACTION] = p_baseline
    return economics.rank(event, probabilities, p_fraud=p_fraud, candidates=candidates)


def make_decision(event=None, *, p: float = 0.55, p_baseline: float = 0.30,
                  p_fraud: float = 0.0, ctx=None, cfg: Optional[Any] = None):
    """A fully decided case, built from pricing and guardrails with no model.

    Mirrors how `Toolbelt.run_plan` assembles a `Decision`, minus the two model
    calls: the root cause comes from the context and the recovery probabilities
    from `ranked_options`. That makes it usable in tests that have no trained
    weights, and — more usefully — it makes the decision's contents something
    the test states rather than something it has to discover.
    """
    from src.guardrails import Guardrails
    from src.schemas import Decision, PAYMENT_FAILURE

    event = event if event is not None else make_event(PAYMENT_FAILURE, 40_000.0)
    ctx = ctx if ctx is not None else clear_context()
    ranked = ranked_options(event, p=p, p_baseline=p_baseline, p_fraud=p_fraud, cfg=cfg)
    verdict = Guardrails(cfg).select(event, ranked, ctx)
    return Decision(
        event_id=event.event_id,
        event_type=event.event_type,
        amount_inr=event.amount_inr,
        customer_id=event.customer.customer_id,
        root_cause=ctx.root_cause,
        root_cause_confidence=float(ctx.root_cause_confidence),
        root_cause_distribution=dict(ctx.root_cause_distribution),
        chosen=verdict.chosen,
        considered=verdict.considered,
        requires_human_approval=verdict.requires_human_approval,
        approval_reason=verdict.approval_reason,
        guardrails_applied=verdict.guardrails_applied,
        rejected_reasons=verdict.rejected_reasons,
    )


# ---------------------------------------------------------------------
# Isolated audit trails
# ---------------------------------------------------------------------

class TempAudit:
    """An audit store, run index and approval queue in a throwaway directory.

    Every test that writes a record uses one of these. Nothing in the suite is
    allowed to append to the shipped trail: it is the demo evidence, its hash
    chain is quoted in the docs, and a test run that lengthened it would make
    those two disagree.
    """

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="recovery_test_audit_")
        self.store = A.AuditStore(os.path.join(self.dir, "decisions.jsonl"))
        self.runs = A.RunIndex(os.path.join(self.dir, "runs.jsonl"))
        self.approvals = A.ApprovalQueue(os.path.join(self.dir, "approvals.jsonl"),
                                         store=self.store)

    def rows(self) -> list[str]:
        if not os.path.exists(self.store.path):
            return []
        with open(self.store.path, "r", encoding="utf-8") as fh:
            return [line for line in fh if line.strip()]

    def size(self) -> int:
        return os.path.getsize(self.store.path) if os.path.exists(self.store.path) else 0

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class AuditCase(unittest.TestCase):
    """Base class for tests that write records.

    Also stands guard over the shipped trail. Every test here is *supposed* to
    write only into its own temp directory, but "supposed to" is how the suite
    once appended 576 records to `data/audit/decisions.jsonl` — an injected
    empty `AuditStore` is falsy, because the class defines `__len__`, so the
    `store or AuditStore()` fallback in `RecoveryAgent.__init__` quietly
    substituted the production trail. Both ends of that were fixed, but the
    failure was invisible for a whole test run, and the only reason it was
    caught is that a temp trail with zero rows in it made an unrelated
    assertion fail. So the size of the real file is checked after every test:
    if a leak like that ever happens again, the test that caused it says so.
    """

    def setUp(self) -> None:
        self.audit = TempAudit()
        self.addCleanup(self.audit.close)
        self.addCleanup(self._assert_shipped_trail_untouched,
                        *(_shipped_trail_sizes()))

    def _assert_shipped_trail_untouched(self, *before: int) -> None:
        after = _shipped_trail_sizes()
        for path, was, now in zip(_SHIPPED_TRAILS, before, after):
            if was != now:
                self.fail(
                    f"this test wrote {now - was:,} bytes into {os.path.basename(path)}, "
                    f"which is shipped evidence and must never be appended to by the "
                    f"suite. Something used a default AuditStore/RunIndex instead of "
                    f"the injected one — check for a truthiness-based fallback."
                )


_SHIPPED_TRAILS = (C.AUDIT_LOG_PATH, C.RUN_INDEX_PATH, C.APPROVALS_PATH)


def _shipped_trail_sizes() -> tuple[int, ...]:
    return tuple(os.path.getsize(p) if os.path.exists(p) else -1
                 for p in _SHIPPED_TRAILS)

