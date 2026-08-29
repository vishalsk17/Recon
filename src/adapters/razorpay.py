"""
Payment gateway adapter (Razorpay-shaped).

Handles the three actions that touch a payment instrument: retry it now,
retry it later, or stop and flag it for fraud review. As with every adapter
here, there is no transport — see src/adapters/base.py for why.

Two things worth reading before the code.

**What "stop and flag" actually does.** It is an internal action and nothing
more: it marks the payment as not-to-be-retried and creates a review item for
the merchant's own fraud team. It does not block the customer, does not
report anyone to a bureau or a shared blacklist, does not notify a third
party, and does not deny future business. This distinction is the whole
reason the action is safe to automate. A system that can automatically
retry a payment is doing something reversible and small; a system that can
automatically brand a person as a fraudster is doing something that follows
them around, and is not something a probability estimate should be trusted
with. The agent may decline to spend money. It may not accuse anyone.

**Why the retry caps are re-checked here.** src/guardrails.py already applied
the per-payment attempt cap, the minimum interval and the never-retry causes.
This adapter checks the attempt cap and the interval again, from the durable
ledger rather than from the in-memory sweep state. The upstream check knows
about this run; the ledger knows about every run. A payment retried twice on
Monday and once on Tuesday has had three attempts, and only the ledger can
see that. Rapid repeated attempts against one instrument are the signature
of card testing, so this is the one place where "we already checked" is not
a good enough reason to skip a check.
"""

from __future__ import annotations

from typing import Optional

from ..schemas import DELAYED_RETRY, IMMEDIATE_RETRY, STOP_AND_FLAG_FRAUD
from .base import (
    Adapter, ActionRequest, ActionResult, STATUS_SIMULATED, simulated_ref,
)


class RazorpayAdapter(Adapter):
    """Gateway-facing actions. Simulation only."""

    name = "razorpay"
    handles = frozenset({IMMEDIATE_RETRY, DELAYED_RETRY, STOP_AND_FLAG_FRAUD})

    def _authorise(self, request: ActionRequest) -> Optional[str]:
        if request.action == STOP_AND_FLAG_FRAUD:
            # Nothing to authorise. Declining to take money and asking a human
            # to look at it cannot harm the customer, so there is no safety
            # check that would ever refuse it.
            return None

        retries = self.cfg["retries"]
        cap = int(retries["max_attempts_per_payment"])
        min_gap = int(retries["min_hours_between_attempts"])

        # Prefer the durable count over whatever the caller passed. The
        # request's `prior_attempts` reflects this sweep's view; the ledger
        # reflects every sweep that ever ran.
        attempts = int(request.prior_attempts)
        if self.ledger is not None:
            attempts = max(attempts, self.ledger.prior_attempts(request.event_id))

        if attempts >= cap:
            return (f"{attempts} attempt(s) already made against this payment and "
                    f"the cap is {cap} — further automated attempts against one "
                    f"instrument are a card-testing pattern, not a recovery "
                    f"strategy")

        if request.action == IMMEDIATE_RETRY and attempts > 0:
            return (f"an immediate retry after {attempts} prior attempt(s) would "
                    f"breach the {min_gap}h minimum interval — schedule a delayed "
                    f"retry instead")

        if request.action == DELAYED_RETRY and int(request.delay_hours) < min_gap:
            return (f"delay of {request.delay_hours}h is under the {min_gap}h "
                    f"minimum interval between attempts")
        return None

    def _simulate(self, request: ActionRequest) -> ActionResult:
        if request.action == STOP_AND_FLAG_FRAUD:
            return ActionResult(
                status=STATUS_SIMULATED, adapter=self.name, action=request.action,
                provider_ref=simulated_ref("flag", request),
                detail=(
                    "would mark this payment as not-to-be-retried and open an "
                    "internal fraud-review item. No customer-visible effect, no "
                    "external report, no block on future business"
                ),
                notes=["internal_only=true", "customer_notified=false"],
            )

        attempts = int(request.prior_attempts)
        if self.ledger is not None:
            attempts = max(attempts, self.ledger.prior_attempts(request.event_id))
        when = ("immediately" if request.action == IMMEDIATE_RETRY
                else f"in {request.delay_hours}h")
        return ActionResult(
            status=STATUS_SIMULATED, adapter=self.name, action=request.action,
            provider_ref=simulated_ref("retry", request),
            detail=(
                f"would re-present order {request.event_id} to the gateway {when} "
                f"for {request.amount_inr:,.2f} INR — attempt "
                f"{attempts + 1} of {self.cfg['retries']['max_attempts_per_payment']}"
            ),
            scheduled_in_hours=int(request.delay_hours),
            notes=[f"attempt_number={attempts + 1}"],
        )
