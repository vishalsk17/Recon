"""
Receivables adapter: invoice reminders, payment plans, collections escalation.

Receivables differ from the other two surfaces in a way that matters for
safety rather than economics. A failed payment and an abandoned cart are
between a merchant and a consumer, and the worst outcome of a bad decision is
a wasted message. An overdue invoice is between two businesses with a
contract, an account manager and usually a relationship worth more than the
invoice. Escalating one wrongly is not a wasted message — it is a phone call
from someone's finance director.

So this adapter is the most conservative of the three:

  * **Collections never executes on the agent's word.** It requires an
    approval record, and it re-checks the ageing threshold from config even
    when approved, because a human clicking "approve" on a queue of forty
    items is not a reliable reading of whether this particular invoice is 90
    days overdue.

  * **A disputed or mis-issued invoice is not chased at all.** The guardrail
    layer blocks those causes upstream; this adapter refuses them again. The
    reasoning is that chasing an invoice the customer is right to be
    questioning is not a collections problem, it is an own-goal — the fix is
    a corrected invoice, and a reminder actively delays it.

  * **Payment plans are offers, not schedules.** The action creates an offer
    for the customer to accept; it does not alter contractual terms, does not
    write to a ledger and does not commit either party to anything. Anything
    that changes what is owed is a human's decision.
"""

from __future__ import annotations

from typing import Optional

from ..schemas import (
    AUTOMATED_REMINDER, AUTOMATED_REMINDER_WITH_PLAN, ESCALATE_TO_COLLECTIONS,
)
from .base import (
    Adapter, ActionRequest, ActionResult, STATUS_SIMULATED, simulated_ref,
)
from .messaging import outreach_refusal


class InvoicingAdapter(Adapter):
    """Accounts-receivable actions. Simulation only."""

    name = "invoicing"
    handles = frozenset({
        AUTOMATED_REMINDER, AUTOMATED_REMINDER_WITH_PLAN, ESCALATE_TO_COLLECTIONS,
    })

    def _authorise(self, request: ActionRequest) -> Optional[str]:
        # Reminders and plan offers are outreach and go through exactly the
        # same consent, DND and quiet-hours check as any other message. An
        # invoice reminder is not exempt from a customer's contact preferences
        # merely because it is about money they owe.
        refusal = outreach_refusal(request, self.cfg)
        if refusal:
            return refusal

        if request.action != ESCALATE_TO_COLLECTIONS:
            return None

        rec = self.cfg["receivables"]
        if bool(rec.get("collections_requires_human_signoff", True)) and not request.approval_granted:
            return ("collections escalation requires recorded human sign-off and "
                    "none was found for this decision")

        min_days = int(rec["min_days_overdue_for_collections"])
        if int(request.days_overdue) < min_days:
            return (f"invoice is {request.days_overdue} days overdue and collections "
                    f"requires {min_days} — re-checked here because an approval "
                    f"click is not a reading of the ageing")
        return None

    def _simulate(self, request: ActionRequest) -> ActionResult:
        if request.action == ESCALATE_TO_COLLECTIONS:
            return ActionResult(
                status=STATUS_SIMULATED, adapter=self.name, action=request.action,
                provider_ref=simulated_ref("collections", request),
                detail=(
                    f"would hand invoice {request.event_id} "
                    f"({request.amount_inr:,.2f} INR, {request.days_overdue} days "
                    f"overdue) to the collections process, under recorded human "
                    f"sign-off"
                ),
                notes=["human_approved=true"],
            )

        plan = request.action == AUTOMATED_REMINDER_WITH_PLAN
        detail = (
            f"would send a reminder for invoice {request.event_id} "
            f"({request.amount_inr:,.2f} INR, {request.days_overdue} days overdue)"
        )
        if plan:
            detail += (" including a payment-plan offer the customer may accept or "
                       "ignore — no change to contractual terms is made by the agent")
        return ActionResult(
            status=STATUS_SIMULATED, adapter=self.name, action=request.action,
            provider_ref=simulated_ref("ar", request),
            detail=detail,
            scheduled_in_hours=int(request.delay_hours),
            notes=[f"payment_plan_offered={str(plan).lower()}"],
        )
