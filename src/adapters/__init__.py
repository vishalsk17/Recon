"""
Action routing.

The dispatcher maps each action in the closed vocabulary to exactly one
adapter, and refuses anything unmapped. That refusal is the point: adding a
new kind of intervention requires editing this table, so "the agent invented a
new way to contact customers" is not a reachable state. A test asserts the
table covers `ALL_ACTIONS` with no gaps and no extras, which is what stops the
two lists drifting apart.

Two actions have adapters that exist only so the routing table is complete and
every decision produces an execution record:

  * `do_nothing` gets a real adapter rather than an early return, because a
    deliberate decision not to act is a decision, and it belongs in the audit
    trail alongside the others. Otherwise the log would silently contain only
    the events the agent chose to act on, which is a misleading picture of
    what it did — roughly a quarter of held-out events resolve to inaction,
    and most of those are cases where inaction was the correct, reasoned
    answer rather than an absence of one.

  * `request_human_review` creates a queue item and nothing else. Its result
    is always `awaiting_approval`, which is not a failure — it is the action
    completing successfully.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..schemas import ALL_ACTIONS, DO_NOTHING, REQUEST_HUMAN_REVIEW
from .base import (
    Adapter, ActionRequest, ActionResult, LiveExecutionRefused,
    ALL_STATUSES, CONSUMING_STATUSES, LIVE_ENV_VAR,
    STATUS_AWAITING, STATUS_DUPLICATE, STATUS_ERROR, STATUS_HALTED,
    STATUS_NO_ACTION, STATUS_REFUSED, STATUS_SIMULATED,
    live_execution_permitted, simulated_ref,
)
from .invoicing import InvoicingAdapter
from .messaging import MessagingAdapter, validate_customer_message
from .razorpay import RazorpayAdapter


class NullAdapter(Adapter):
    """Executes `do_nothing`, which means recording that nothing was done."""

    name = "none"
    handles = frozenset({DO_NOTHING})

    def _simulate(self, request: ActionRequest) -> ActionResult:
        return ActionResult(
            status=STATUS_NO_ACTION, adapter=self.name, action=request.action,
            detail=request.reason or "no action taken, by decision",
        )

    def _live(self, request: ActionRequest) -> ActionResult:
        # The one adapter where live and simulated are genuinely identical,
        # because there is no side effect to withhold.
        return self._simulate(request)


class ReviewQueueAdapter(Adapter):
    """Puts a case in front of a person. No customer-visible effect."""

    name = "review_queue"
    handles = frozenset({REQUEST_HUMAN_REVIEW})
    queues_for_human = True

    def _simulate(self, request: ActionRequest) -> ActionResult:
        return ActionResult(
            status=STATUS_AWAITING, adapter=self.name, action=request.action,
            provider_ref=simulated_ref("review", request),
            detail=(request.reason or "queued for analyst review")
                   + " — nothing is sent to the customer while a case is queued",
            notes=["customer_notified=false"],
        )

    def _live(self, request: ActionRequest) -> ActionResult:
        return self._simulate(request)


class Dispatcher:
    """Routes an ActionRequest to its adapter and returns the result.

    Holds one instance of each adapter, all sharing the same config and the
    same execution ledger, so idempotency and durable attempt counts are
    consistent no matter which action is being executed.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, ledger: Any = None):
        self.cfg = cfg
        self.ledger = ledger
        self._adapters: list[Adapter] = [
            RazorpayAdapter(cfg, ledger),
            MessagingAdapter(cfg, ledger),
            InvoicingAdapter(cfg, ledger),
            NullAdapter(cfg, ledger),
            ReviewQueueAdapter(cfg, ledger),
        ]
        self._route: dict[str, Adapter] = {}
        for adapter in self._adapters:
            for action in adapter.handles:
                if action in self._route:
                    raise RuntimeError(
                        f"{action!r} is claimed by both {self._route[action].name} "
                        f"and {adapter.name} — routing must be unambiguous"
                    )
                self._route[action] = adapter

    @property
    def routing_table(self) -> dict[str, str]:
        return {action: adapter.name for action, adapter in self._route.items()}

    def adapter_for(self, action: str) -> Adapter:
        adapter = self._route.get(action)
        if adapter is None:
            raise ValueError(
                f"no adapter registered for action {action!r}. The action "
                f"vocabulary is closed: add it to src/schemas.py::ALL_ACTIONS "
                f"and route it here, deliberately."
            )
        return adapter

    def execute(self, request: ActionRequest) -> ActionResult:
        adapter = self.adapter_for(request.action)
        try:
            return adapter.execute(request)
        except LiveExecutionRefused as exc:
            # Surfaced as a recorded error rather than a crash, so a sweep that
            # somehow reaches the live path stops that action and continues,
            # leaving an audit record of the attempt instead of a traceback and
            # a half-processed batch.
            return ActionResult(
                status=STATUS_ERROR, adapter=adapter.name, action=request.action,
                detail=str(exc),
            )


def unrouted_actions() -> list[str]:
    """Actions in the vocabulary with no adapter. Should always be empty."""
    table = Dispatcher().routing_table
    return [a for a in ALL_ACTIONS if a not in table]


__all__ = [
    "Adapter", "ActionRequest", "ActionResult", "Dispatcher",
    "LiveExecutionRefused", "NullAdapter", "ReviewQueueAdapter",
    "RazorpayAdapter", "MessagingAdapter", "InvoicingAdapter",
    "validate_customer_message", "live_execution_permitted", "simulated_ref",
    "unrouted_actions", "LIVE_ENV_VAR", "ALL_STATUSES", "CONSUMING_STATUSES",
    "STATUS_SIMULATED", "STATUS_NO_ACTION", "STATUS_REFUSED",
    "STATUS_DUPLICATE", "STATUS_AWAITING", "STATUS_HALTED", "STATUS_ERROR",
]
