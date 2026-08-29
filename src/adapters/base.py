"""
Adapter base: the boundary between deciding and doing.

Everything above this line reasons about money. Everything below it would
*move* money, or reach a customer, if a transport existed. So this is the
right place to be blunt about what this build will and will not do.

**There is no live transport in this repository.** No adapter here contains
an HTTP call to a payment gateway, an email service or a WhatsApp provider.
Not a commented-out one, not one behind a feature flag. `_simulate()` is
implemented; `_live()` raises. A test asserts that no module under
src/adapters imports a network client, so this stays true as the code grows.

That is a deliberate choice, not an unfinished one. A recovery agent is by
construction a program whose purpose is to charge cards and message people at
scale. The failure modes are not subtle — a loop bug is a retry storm against
an issuer, an off-by-one in a frequency cap is a harassment complaint, and a
mis-scoped credential is someone else's money. Shipping the decision engine
without the transport means the interesting part is fully reviewable and the
dangerous part is absent. Wiring a provider is then a deliberate act by
someone with production accountability, in a repository with real secret
management, not a flag a demo can trip.

The interlock is still built and still tested, because the *shape* of the
safety mechanism is part of the deliverable:

    dry_run: false  in config/policy.yaml
    AND  RECOVERY_AGENT_ALLOW_LIVE=1  in the environment

Both are required. One alone does nothing. Even with both, `_live()` refuses,
because there is nothing behind it. Two independent keys means neither a
config commit nor an environment mistake can act alone, and the environment
half deliberately cannot be set from inside the repo.

Beyond that interlock, every adapter re-checks the constraints the guardrail
layer already applied — consent, quiet hours, retry caps, discount ceilings.
That duplication is intentional. src/guardrails.py is the policy engine; these
are the last gate before egress. If a future caller ever builds an
ActionRequest by hand and skips the pipeline, the caps still hold. A safety
property that lives in exactly one place is one refactor away from not
existing.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .. import config as C

# Environment variable forming the second half of the live-execution interlock.
LIVE_ENV_VAR = "RECOVERY_AGENT_ALLOW_LIVE"

# Execution statuses. Closed set, mirrored in the audit records and counted by
# the dashboard, so a new one cannot appear without a deliberate change here.
STATUS_SIMULATED = "simulated"          # would have happened; recorded, not sent
STATUS_NO_ACTION = "no_action"          # do_nothing — nothing to do, by design
STATUS_REFUSED = "refused"              # an adapter-level safety check declined
STATUS_DUPLICATE = "skipped_duplicate"  # idempotency key already executed
STATUS_AWAITING = "awaiting_approval"   # queued for a human, nothing sent
STATUS_HALTED = "halted"                # kill switch engaged
STATUS_ERROR = "error"

ALL_STATUSES = (
    STATUS_SIMULATED, STATUS_NO_ACTION, STATUS_REFUSED, STATUS_DUPLICATE,
    STATUS_AWAITING, STATUS_HALTED, STATUS_ERROR,
)

# Statuses that consumed a real allowance (a contact, a gateway attempt).
# Refusals and duplicates deliberately do not: an action that never happened
# must not eat into a customer's contact budget.
CONSUMING_STATUSES = frozenset({STATUS_SIMULATED})


class LiveExecutionRefused(RuntimeError):
    """Raised when something tries to execute for real."""


@dataclass(frozen=True)
class ActionRequest:
    """Everything an adapter is allowed to know.

    Note what is absent. No card number, no email address, no phone number,
    no customer name. The adapter is told *which customer* and *which
    channel*; resolving that to a destination is the job of the provider
    integration that this build does not ship. So the decision layer, the
    audit trail and the adapters collectively never hold a contactable
    identifier — only an opaque customer id.

    `consented`, `dnd_flagged` and `local_hour` are passed as plain values
    rather than by handing over the CustomerProfile, for the same reason:
    the adapter gets the answers it needs to re-check a safety rule, and
    nothing else.
    """
    run_id: str
    decision_id: str
    event_id: str
    event_type: str
    action: str
    idempotency_key: str
    customer_id: str
    amount_inr: float = 0.0
    discount_pct: float = 0.0
    delay_hours: int = 0
    channel: Optional[str] = None
    consented: bool = False
    dnd_flagged: bool = False
    local_hour: int = 12
    prior_attempts: int = 0
    days_overdue: int = 0
    approval_granted: bool = False
    requires_approval: bool = False
    message_body: str = ""
    reason: str = ""


@dataclass
class ActionResult:
    """What happened, or what would have happened."""
    status: str
    adapter: str
    action: str
    provider_ref: str = ""
    detail: str = ""
    scheduled_in_hours: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def consumed_allowance(self) -> bool:
        return self.status in CONSUMING_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "adapter": self.adapter,
            "action": self.action,
            "provider_ref": self.provider_ref,
            "detail": self.detail,
            "scheduled_in_hours": self.scheduled_in_hours,
            "notes": list(self.notes),
        }


def live_execution_permitted(cfg: Optional[Mapping[str, Any]] = None) -> bool:
    """True only when both halves of the interlock are set.

    Kept as a function rather than a constant so it is evaluated at call
    time. A module-level boolean read at import would be a stale answer if
    the environment changed, and "stale" is the wrong failure direction for
    a question about whether real money may move.
    """
    cfg = cfg or C.load_config()
    config_allows = not bool(cfg["execution"].get("dry_run", True))
    env_allows = os.environ.get(LIVE_ENV_VAR, "") == "1"
    return config_allows and env_allows


def simulated_ref(prefix: str, request: ActionRequest) -> str:
    """A deterministic stand-in for a provider reference.

    Deterministic so that replaying a run produces identical audit records
    and a diff of two runs shows only genuine differences. Prefixed `sim_`
    so no reference in the audit trail can ever be mistaken for one issued
    by a real provider.
    """
    digest = hashlib.sha256(
        f"{prefix}|{request.idempotency_key}".encode()
    ).hexdigest()[:14]
    return f"sim_{prefix}_{digest}"


class Adapter:
    """Template for every side-effecting integration.

    Subclasses implement `_authorise` (adapter-level safety checks) and
    `_simulate`. They do not override `execute`, which enforces the ordering
    of the checks: halt, then approval, then idempotency, then safety, then
    the interlock. That ordering is itself a safety property — for instance,
    the kill switch is checked before anything else so that engaging it stops
    even actions that would otherwise be considered already-approved.
    """

    name = "adapter"
    handles: frozenset[str] = frozenset()
    # True for the adapter whose whole job is to put a case in front of a
    # person. Without this flag the generic approval gate below would
    # short-circuit `request_human_review` — the one action for which
    # "waiting on a human" is success rather than a block — and the adapter
    # would be unreachable dead code.
    queues_for_human = False

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None,
                 ledger: Any = None):
        self.cfg = cfg or C.load_config()
        self.ledger = ledger

    # -- subclass hooks ----------------------------------------------

    def _authorise(self, request: ActionRequest) -> Optional[str]:
        """Return a refusal reason, or None to permit. Override in subclasses."""
        return None

    def _simulate(self, request: ActionRequest) -> ActionResult:
        raise NotImplementedError

    def _live(self, request: ActionRequest) -> ActionResult:
        """Deliberately unimplemented. See this module's docstring."""
        raise LiveExecutionRefused(
            f"{self.name}: live execution is not implemented in this build. "
            f"There is no transport behind this adapter — it can simulate and "
            f"record, and that is all it can do. Integrating a real provider is "
            f"an intentional act for a repository with production credentials, "
            f"secret management and an on-call owner, not a flag flipped in a "
            f"demo. See src/adapters/base.py for the reasoning."
        )

    # -- the fixed pipeline ------------------------------------------

    def execute(self, request: ActionRequest) -> ActionResult:
        if request.action not in self.handles:
            raise ValueError(
                f"{self.name} does not handle {request.action!r} — the action "
                f"vocabulary is closed and routing is explicit"
            )

        if C.kill_switch_engaged(self.cfg):
            return ActionResult(
                status=STATUS_HALTED, adapter=self.name, action=request.action,
                detail=(
                    f"kill switch file "
                    f"{self.cfg['execution'].get('kill_switch_file', 'HALT')!r} is "
                    f"present at the project root — no action executes, including "
                    f"in dry run"
                ),
            )

        if (request.requires_approval and not request.approval_granted
                and not self.queues_for_human):
            return ActionResult(
                status=STATUS_AWAITING, adapter=self.name, action=request.action,
                detail=request.reason or "queued for human sign-off; nothing sent",
            )

        if self.ledger is not None and request.idempotency_key:
            if self.ledger.has_executed(request.idempotency_key):
                return ActionResult(
                    status=STATUS_DUPLICATE, adapter=self.name, action=request.action,
                    detail=(
                        f"idempotency key {request.idempotency_key[:12]}… already "
                        f"executed in an earlier run — re-running a sweep replays "
                        f"decisions but never repeats a side effect"
                    ),
                )

        refusal = self._authorise(request)
        if refusal:
            return ActionResult(
                status=STATUS_REFUSED, adapter=self.name, action=request.action,
                detail=refusal,
            )

        if live_execution_permitted(self.cfg):
            return self._live(request)
        return self._simulate(request)
