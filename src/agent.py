"""
The orchestrator: run a sweep, record it, execute it.

`python -m src.agent run` is the main entry point. It loads the events on each
surface, runs the fixed plan from src/tools.py over each one, writes an
append-only audit record, dispatches the chosen action to its adapter, and
prints a money-denominated summary.

Three ordering decisions in here are load-bearing rather than incidental.

**Events are worked in descending value-at-risk order.** Sweep-wide budgets —
the retry ceiling and the discount budget — are order dependent by
construction: whoever is processed after a budget runs out gets refused. On
the held-out sweep the retry budget binds exactly, so this is not a
hypothetical. Working the largest exposures first means a budget that runs out
has been spent on the events that mattered most, rather than on whatever order
the CSV happened to be in. Value at risk, not face value: an invoice is worth
its whole face amount to recover while a cart is worth its margin, so sorting
on raw amount would systematically over-prioritise carts by roughly 3x.

**The durable ledger overrides the static customer dimension.** A frequency cap
of two contacts per seven days is only real if the second sweep can see what
the first one sent. The event's own `contacts_last_7d` reflects the source
system at extract time; the audit trail reflects what this agent actually did.
The agent takes the stricter of the two before any guardrail runs, so a cap
cannot be evaded by running twice. Measured on 180 held-out events: a clean
ledger produces 81 customer contacts, and running the identical sweep again
immediately afterwards produces 14, with 28 actions refused outright as exact
duplicates and only 2 new side effects. The share of events resolving to
`do_nothing` rises from 20% to 56% — which is the correct answer, because by
then there is nothing left to say to those customers this week.

**Decide, record, then execute — in that order.** The audit record is written
before the adapter is called, not after. If the process dies mid-sweep, the
decision is still on record with no execution beside it, which is a state a
reviewer can interpret. The reverse order can produce a side effect with no
record of why, which is the one outcome that must not be possible.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from . import audit as A
from . import config as C
from . import dataio
from .adapters import (
    ActionRequest, Dispatcher, STATUS_DUPLICATE, live_execution_permitted,
)
from .economics import Economics
from .guardrails import SweepBudget
from .schemas import (
    ACTION_CHANNEL, EVENT_TYPES, OUTREACH_ACTIONS, RETRY_ACTIONS,
    Decision, RiskEvent,
)
from .tools import PLAN, Toolbelt, arithmetic_for

CONTACT_WINDOW_HOURS = 24 * 7


# ---------------------------------------------------------------------
# Idempotency windows
# ---------------------------------------------------------------------

def idempotency_window(action: str, run_id: str, now: Optional[datetime] = None) -> str:
    """The scope within which an action may happen at most once.

    Retries and outreach get a calendar-day window, so the same customer
    cannot be charged or messaged twice for the same event on the same day
    even if the sweep is run repeatedly. Everything else is scoped to the run,
    because recording a second `do_nothing` or opening a second review item is
    harmless and a per-run scope keeps the audit trail readable.

    The consequence is deliberate and worth knowing before running the demo
    twice: a second sweep on the same day reports its retries and messages as
    `skipped_duplicate`. That is idempotency working, not a failure.
    """
    if action in RETRY_ACTIONS or action in OUTREACH_ACTIONS:
        now = now or datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d")
    return run_id


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

@dataclass
class RunSummary:
    run_id: str
    dry_run: bool
    events: int = 0
    at_risk_inr: float = 0.0
    value_at_risk_inr: float = 0.0
    projected_net_recovery_inr: float = 0.0
    projected_transactional_inr: float = 0.0
    projected_retention_inr: float = 0.0
    actions: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    gated_for_approval: int = 0
    guardrail_blocks: int = 0
    discount_committed_inr: float = 0.0
    retries_issued: int = 0
    surfaces: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "events": self.events,
            "at_risk_inr": round(self.at_risk_inr, 2),
            "value_at_risk_inr": round(self.value_at_risk_inr, 2),
            "projected_net_recovery_inr": round(self.projected_net_recovery_inr, 2),
            "projected_transactional_inr": round(self.projected_transactional_inr, 2),
            "projected_retention_inr": round(self.projected_retention_inr, 2),
            "actions": dict(sorted(self.actions.items(), key=lambda kv: -kv[1])),
            "statuses": dict(sorted(self.statuses.items(), key=lambda kv: -kv[1])),
            "gated_for_approval": self.gated_for_approval,
            "guardrail_blocks": self.guardrail_blocks,
            "discount_committed_inr": round(self.discount_committed_inr, 2),
            "retries_issued": self.retries_issued,
            "surfaces": list(self.surfaces),
        }


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------

class RecoveryAgent:
    """Runs sweeps. One instance per process; models are loaded once."""

    def __init__(self,
                 cfg: Optional[Mapping[str, Any]] = None,
                 toolbelt: Optional[Toolbelt] = None,
                 store: Optional[A.AuditStore] = None,
                 runs: Optional[A.RunIndex] = None,
                 approvals: Optional[A.ApprovalQueue] = None):
        self.cfg = cfg or C.load_config()
        self.toolbelt = toolbelt if toolbelt is not None else Toolbelt(self.cfg)
        self.store = store if store is not None else A.AuditStore()
        self.runs = runs if runs is not None else A.RunIndex()
        self.approvals = (approvals if approvals is not None
                          else A.ApprovalQueue(store=self.store))
        self.economics = Economics(self.cfg)
        self.ledger = A.ExecutionLedger.load(self.store)
        self.dispatcher = Dispatcher(self.cfg, self.ledger)

    # -- ledger reconciliation ---------------------------------------

    def reconcile_with_ledger(self, event: RiskEvent) -> RiskEvent:
        """Take the stricter of the source system's contact history and ours.

        Returns the event unchanged when the ledger has nothing to add, so a
        first run does no copying. Both fields move only in the safe
        direction: contacts up, hours-since down.
        """
        cid = event.customer.customer_id
        ledger_contacts = self.ledger.contacts_within(cid, CONTACT_WINDOW_HOURS)
        ledger_hours = self.ledger.hours_since_last_contact(cid)
        if not ledger_contacts and ledger_hours is None:
            return event

        customer = event.customer
        contacts = max(customer.contacts_last_7d, ledger_contacts)
        hours = customer.hours_since_last_contact
        if ledger_hours is not None:
            hours = min(hours, ledger_hours)
        if contacts == customer.contacts_last_7d and hours == customer.hours_since_last_contact:
            return event
        return dataclasses.replace(
            event,
            customer=dataclasses.replace(
                customer, contacts_last_7d=contacts, hours_since_last_contact=hours,
            ),
        )

    def value_at_risk(self, event: RiskEvent) -> float:
        """What recovering this event is actually worth, for prioritisation."""
        return event.amount_inr * self.economics.margin_fraction(event)

    # -- single event -------------------------------------------------

    def decide(self, event: RiskEvent, budget: Optional[SweepBudget] = None) -> Decision:
        """Decide one event without recording or executing anything.

        Used by the read-only API endpoint, where the caller wants to see what
        the agent would do without that inspection itself becoming an action.
        """
        return self.toolbelt.run_plan(self.reconcile_with_ledger(event), budget).decision

    def build_request(self, decision: Decision, event: RiskEvent, run_id: str, *,
                      approval_granted: bool = False,
                      message_body: str = "") -> ActionRequest:
        """Assemble the adapter's view of a decision.

        Everything the adapter needs to re-check a safety rule, and nothing
        else. Note there is no contact address and no payment instrument here:
        the adapter is told the channel and the opaque customer id, and
        resolving those to a destination is the job of the provider
        integration this build does not ship.
        """
        action = decision.chosen.candidate.action
        customer = event.customer
        channel = decision.chosen.candidate.channel or ACTION_CHANNEL.get(action)
        return ActionRequest(
            run_id=run_id,
            decision_id=A.decision_id(run_id, decision.event_id),
            event_id=decision.event_id,
            event_type=decision.event_type,
            action=action,
            idempotency_key=A.idempotency_key(
                decision.event_id, action, idempotency_window(action, run_id)
            ),
            customer_id=decision.customer_id,
            amount_inr=decision.amount_inr,
            discount_pct=decision.chosen.candidate.discount_pct,
            delay_hours=decision.chosen.candidate.delay_hours,
            channel=channel,
            consented=customer.has_consent(channel) if channel else False,
            dnd_flagged=customer.dnd_flagged,
            local_hour=event.occurred_at_hour,
            prior_attempts=int(event.features.get("retry_count", 0) or 0),
            days_overdue=int(float(event.features.get("days_overdue", 0) or 0)),
            approval_granted=approval_granted,
            requires_approval=decision.requires_human_approval,
            message_body=message_body,
            reason=decision.approval_reason or "; ".join(decision.chosen.notes[:1]),
        )

    @staticmethod
    def _execution_context(request: ActionRequest) -> dict[str, Any]:
        """The scalars needed to replay this action after a human approves it.

        Recorded on the decision so the approval path does not have to reload
        and re-derive the event. Only non-identifying scalars — the same
        forbidden-key screen in src/audit.py applies to this block as to
        everything else.
        """
        return {
            "channel": request.channel,
            "consented": request.consented,
            "dnd_flagged": request.dnd_flagged,
            "local_hour": request.local_hour,
            "prior_attempts": request.prior_attempts,
            "days_overdue": request.days_overdue,
            "discount_pct": request.discount_pct,
            "delay_hours": request.delay_hours,
            "idempotency_key": request.idempotency_key,
        }

    # -- sweep --------------------------------------------------------

    def run(self, *,
            surfaces: Optional[Iterable[str]] = None,
            split: Optional[str] = "test",
            limit_per_surface: Optional[int] = None,
            execute: bool = True,
            run_id: Optional[str] = None,
            verbose: bool = False) -> RunSummary:
        C.ensure_dirs()
        surfaces = list(surfaces) if surfaces else list(EVENT_TYPES)
        run_id = run_id or A.new_run_id()
        dry_run = not live_execution_permitted(self.cfg)

        events: list[RiskEvent] = []
        for surface in surfaces:
            events.extend(dataio.load_events(surface, split, limit_per_surface))
        events = [self.reconcile_with_ledger(e) for e in events]

        # Highest value at risk first. See this module's docstring.
        events.sort(key=self.value_at_risk, reverse=True)

        # One scoring pass per surface, after reconciliation and before any
        # decision. See Toolbelt.prime — this is what makes every event in a
        # run priced against identical model state.
        self.toolbelt.clear_primed()
        self.toolbelt.prime(events)

        at_risk_total = sum(e.amount_inr for e in events)
        budget = SweepBudget(at_risk_total_inr=at_risk_total)

        summary = RunSummary(run_id=run_id, dry_run=dry_run, events=len(events),
                             at_risk_inr=at_risk_total, surfaces=surfaces)
        summary.value_at_risk_inr = sum(self.value_at_risk(e) for e in events)

        self.runs.start(run_id, dry_run=dry_run, surfaces=surfaces, split=split,
                        event_count=len(events), at_risk_inr=at_risk_total)

        if C.kill_switch_engaged(self.cfg):
            print(f"kill switch engaged ({self.cfg['execution']['kill_switch_file']} "
                  f"present) — every event will resolve to no action.", file=sys.stderr)

        for event in events:
            plan = self.toolbelt.run_plan(event, budget)
            decision = plan.decision
            action = decision.chosen.candidate.action

            budget.record(event, decision.chosen)

            request = self.build_request(decision, event, run_id)
            record = self.store.append_decision(
                decision, run_id, dry_run=dry_run,
                explanation=arithmetic_for(decision),
                extra={
                    "tool_trace": plan.trace_dicts(),
                    "execution_context": self._execution_context(request),
                    "value_at_risk_inr": round(self.value_at_risk(event), 2),
                },
            )

            summary.actions[action] = summary.actions.get(action, 0) + 1
            summary.projected_net_recovery_inr += decision.expected_net_recovery_inr
            summary.projected_retention_inr += decision.chosen.ltv_component_inr
            summary.projected_transactional_inr += (
                decision.expected_net_recovery_inr - decision.chosen.ltv_component_inr
            )
            summary.guardrail_blocks += sum(1 for s in decision.considered if s.blocked_by)
            if decision.requires_human_approval:
                summary.gated_for_approval += 1

            if not execute:
                continue

            result = self.dispatcher.execute(request)
            summary.statuses[result.status] = summary.statuses.get(result.status, 0) + 1
            self.store.append_execution(
                run_id=run_id, decision_id_=record["decision_id"],
                event_id=decision.event_id, action=action,
                adapter=result.adapter, status=result.status,
                idempotency_key_=request.idempotency_key, dry_run=dry_run,
                provider_ref=result.provider_ref, detail=result.detail,
                extra={"scheduled_in_hours": result.scheduled_in_hours,
                       "notes": result.notes},
            )
            # Keep the in-process ledger current so later events in the same
            # sweep see this one. Without it, idempotency would only hold
            # across runs and not within one.
            if result.consumed_allowance and request.idempotency_key:
                self.ledger.executed_keys.add(request.idempotency_key)

            if verbose:
                print(f"  {decision.event_id:<14} {action:<42} "
                      f"{decision.expected_net_recovery_inr:>10,.2f} INR  {result.status}")

        summary.discount_committed_inr = budget.discount_committed_inr
        summary.retries_issued = budget.retries_issued
        self.runs.finish(run_id, summary=summary.to_dict(),
                         chain_head=self.store.chain_head())
        return summary

    # -- approval workflow -------------------------------------------

    def resolve_approval(self, decision_id_: str, *, approver: str, granted: bool,
                         reason: str = "", execute: bool = True) -> dict[str, Any]:
        """Record a human's verdict on a gated decision, then act on it.

        This is the second half of improvement #6. Gating an action is only a
        workflow if there is a way to release it, and releasing it has to
        produce the side effect that was withheld — otherwise "requires
        approval" quietly means "will never happen".
        """
        record = A.find_decision(decision_id_, self.store)
        if record is None:
            raise KeyError(f"no decision recorded with id {decision_id_!r}")
        if not record.get("requires_human_approval"):
            raise ValueError(
                f"decision {decision_id_} was not gated for approval — there is "
                f"nothing to release"
            )
        if decision_id_ in self.approvals.resolutions():
            raise ValueError(f"decision {decision_id_} has already been resolved")

        self.approvals.resolve(decision_id_, approver=approver, granted=granted,
                               reason=reason)
        if not granted:
            return {"decision_id": decision_id_, "granted": False,
                    "status": "declined",
                    "detail": "approval declined; no action taken"}
        if not execute:
            return {"decision_id": decision_id_, "granted": True,
                    "status": "approved_not_executed",
                    "detail": "approval recorded; execution deferred"}

        ctx = record.get("execution_context", {}) or {}
        request = ActionRequest(
            run_id=str(record["run_id"]),
            decision_id=str(record["decision_id"]),
            event_id=str(record["event_id"]),
            event_type=str(record["event_type"]),
            action=str(record["action"]),
            idempotency_key=str(ctx.get("idempotency_key", "")),
            customer_id=str(record["customer_id"]),
            amount_inr=float(record["amount_inr"]),
            discount_pct=float(ctx.get("discount_pct", 0.0) or 0.0),
            delay_hours=int(ctx.get("delay_hours", 0) or 0),
            channel=ctx.get("channel"),
            consented=bool(ctx.get("consented", False)),
            dnd_flagged=bool(ctx.get("dnd_flagged", False)),
            local_hour=int(ctx.get("local_hour", 12)),
            prior_attempts=int(ctx.get("prior_attempts", 0) or 0),
            days_overdue=int(ctx.get("days_overdue", 0) or 0),
            approval_granted=True,
            requires_approval=True,
            reason=f"released by {approver}",
        )
        result = self.dispatcher.execute(request)
        self.store.append_execution(
            run_id=request.run_id, decision_id_=request.decision_id,
            event_id=request.event_id, action=request.action,
            adapter=result.adapter, status=result.status,
            idempotency_key_=request.idempotency_key,
            dry_run=not live_execution_permitted(self.cfg),
            provider_ref=result.provider_ref, detail=result.detail,
            extra={"released_by": approver, "post_approval": True},
        )
        return {"decision_id": decision_id_, "granted": True,
                "status": result.status, "detail": result.detail}


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_summary(summary: RunSummary, cfg: Mapping[str, Any]) -> None:
    s = summary
    mode = "DRY RUN (adapters simulate; nothing is sent)" if s.dry_run else "LIVE"
    print()
    print("=" * 74)
    print(f"  RUN {s.run_id}    {mode}")
    print("=" * 74)
    print(f"  surfaces               {', '.join(s.surfaces)}")
    print(f"  events worked          {s.events:,}")
    print(f"  gross amount at risk   {s.at_risk_inr:>16,.2f} INR")
    print(f"  value at risk          {s.value_at_risk_inr:>16,.2f} INR   "
          f"(margin on sales, face value on receivables)")
    print()
    print("  PROJECTED EXPECTED NET RECOVERY — incremental vs. doing nothing")
    print(f"    this transaction      {s.projected_transactional_inr:>16,.2f} INR   "
          f"({s.projected_transactional_inr / max(1.0, s.value_at_risk_inr):.1%} of value at risk)")
    print(f"    retained future value {s.projected_retention_inr:>16,.2f} INR   "
          f"(LTV term, weight {float(cfg['economics']['ltv_retention_weight']):.2f})")
    print(f"    total                 {s.projected_net_recovery_inr:>16,.2f} INR")
    print("    The two lines are separated deliberately. Only the first is money that")
    print("    lands in this billing cycle; the second is a weighted claim about future")
    print("    revenue and is the softer of the two numbers. Quote them apart.")
    print()
    print("  ACTIONS CHOSEN")
    for action, count in sorted(s.actions.items(), key=lambda kv: -kv[1]):
        print(f"    {action:<46} {count:>6}  ({count / max(1, s.events):>4.0%})")
    if s.statuses:
        print()
        print("  EXECUTION STATUS")
        for status, count in sorted(s.statuses.items(), key=lambda kv: -kv[1]):
            print(f"    {status:<46} {count:>6}")
        if s.statuses.get(STATUS_DUPLICATE):
            print(f"    ^ {s.statuses[STATUS_DUPLICATE]} action(s) were already executed in an "
                  f"earlier run today and were skipped.")
            print(f"      That is the idempotency check working, not a failure.")
    print()
    print("  SAFETY")
    print(f"    options blocked by guardrails                {s.guardrail_blocks:>6}")
    print(f"    decisions gated for human approval           {s.gated_for_approval:>6}"
          f"  ({s.gated_for_approval / max(1, s.events):.0%})")
    print(f"    retries issued / sweep ceiling               "
          f"{s.retries_issued:>6} / {cfg['retries']['max_retries_per_sweep']}")
    print(f"    discount committed                     {s.discount_committed_inr:>12,.0f} INR"
          f"  of {s.at_risk_inr * float(cfg['limits']['max_discount_budget_pct_of_at_risk']) / 100:,.0f} INR budget")
    print()
    print(f"  audit  {C.AUDIT_LOG_PATH}")
    print(f"  next   python -m src.agent pending      # decisions awaiting sign-off")
    print(f"         python -m src.agent verify       # check the audit chain")
    print(f"         python -m src.benchmark          # uplift vs. baselines")
    print("=" * 74)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    agent = RecoveryAgent()
    surfaces = [args.surface] if args.surface else None
    summary = agent.run(
        surfaces=surfaces,
        split=None if args.split == "all" else args.split,
        limit_per_surface=args.limit,
        execute=not args.no_execute,
        verbose=args.verbose,
    )
    print_summary(summary, agent.cfg)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = A.AuditStore().verify_chain()
    print(f"audit file : {C.AUDIT_LOG_PATH}")
    print(f"records    : {result['records']:,}")
    if result["ok"]:
        print(f"integrity  : OK — {result['reason']}")
        print(f"chain head : {result['head']}")
        return 0
    print(f"integrity  : FAILED at line {result['broken_at_line']}")
    print(f"reason     : {result['reason']}")
    return 1


def _cmd_pending(args: argparse.Namespace) -> int:
    queue = A.ApprovalQueue()
    items = queue.pending(run_id=args.run_id)
    if not items:
        print("no decisions awaiting sign-off")
        return 0
    print(f"{len(items)} decision(s) awaiting sign-off\n")
    for item in items[:args.limit]:
        print(f"  {item['decision_id']}  {item['event_type']:<22} "
              f"{item['amount_inr']:>12,.2f} INR  {item['action']}")
        print(f"      why gated : {item['approval_reason']}")
        print(f"      economics : {item['arithmetic'][:150]}")
        print()
    if len(items) > args.limit:
        print(f"  ... and {len(items) - args.limit} more")
    print("release one with:")
    print("  python -m src.agent approve <decision_id> --approver \"your name\"")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    agent = RecoveryAgent()
    try:
        result = agent.resolve_approval(
            args.decision_id, approver=args.approver,
            granted=not args.deny, reason=args.reason,
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"decision {result['decision_id']}: {result['status']}")
    print(f"  {result['detail']}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    """Print the agent's capabilities. Useful in a review; trivial to write."""
    from .adapters import Dispatcher as D
    from .schemas import ALL_ACTIONS
    print("PLAN (fixed sequence, no branches, no LLM in the loop)")
    for i, step in enumerate(PLAN, 1):
        print(f"  {i}. {step}")
    print("\nACTION VOCABULARY (closed) -> ADAPTER")
    table = D().routing_table
    for action in ALL_ACTIONS:
        print(f"  {action:<46} {table[action]}")
    print(f"\nEXECUTION  dry_run={C.load_config()['execution']['dry_run']}  "
          f"live_permitted={live_execution_permitted()}")
    print("  No adapter in this build has a live transport. See src/adapters/base.py.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.agent",
        description="Bounded revenue-recovery agent. Dry run by default.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="work a sweep of at-risk events")
    run.add_argument("--surface", choices=list(EVENT_TYPES),
                     help="restrict to one surface (default: all three)")
    run.add_argument("--split", default="test", choices=["train", "test", "all"],
                     help="which deterministic split to work (default: test)")
    run.add_argument("--limit", type=int, default=None,
                     help="cap events per surface, for a quick demo")
    run.add_argument("--no-execute", action="store_true",
                     help="decide and record only; do not call adapters")
    run.add_argument("--verbose", action="store_true", help="print every decision")
    run.set_defaults(func=_cmd_run)

    verify = sub.add_parser("verify", help="check the audit hash chain")
    verify.set_defaults(func=_cmd_verify)

    pending = sub.add_parser("pending", help="list decisions awaiting human sign-off")
    pending.add_argument("--run-id", default=None)
    pending.add_argument("--limit", type=int, default=10)
    pending.set_defaults(func=_cmd_pending)

    approve = sub.add_parser("approve", help="release or decline a gated decision")
    approve.add_argument("decision_id")
    approve.add_argument("--approver", required=True,
                         help="who is signing off (recorded verbatim)")
    approve.add_argument("--reason", default="")
    approve.add_argument("--deny", action="store_true", help="decline instead of approve")
    approve.set_defaults(func=_cmd_approve)

    plan = sub.add_parser("plan", help="print the fixed plan and action vocabulary")
    plan.set_defaults(func=_cmd_plan)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
