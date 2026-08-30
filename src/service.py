"""
The application layer behind the Recovery Command Centre.

Every function here takes plain arguments and returns JSON-able dicts. None of
them know what HTTP is, which is the point: the read model, the approval flow
and the narration hand-off can all be exercised in a unit test without opening
a socket, and src/server.py stays thin enough to audit in one sitting.

Two design choices are load-bearing rather than stylistic.

**The service reads the audit trail, not the models.** With one exception
(`explain_event`, which prices a single event on demand) every endpoint is a
query over the append-only log written by a previous sweep. That is not a
performance decision. It means the dashboard shows what the agent *did*, which
is the only thing an operator can act on, and it means opening the dashboard
cannot cause a decision to be made. A dashboard that re-scores events on page
load would show numbers no audit record supports, and the first time those two
disagreed the log would be the thing people stopped trusting.

**There is exactly one write path, and it releases an already-made decision.**
`resolve_approval` cannot invent an action, change an amount, or reach a
customer the agent had not already decided to reach — it looks up a decision
that was gated, and records a person's yes or no. Everything else about the
side effect was fixed when the decision was written. An approval endpoint that
accepted an action name would be a remote-code-execution hole with a friendly
name, so this one accepts a decision id, a boolean, and who is signing.

**Nothing here starts a sweep.** An earlier version of this module exposed a
`start_sweep` guarded three ways — dry-run only, kill switch honoured, capped
at 50 events per surface — and it worked. It was removed anyway, for two
reasons. The dashboard never called it, so it was capability with no consumer
sitting on a port with no authentication; and its existence meant the paragraph
above had to be written as "one write path, plus a guarded second one", which
is a materially weaker thing to be able to say. Sweeps run from the CLI, where
the command names the scale of what it is about to do. If you are reading this
because you want the button back: the argument against it is not that the
guards were inadequate, it is that a read model with a single boolean write is
a surface you can describe completely in one sentence, and that is worth more
than a click.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional

from . import audit as A
from . import config as C
from . import dataio
from .agent import RecoveryAgent
from .schemas import EVENT_TYPES

# The dashboard is an operator tool, not a data export. Every list endpoint is
# capped so a stray request cannot pull a hundred megabytes of audit records
# through a browser tab.
MAX_PAGE = 200
DEFAULT_PAGE = 50


class ServiceError(Exception):
    """An error with an HTTP status attached, raised by the layer above."""

    def __init__(self, status: int, message: str, detail: Any = None):
        self.status = status
        self.message = message
        self.detail = detail
        super().__init__(message)


def _page(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_PAGE
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise ServiceError(400, f"limit must be an integer, got {limit!r}")
    if value < 1:
        raise ServiceError(400, "limit must be at least 1")
    return min(value, MAX_PAGE)


# ---------------------------------------------------------------------
# Read model
# ---------------------------------------------------------------------

def health() -> dict[str, Any]:
    """Enough to tell a broken deployment from a working one at a glance."""
    from . import narrator
    cfg = C.load_config()
    have_models = (os.path.exists(C.ROOT_CAUSE_MODEL_PATH)
                   and os.path.exists(C.UPLIFT_MODEL_PATH))
    return {
        "ok": True,
        "code_version": C.CODE_VERSION,
        "policy_version": C.policy_version(cfg),
        "dry_run": bool(cfg["execution"]["dry_run"]),
        "live_transport_available": False,   # structurally, not by configuration
        "kill_switch_engaged": C.kill_switch_engaged(cfg),
        "models_trained": have_models,
        "narration_available": narrator.available(cfg),
        # Which vendor narration would use right now, or null. A name only —
        # never a key, and never the endpoint, because a health endpoint is
        # the first thing anyone reads and the last place a secret should be.
        "narration_provider": narrator.configured_provider(cfg),
        "audit_records": len(A.AuditStore()),
    }


def overview() -> dict[str, Any]:
    """The headline numbers, taken from the most recent finished run.

    The transactional and retention components of expected net recovery are
    returned separately and never pre-added. The retention term is a weighted
    claim about future revenue; on a small transaction it can dwarf the money
    actually in play, so a single blended figure would be the most misleading
    number on the page. The dashboard shows them apart for the same reason the
    CLI does.
    """
    runs_rows = A.RunIndex().read()
    finished = [r for r in runs_rows if r.get("phase") == "finished"]
    latest = finished[-1] if finished else {}

    pending = A.ApprovalQueue().pending()
    return {
        "has_run": bool(finished),
        "run_id": latest.get("run_id"),
        "finished_at": latest.get("recorded_at"),
        "dry_run": latest.get("dry_run", True),
        "events": latest.get("events", 0),
        "at_risk_inr": latest.get("at_risk_inr", 0.0),
        "value_at_risk_inr": latest.get("value_at_risk_inr", 0.0),
        "projected_transactional_inr": latest.get("projected_transactional_inr", 0.0),
        "projected_retention_inr": latest.get("projected_retention_inr", 0.0),
        "actions": latest.get("actions", {}),
        "statuses": latest.get("statuses", {}),
        "guardrail_blocks": latest.get("guardrail_blocks", 0),
        "gated_for_approval": latest.get("gated_for_approval", 0),
        "retries_issued": latest.get("retries_issued", 0),
        "discount_committed_inr": latest.get("discount_committed_inr", 0.0),
        "surfaces": latest.get("surfaces", []),
        "chain_head": latest.get("chain_head", ""),
        "pending_approvals": len(pending),
        "total_runs": len(finished),
    }


def runs(limit: Optional[int] = None) -> dict[str, Any]:
    """Run history, newest first, with the started/finished pair collapsed.

    `RunIndex.finish` splats the summary into the record rather than nesting
    it, so these fields are read from the top level. Runs that started and
    never finished are omitted on purpose: an unfinished run has no summary to
    report, and inventing zeros for one would make a crash look like a sweep
    that found nothing to do.
    """
    rows = A.RunIndex().read()
    started = {r.get("run_id"): r for r in rows if r.get("phase") == "started"}
    out = []
    for row in rows:
        if row.get("phase") != "finished":
            continue
        began = started.get(row.get("run_id")) or {}
        out.append({
            "run_id": row.get("run_id"),
            "started_at": began.get("recorded_at"),
            "finished_at": row.get("recorded_at"),
            "split": began.get("split", ""),
            "dry_run": row.get("dry_run", True),
            "events": row.get("events", 0),
            "at_risk_inr": row.get("at_risk_inr", 0.0),
            "value_at_risk_inr": row.get("value_at_risk_inr", 0.0),
            "projected_transactional_inr": row.get("projected_transactional_inr", 0.0),
            "projected_retention_inr": row.get("projected_retention_inr", 0.0),
            "gated_for_approval": row.get("gated_for_approval", 0),
            "guardrail_blocks": row.get("guardrail_blocks", 0),
            "chain_head": row.get("chain_head", ""),
            "policy_version": began.get("policy_version", ""),
            "code_version": began.get("code_version", ""),
        })
    out.reverse()
    return {"runs": out[:_page(limit)], "total": len(out)}


def _decision_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """The compact form used in lists. Deliberately not the full record.

    `AuditStore.append_decision` writes the decision's fields at the top level
    of the record rather than nesting them, so this reads them from there. The
    timestamp key is `recorded_at`, set by the store, never by the caller.
    """
    chosen = record.get("chosen", {}) or {}
    considered = record.get("considered") or []
    return {
        "decision_id": record.get("decision_id"),
        "run_id": record.get("run_id"),
        "recorded_at": record.get("recorded_at"),
        "event_id": record.get("event_id"),
        "event_type": record.get("event_type"),
        "amount_inr": record.get("amount_inr", 0.0),
        "value_at_risk_inr": record.get("value_at_risk_inr", 0.0),
        "customer_id": record.get("customer_id"),
        "root_cause": record.get("root_cause"),
        "root_cause_confidence": record.get("root_cause_confidence", 0.0),
        "action": record.get("action") or chosen.get("action"),
        "channel": chosen.get("channel"),
        "discount_pct": chosen.get("discount_pct", 0.0),
        "delay_hours": chosen.get("delay_hours", 0),
        "expected_net_recovery_inr": record.get("expected_net_recovery_inr", 0.0),
        "ltv_component_inr": chosen.get("ltv_component_inr", 0.0),
        "p_recover": chosen.get("p_recover", 0.0),
        "p_recover_baseline": chosen.get("p_recover_baseline", 0.0),
        "probability_is_assumed": chosen.get("probability_is_assumed", False),
        "requires_human_approval": record.get("requires_human_approval", False),
        "approval_reason": record.get("approval_reason", ""),
        "dry_run": record.get("dry_run", True),
        "blocked_count": sum(1 for s in considered if s.get("blocked_by")),
        "options_considered": len(considered),
    }


def decisions(run_id: Optional[str] = None, limit: Optional[int] = None,
              action: Optional[str] = None, surface: Optional[str] = None
              ) -> dict[str, Any]:
    """Decisions from a run, largest value at risk first."""
    store = A.AuditStore()
    rid = run_id or A.RunIndex().latest_run_id()
    if rid is None:
        return {"run_id": None, "decisions": [], "total": 0}
    rows = [_decision_row(r) for r in A.decisions_for_run(rid, store)]
    if action:
        rows = [r for r in rows if r["action"] == action]
    if surface:
        if surface not in EVENT_TYPES:
            raise ServiceError(400, f"unknown surface {surface!r}",
                               {"known": list(EVENT_TYPES)})
        rows = [r for r in rows if r["event_type"] == surface]
    rows.sort(key=lambda r: -float(r["value_at_risk_inr"] or 0.0))
    return {"run_id": rid, "decisions": rows[:_page(limit)], "total": len(rows)}


def pending_approvals(run_id: Optional[str] = None, limit: Optional[int] = None
                      ) -> dict[str, Any]:
    """The work queue: decisions the agent made but will not execute alone.

    `ApprovalQueue.pending` already returns the full decision records with
    resolved ones filtered out, so there is no second lookup here. Sorted by
    value at risk so the reviewer's attention goes where the money is.

    The size of this queue is the honest measure of how much autonomy the
    system is actually claiming. On the held-out split roughly a fifth of
    decisions and a majority of the value land here, because the gate fires on
    the largest amounts — that is a design outcome, not a shortfall.
    """
    rows = A.ApprovalQueue().pending(run_id)
    out = [_decision_row(r) for r in rows]
    out.sort(key=lambda r: -float(r["value_at_risk_inr"] or 0.0))
    total_value = sum(float(r["value_at_risk_inr"] or 0.0) for r in out)
    total_enr = sum(float(r["expected_net_recovery_inr"] or 0.0) for r in out)
    return {
        "pending": out[:_page(limit)],
        "total": len(out),
        "total_value_at_risk_inr": round(total_value, 2),
        "total_expected_net_recovery_inr": round(total_enr, 2),
    }


def decision_detail(decision_id: str) -> dict[str, Any]:
    """The full evidence for one decision.

    improvements.md item 7 asks that the audit trail be decision *evidence*
    rather than a log line, so this returns everything: the considered set with
    each option's economics and the guardrails that refused it, the tool trace,
    the term-by-term arithmetic, the execution context, and whatever the
    dispatcher subsequently did. It is the screen a reviewer should be able to
    defend a decision from, six months later, to someone who is annoyed.
    """
    store = A.AuditStore()
    record = A.find_decision(decision_id, store)
    if record is None:
        raise ServiceError(404, f"no decision with id {decision_id!r}")
    considered = list(record.get("considered") or [])
    considered.sort(key=lambda s: -float(s.get("expected_net_recovery_inr") or 0.0))

    executions = [
        {k: v for k, v in r.items() if k not in {"record_hash", "prev_hash"}}
        for r in store.read()
        if r.get("record_type") == "execution" and r.get("decision_id") == decision_id
    ]
    resolution = A.ApprovalQueue().resolutions().get(decision_id)
    return {
        "summary": _decision_row(record),
        "root_cause_distribution": record.get("root_cause_distribution", {}),
        "considered": considered,
        "guardrails_applied": record.get("guardrails_applied", []),
        "rejected_reasons": record.get("rejected_reasons", {}),
        "arithmetic": record.get("arithmetic", ""),
        "tool_trace": record.get("tool_trace", []),
        "execution_context": record.get("execution_context", {}),
        "executions": executions,
        "approval": resolution,
        "record_hash": record.get("record_hash", ""),
        "prev_hash": record.get("prev_hash", ""),
        "policy_version": record.get("policy_version", ""),
        "code_version": record.get("code_version", ""),
    }


def verify_audit() -> dict[str, Any]:
    """Recompute the hash chain over the whole log."""
    return A.AuditStore().verify_chain()


def benchmark() -> dict[str, Any]:
    """The saved uplift report, if one has been produced."""
    if not os.path.exists(C.BENCHMARK_PATH):
        raise ServiceError(
            404, "no benchmark report yet",
            {"hint": "run `python -m src.benchmark` to produce one"},
        )
    with open(C.BENCHMARK_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def model_card() -> dict[str, Any]:
    """Held-out metrics for both model families, as trained.

    Served from the report `src/train.py` writes, so the dashboard cannot
    show a metric that was not produced by the training run that made the
    weights currently on disk.
    """
    path = os.path.join(C.ARTIFACT_DIR, "training_report.json")
    if not os.path.exists(path):
        raise ServiceError(404, "no training report yet",
                           {"hint": "run `python -m src.train`"})
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _plain(value: Any) -> Any:
    """Convert `config.load_config`'s frozen structures into JSON-able ones.

    The loader deep-freezes the config — mappings become read-only proxies and
    sequences become tuples — so that no module can mutate the policy at
    runtime. `json` handles tuples but not the proxies, so this unwraps both.
    """
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def policy_snapshot() -> dict[str, Any]:
    """The limits currently in force, so the dashboard can show the leash.

    A curated subset rather than the whole config: the point is to let an
    operator see what is binding, not to hand a browser every knob. Nothing
    here is writable — the config is loaded frozen and there is no endpoint
    that edits it. Changing policy means editing config/policy.yaml and
    restarting, which leaves a diff in version control instead of an
    unattributable change in a running process.
    """
    cfg = C.load_config()
    contact = cfg["contact"]
    return {
        "policy_version": C.policy_version(cfg),
        "code_version": C.CODE_VERSION,
        "limits": _plain(cfg["limits"]),
        "quiet_hours": {
            "start_hour": contact["quiet_hours_start"],
            "end_hour": contact["quiet_hours_end"],
            "note": "local time; outreach inside this window is deferred, "
                    "not cancelled, and the deferral is enforced again at the "
                    "egress boundary",
        },
        "contact": _plain(contact),
        "retries": _plain(cfg["retries"]),
        "receivables": _plain(cfg["receivables"]),
        "execution": {
            "dry_run": bool(cfg["execution"]["dry_run"]),
            "live_transport_available": False,
            "note": "no live provider transport is implemented in this build; "
                    "the dispatcher's live path raises unconditionally",
        },
    }


def explain_event(event_id: str) -> dict[str, Any]:
    """Price one event on demand, without recording or executing anything.

    The single read path that touches the models. It exists because "what
    would you do with this one" is the question an operator asks while looking
    at something the last sweep did not cover, and answering it by starting a
    sweep would be a side effect in response to a question.

    `RecoveryAgent.decide` is the method that makes no audit record and calls
    no adapter, which is exactly the property needed here. Nothing this returns
    can be acted on: to act, a decision has to exist in the log.

    Two fields carry the distinction that matters. `call_wrote_nothing` is an
    assertion about *this request* — it is always true, and it is stated rather
    than implied because a response containing a priced action and a rupee
    figure otherwise looks identical to one describing an action that was taken.
    `recorded_decision_id` is a fact about the *event*: the decision the log
    already holds for it, if any, so an operator can compare what the agent
    would do now against what it actually did. An earlier version returned a
    single `recorded: false`, which read as "this event is not in the audit
    trail" while meaning "this call did not add to it" — true either way, but
    answering a question nobody asked and contradicting the one they did.
    """
    for surface in EVENT_TYPES:
        for event in dataio.load_events(surface, None, None):
            if event.event_id == event_id:
                agent = RecoveryAgent()
                agent.toolbelt.prime([event])
                decision = agent.decide(event)
                prior = _recorded_decision_for(event_id)
                return {
                    "event_id": event_id,
                    "surface": surface,
                    "amount_inr": event.amount_inr,
                    "would_choose": decision.chosen.to_dict(),
                    "root_cause": decision.root_cause,
                    "root_cause_confidence": decision.root_cause_confidence,
                    "requires_human_approval": decision.requires_human_approval,
                    "approval_reason": decision.approval_reason,
                    "considered": [s.to_dict() for s in decision.considered],
                    # True by construction: `decide` writes no audit record and
                    # calls no adapter. Asserted in the payload so the dashboard
                    # can say it on screen instead of the reader assuming it.
                    "call_wrote_nothing": True,
                    "recorded_decision_id": prior["decision_id"] if prior else None,
                    "recorded_at": prior["recorded_at"] if prior else None,
                    "recorded_action": prior["action"] if prior else None,
                }
    raise ServiceError(404, f"no event with id {event_id!r}")


def _recorded_decision_for(event_id: str) -> Optional[dict[str, Any]]:
    """The compact row for the log's latest decision about an event, or None."""
    record = A.latest_decision_for_event(event_id)
    return _decision_row(record) if record else None


# ---------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------

def resolve_approval(decision_id: str, *, approver: str, granted: bool,
                     reason: str = "") -> dict[str, Any]:
    """Record a person's sign-off, and release the side effect if granted.

    Anonymous approvals are refused. An audit trail whose approvals are signed
    "system" or "" answers the question "who authorised this" with "nobody",
    which is worse than having no approval step at all — it manufactures the
    appearance of oversight. The CLI refuses them too; this is the same rule at
    a second entry point rather than a new one.
    """
    who = (approver or "").strip()
    if not who:
        raise ServiceError(400, "an approver name is required",
                           {"why": "approvals are attributed to a person; an "
                                   "unsigned approval is not oversight"})
    if who.lower() in {"system", "agent", "automation", "none", "null", "-"}:
        raise ServiceError(
            400, f"{who!r} is not a person",
            {"why": "the approval gate exists so that a human takes "
                    "responsibility; naming the machine defeats it"},
        )
    agent = RecoveryAgent()
    try:
        return agent.resolve_approval(decision_id, approver=who, granted=bool(granted),
                                      reason=reason or "", execute=True)
    except KeyError as exc:
        raise ServiceError(404, str(exc)) from exc
    except ValueError as exc:
        raise ServiceError(409, str(exc)) from exc


def narrate(decision_id: str, role: str) -> dict[str, Any]:
    """Generate operator-facing or customer-facing text for a recorded decision.

    Requires a real API key and says so plainly when there is not one. The
    503 is the designed behaviour, not a degradation: see src/narrator.py on
    why a template fallback was rejected.

    The decision is loaded from the audit log rather than recomputed, so the
    text is narration of something that actually happened. A draft is never
    dispatched from here — it is returned for a person to read, and if it is a
    customer message it still has to travel through the approval queue and the
    egress validator to reach anyone.
    """
    from . import narrator as N

    record = A.find_decision(decision_id)
    if record is None:
        raise ServiceError(404, f"no decision with id {decision_id!r}")
    if not N.available():
        raise ServiceError(
            503, f"narration is unavailable: no LLM API key is set ({N.credentials_hint()})",
            {"why": "this build requires a real API key and has no template "
                    "fallback, so that nobody mistakes canned text for "
                    "generated text",
             "providers": [p.name for p in N.PROVIDERS],
             "note": "the recovery pipeline does not need narration; "
                     "sweeps, approvals and the audit trail all work without it"},
        )
    decision = _decision_from_record(record)
    try:
        narrator = N.Narrator()
        draft = narrator.narrate(role, decision)
    except N.RoleNotPermitted as exc:
        raise ServiceError(400, str(exc)) from exc
    except N.DraftRejected as exc:
        # A refused draft is reported as a refusal, with the reasons. The
        # alternative — returning the text with a warning — would put
        # unvalidated model output on an operator's screen looking like output
        # that passed.
        raise ServiceError(422, "the generated text was refused by validation",
                           {"problems": exc.problems}) from exc
    except N.NarratorError as exc:
        raise ServiceError(502, str(exc)) from exc
    return {"decision_id": decision_id, **draft.to_dict()}


def _decision_from_record(record: Mapping[str, Any]):
    """Rebuild enough of a Decision from its audit record to narrate it.

    Only the fields the fact sheet reads are reconstructed, and every one of
    them is copied from the record rather than recomputed. That matters: if
    this re-scored the event, the narration could describe an action the log
    does not contain, which is the one thing the narrator is built to prevent.

    The record stores decision fields at the top level, with the chosen and
    considered actions as `ScoredAction.to_dict()` payloads — flat dicts whose
    `action`, `discount_pct`, `delay_hours` and `channel` keys came from the
    nested candidate. So the candidate is rebuilt from those four keys.
    """
    from .schemas import CandidateAction, Decision, ScoredAction

    chosen = record.get("chosen", {}) or {}

    def to_scored(payload: Mapping[str, Any]) -> ScoredAction:
        return ScoredAction(
            candidate=CandidateAction(
                action=payload.get("action", "do_nothing"),
                discount_pct=float(payload.get("discount_pct", 0.0) or 0.0),
                delay_hours=int(payload.get("delay_hours", 0) or 0),
                channel=payload.get("channel"),
            ),
            p_recover=float(payload.get("p_recover", 0.0) or 0.0),
            p_recover_baseline=float(payload.get("p_recover_baseline", 0.0) or 0.0),
            uplift=float(payload.get("uplift", 0.0) or 0.0),
            gross_value_inr=float(payload.get("gross_value_inr", 0.0) or 0.0),
            action_cost_inr=float(payload.get("action_cost_inr", 0.0) or 0.0),
            expected_failure_cost_inr=float(payload.get("expected_failure_cost_inr", 0.0) or 0.0),
            expected_chargeback_cost_inr=float(
                payload.get("expected_chargeback_cost_inr", 0.0) or 0.0),
            cx_penalty_inr=float(payload.get("cx_penalty_inr", 0.0) or 0.0),
            ltv_component_inr=float(payload.get("ltv_component_inr", 0.0) or 0.0),
            expected_net_recovery_inr=float(
                payload.get("expected_net_recovery_inr", 0.0) or 0.0),
            allowed=bool(payload.get("allowed", True)),
            blocked_by=list(payload.get("blocked_by") or []),
            notes=list(payload.get("notes") or []),
            probability_is_assumed=bool(payload.get("probability_is_assumed", False)),
        )

    return Decision(
        event_id=record.get("event_id", ""),
        event_type=record.get("event_type", ""),
        amount_inr=float(record.get("amount_inr", 0.0) or 0.0),
        customer_id=record.get("customer_id", ""),
        root_cause=record.get("root_cause", ""),
        root_cause_confidence=float(record.get("root_cause_confidence", 0.0) or 0.0),
        root_cause_distribution=dict(record.get("root_cause_distribution") or {}),
        chosen=to_scored(chosen),
        considered=[to_scored(s) for s in (record.get("considered") or [])],
        requires_human_approval=bool(record.get("requires_human_approval", False)),
        approval_reason=record.get("approval_reason", ""),
        guardrails_applied=list(record.get("guardrails_applied") or []),
        rejected_reasons=dict(record.get("rejected_reasons") or {}),
    )
