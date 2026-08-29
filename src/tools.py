"""
The agent's tools, and the fixed plan it executes over them.

"Agent" is a loaded word, so it is worth being precise about what this one is.
It is not an LLM in a loop deciding what to do next. It is a deterministic
orchestrator over a closed set of named tools, executed in a fixed order,
where every call is recorded. The LLM (src/narrator.py) is downstream of all
of it and may only put already-decided facts into words.

That is a deliberate architectural choice rather than a limitation. The
autonomy in a recovery system is not "what should I try next" — the pipeline
is genuinely always the same: work out why it failed, estimate what each
intervention would achieve, price them, check the rules, act. The autonomy is
in the *judgement at each step*, and that judgement is the models and the
economics. Letting a language model choose the sequence would add
non-determinism to the one part of the system that has no need of it, and
would make the audit question "why did you do this" unanswerable in general.

So the plan is a tuple you can read:

    PLAN = (classify_root_cause, generate_candidates, estimate_recovery,
            score_actions, apply_guardrails)

Five steps, no branches, no loops. `Toolbelt.run_plan` walks exactly that
sequence for one event and returns the decision plus the trace of what each
tool was asked and what it answered. The trace goes into the audit record,
which means "which tools ran, in what order, with what result" is a matter of
record rather than of reading the source and hoping it matches production.

The tools are also the seam for testing. Each one is independently callable
with plain arguments, so a test can feed `score_actions` a hand-built
probability dict and assert the arithmetic without standing up a model.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import pandas as pd

from . import config as C
from . import dataio
from .economics import Economics, build_candidates, candidate_action_key, explain
from .guardrails import GuardrailContext, Guardrails, SweepBudget, Verdict
from .ml.root_cause import RootCauseBundle
from .ml.uplift import RecoveryBundle
from .schemas import CandidateAction, Decision, RiskEvent, ScoredAction

# The agent's entire repertoire. Named here so the answer to "what can this
# thing do" is a five-element tuple rather than a code review.
PLAN: tuple[str, ...] = (
    "classify_root_cause",
    "generate_candidates",
    "estimate_recovery",
    "score_actions",
    "apply_guardrails",
)


@dataclass
class ToolCall:
    """One recorded invocation.

    `inputs` and `outputs` are deliberately *summaries*, not the full
    arguments. A trace that embedded every feature vector would be unreadable
    and would drag customer data into the audit file, which is exactly what
    src/audit.py refuses to accept. So each tool declares what is worth
    recording about its own call — normally the event id and a handful of
    scalars.
    """
    tool: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class PlanResult:
    """Everything one pass of the plan produced."""
    decision: Decision
    verdict: Verdict
    trace: list[ToolCall] = field(default_factory=list)

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.trace]


class Toolbelt:
    """Holds the models and exposes them as named, traced tools.

    Constructed once per process. Loading the model artifacts is the expensive
    part, and doing it per event would make a sweep quadratically slower for
    no reason, so the belt is long-lived and the per-event state (the sweep
    budget) is passed in rather than held here.
    """

    def __init__(self,
                 cfg: Optional[Mapping[str, Any]] = None,
                 root_cause: Optional[RootCauseBundle] = None,
                 recovery: Optional[RecoveryBundle] = None):
        self.cfg = cfg or C.load_config()
        self.root_cause = root_cause or RootCauseBundle.load(C.ROOT_CAUSE_MODEL_PATH)
        self.recovery = recovery or RecoveryBundle.load(C.UPLIFT_MODEL_PATH)
        self.economics = Economics(self.cfg)
        self.guardrails = Guardrails(self.cfg)
        self._trace: list[ToolCall] = []
        self._primed: dict[str, tuple[tuple[str, float, dict], dict[str, float]]] = {}

        self._tools: dict[str, Callable[..., Any]] = {
            "classify_root_cause": self._classify_root_cause,
            "generate_candidates": self._generate_candidates,
            "estimate_recovery": self._estimate_recovery,
            "score_actions": self._score_actions,
            "apply_guardrails": self._apply_guardrails,
        }
        missing = [name for name in PLAN if name not in self._tools]
        if missing:
            raise RuntimeError(f"PLAN references unimplemented tools: {missing}")

    # -- registry ----------------------------------------------------

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name, recording the call.

        Unknown names raise. There is no fallback, no fuzzy match and no
        dynamic registration — the same closed-vocabulary argument that
        applies to actions applies to tools.
        """
        fn = self._tools.get(name)
        if fn is None:
            raise ValueError(
                f"unknown tool {name!r}. The toolbelt is closed; available "
                f"tools are {sorted(self._tools)}."
            )
        started = time.perf_counter()
        summary_in, result, summary_out = fn(**kwargs)
        elapsed = (time.perf_counter() - started) * 1000.0
        self._trace.append(ToolCall(name, summary_in, summary_out, elapsed))
        return result

    def reset_trace(self) -> None:
        self._trace = []

    def take_trace(self) -> list[ToolCall]:
        trace, self._trace = self._trace, []
        return trace

    # -- batch scoring -----------------------------------------------

    def prime(self, events: Iterable[RiskEvent]) -> int:
        """Score a whole sweep's worth of events in one pass per surface.

        This started as a speed fix — the feature pipeline is pandas, and
        running it over a one-row frame per event costs about 70ms of pure
        framework overhead, which is roughly two minutes across a full sweep
        and almost none of it arithmetic. Batching takes that to under a
        millisecond an event.

        It turned out to matter for a second reason. Scoring the sweep in one
        pass means every event is priced against exactly the same model state.
        Per-event scoring leaves open the possibility of two events in one run
        being priced by different weights — a reload, a hot-swapped artifact —
        and a run where the pricing basis shifted halfway through is a run
        whose totals cannot be reconciled.

        Call this *after* any adjustment to the events, not before. The
        contact-history fields the agent reconciles against the audit ledger
        are model inputs, so priming stale events would cache probabilities
        that the reconciled event no longer justifies.
        """
        by_surface: dict[str, list[RiskEvent]] = defaultdict(list)
        for event in events:
            by_surface[event.event_type].append(event)

        cached = 0
        for surface, group in by_surface.items():
            frame = pd.DataFrame([dataio.event_to_feature_row(e) for e in group])
            distributions = self.root_cause[surface].predict_distribution(frame)
            probabilities = self.recovery[surface].predict_all(frame)
            for i, event in enumerate(group):
                dist = {k: float(v) for k, v in distributions[i].items()}
                top = max(dist, key=dist.get)
                self._primed[event.event_id] = (
                    (top, dist[top], dist),
                    {k: float(v[i]) for k, v in probabilities.items()},
                )
                cached += 1
        return cached

    def clear_primed(self) -> None:
        self._primed = {}

    @property
    def primed_count(self) -> int:
        return len(self._primed)

    # -- tools -------------------------------------------------------
    #
    # Each returns (input_summary, result, output_summary). The summaries are
    # what lands in the audit trace; the result is what the caller uses.

    def _classify_root_cause(self, event: RiskEvent) -> tuple[dict, tuple, dict]:
        """Why did this event happen? Returns (cause, confidence, distribution)."""
        primed = self._primed.get(event.event_id)
        if primed is not None:
            cause, confidence, distribution = primed[0]
        else:
            row = dataio.event_to_feature_row(event)
            cause, confidence, distribution = self.root_cause[event.event_type].predict_one(row)
        return (
            {"event_id": event.event_id, "surface": event.event_type,
             "scored_in_batch": primed is not None},
            (cause, confidence, distribution),
            {
                "root_cause": cause,
                "confidence": round(float(confidence), 4),
                # Only the runners-up, so the trace stays legible. The full
                # distribution is recorded once on the decision itself.
                "next_most_likely": [
                    {"cause": k, "p": round(float(v), 4)}
                    for k, v in sorted(distribution.items(), key=lambda kv: -kv[1])[1:3]
                ],
            },
        )

    def _generate_candidates(self, event: RiskEvent) -> tuple[dict, list, dict]:
        """Which actions are even available on this surface?"""
        candidates = build_candidates(event, self.cfg)
        return (
            {"event_id": event.event_id, "surface": event.event_type},
            candidates,
            {"count": len(candidates),
             "variants": [candidate_action_key(c) for c in candidates]},
        )

    def _estimate_recovery(self, event: RiskEvent) -> tuple[dict, dict, dict]:
        """What is P(recovered) under each action variant, including inaction?"""
        primed = self._primed.get(event.event_id)
        if primed is not None:
            probs = primed[1]
        else:
            row = dataio.event_to_feature_row(event)
            probs = self.recovery[event.event_type].predict_one_all(row)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return (
            {"event_id": event.event_id, "surface": event.event_type,
             "scored_in_batch": primed is not None},
            probs,
            {"n_variants": len(probs),
             "baseline_do_nothing": round(float(probs.get("do_nothing", 0.0)), 4),
             "highest": [{"variant": k, "p": round(float(v), 4)} for k, v in ranked[:3]]},
        )

    def _score_actions(self, event: RiskEvent, probabilities: Mapping[str, float],
                       p_fraud: float,
                       candidates: Optional[list[CandidateAction]] = None
                       ) -> tuple[dict, list[ScoredAction], dict]:
        """Convert probabilities into expected net recovery, in rupees."""
        ranked = self.economics.rank(event, probabilities, p_fraud, candidates)
        return (
            {"event_id": event.event_id, "p_fraud": round(float(p_fraud), 4),
             "n_candidates": len(ranked)},
            ranked,
            {"best": ranked[0].candidate.action if ranked else None,
             "best_enr_inr": round(ranked[0].expected_net_recovery_inr, 2) if ranked else None,
             "top3": [{"action": s.candidate.action,
                       "enr_inr": round(s.expected_net_recovery_inr, 2)}
                      for s in ranked[:3]]},
        )

    def _apply_guardrails(self, event: RiskEvent, ranked: list[ScoredAction],
                          context: GuardrailContext) -> tuple[dict, Verdict, dict]:
        """Screen every option against policy and pick the best permitted one."""
        verdict = self.guardrails.select(event, ranked, context)
        blocked = [s for s in verdict.considered if s.blocked_by]
        return (
            {"event_id": event.event_id, "n_considered": len(ranked),
             "root_cause": context.root_cause},
            verdict,
            {"chosen": verdict.chosen.candidate.action,
             "n_blocked": len(blocked),
             "requires_human_approval": verdict.requires_human_approval,
             "guardrails_fired": len(verdict.guardrails_applied)},
        )

    # -- the plan ----------------------------------------------------

    def run_plan(self, event: RiskEvent, budget: Optional[SweepBudget] = None) -> PlanResult:
        """Execute the fixed five-step plan for one event.

        No branches. Every event goes through every step, including the ones
        that will obviously end in `do_nothing`, because a decision to do
        nothing that skipped the economics is not a decision — it is an
        omission, and it would leave the audit trail unable to show that the
        alternatives were priced and found wanting.
        """
        self.reset_trace()
        budget = budget if budget is not None else SweepBudget()

        cause, confidence, distribution = self.call("classify_root_cause", event=event)
        candidates = self.call("generate_candidates", event=event)
        probabilities = self.call("estimate_recovery", event=event)

        p_fraud = float(distribution.get("fraud_suspected", 0.0))
        ranked = self.call("score_actions", event=event, probabilities=probabilities,
                           p_fraud=p_fraud, candidates=candidates)

        context = GuardrailContext(
            root_cause=cause,
            root_cause_confidence=float(confidence),
            root_cause_distribution=dict(distribution),
            budget=budget,
        )
        verdict = self.call("apply_guardrails", event=event, ranked=ranked, context=context)

        decision = Decision(
            event_id=event.event_id,
            event_type=event.event_type,
            amount_inr=event.amount_inr,
            customer_id=event.customer.customer_id,
            root_cause=cause,
            root_cause_confidence=float(confidence),
            root_cause_distribution={k: float(v) for k, v in distribution.items()},
            chosen=verdict.chosen,
            considered=verdict.considered,
            requires_human_approval=verdict.requires_human_approval,
            approval_reason=verdict.approval_reason,
            guardrails_applied=verdict.guardrails_applied,
            rejected_reasons=verdict.rejected_reasons,
        )
        return PlanResult(decision=decision, verdict=verdict, trace=self.take_trace())


def arithmetic_for(decision: Decision) -> str:
    """Term-by-term breakdown of the chosen action, for the audit record."""
    return explain(decision.chosen)
