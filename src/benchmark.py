"""
Does the agent actually recover more money than the obvious alternatives?

This is the question improvement #2 exists to answer, and it is the only
question in the project where being wrong is embarrassing rather than merely
suboptimal. A recovery agent that reports 81% classification accuracy has said
nothing about whether a merchant is better off running it. So this module
throws away accuracy entirely and scores every policy in rupees, against the
same held-out events, using the simulator's counterfactual outcomes.

How the money is counted
------------------------

The synthetic data carries, for every event, the outcome under *every* action —
the `po_*` columns. That is oracle knowledge, available only because the data
is synthetic, and it is the reason an offline comparison is possible at all: for
each event we can ask what would have happened under the action each policy
chose, not just under the one that was logged. `src/dataio.py` keeps those
columns quarantined behind `extract_outcomes` so nothing on the decision path
can reach them.

Realised value for a chosen action is

    recovered x amount x value_basis            (margin on sales, face on AR)
      - discount given, but only if it converted
      - chargeback at 1.5x, but only if the event was truly fraudulent
        *and* the attempt succeeded
      - the action's own cost

and every policy is reported as the *difference* from doing nothing on the same
event. That mirrors how the agent scores itself, so the benchmark measures the
same quantity the agent optimises rather than a rearranged version of it.

Two honest conventions, both of which cost the agent money
----------------------------------------------------------

`request_human_review` has no counterfactual column, because "what happens when
a person looks at it" is not something the simulator models. It is scored as
the do-nothing outcome minus the review cost — that is, the reviewer is assumed
to recover nothing. That is certainly pessimistic; a real analyst would save
some of those cases. It is chosen deliberately, because the alternative is to
invent a recovery rate for human beings and then quote a total that depends on
it.

`stop_and_flag_fraud` needs no convention. It is scored as no attempt made,
which means it forgoes whatever do-nothing would have earned and avoids
whatever do-nothing would have lost to a chargeback. Both halves come from
ground truth in the data, so the case for stopping a fraudulent payment is
made by the arithmetic and not by an assumption.

What the baselines are for
--------------------------

Beating "do nothing" is not an achievement — every intervention beats it on
gross recovery, which is exactly why gross recovery is the wrong metric. The
baselines that matter are the ones a merchant would plausibly run instead:
retry everything, contact everyone, discount everyone, or a page of
if-this-then-that rules. `rules_true_cause` is deliberately given the
*ground-truth* root cause, an advantage the agent does not get, so that nobody
can attribute the agent's margin to a weak straw man. `best_fixed_action` is
chosen with hindsight over the same test set. `oracle_per_event` picks the
best action per event knowing the answer, and is the ceiling no policy can
pass.

`agent_unattended` is the number to quote if you want the conservative one: it
treats every decision the guardrails routed to a human as if nothing happened.
`agent_no_guardrails` prices the safety layer — the gap between it and `agent`
is what the merchant pays for consent, frequency caps, retry discipline and
sign-off thresholds.

Confidence intervals
--------------------

Each policy's per-event scores are paired, because every policy is evaluated on
the same events. So the interval is computed on the *paired difference* by
bootstrap resampling event indices (2,000 replicates, percentile method), which
respects that pairing. An interval on each policy's total separately would be
much wider and would not answer the question anyone is asking, which is whether
the difference is real.

Run it with `python -m src.benchmark`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from . import dataio
from .economics import (
    Economics, P_RECOVER_STOP_AND_FLAG, build_candidates, candidate_action_key,
)
from .guardrails import GuardrailContext, Guardrails, SweepBudget
from .ml.root_cause import RootCauseBundle
from .ml.uplift import RecoveryBundle
from .schemas import (
    ACTION_CHANNEL, AUTOMATED_REMINDER, AUTOMATED_REMINDER_WITH_PLAN,
    CHECKOUT_ABANDONMENT, DELAYED_RETRY, DO_NOTHING, ESCALATE_TO_COLLECTIONS,
    EVENT_TYPES, IMMEDIATE_RETRY, OFFER_BOUNDED_DISCOUNT, OUTREACH_ACTIONS,
    OVERDUE_RECEIVABLE, PAYMENT_FAILURE, PROMPT_NEW_PAYMENT_METHOD,
    REQUEST_HUMAN_REVIEW, SEND_REMINDER_EMAIL, SEND_REMINDER_WHATSAPP,
    STOP_AND_FLAG_FRAUD, CandidateAction, RiskEvent, ScoredAction,
)
from .tools import Toolbelt

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260826


def thawed(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """A mutable deep copy of the frozen policy config.

    `src/config.py` freezes the config on purpose, so a benchmark variant that
    needs to change a weight has to make its own copy rather than reach into
    the shared one. Used by `agent_transactional`, which sets the LTV weight to
    zero to isolate the objective mismatch described below.
    """
    if isinstance(cfg, Mapping):
        return {k: thawed(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [thawed(v) for v in cfg]
    return cfg


# ---------------------------------------------------------------------
# What a policy decides
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """One policy's decision for one event."""
    action: str
    discount_pct: float = 0.0
    delay_hours: int = 0
    gated: bool = False          # would have waited for a human

    @property
    def action_key(self) -> str:
        return candidate_action_key(
            CandidateAction(self.action, discount_pct=self.discount_pct,
                            delay_hours=self.delay_hours)
        )


NOTHING = Choice(DO_NOTHING)


# ---------------------------------------------------------------------
# Realised money
# ---------------------------------------------------------------------

@dataclass
class Realised:
    recovered: bool
    gross_inr: float
    discount_inr: float
    chargeback_inr: float
    action_cost_inr: float

    @property
    def net_inr(self) -> float:
        return self.gross_inr - self.discount_inr - self.chargeback_inr - self.action_cost_inr


class Scorer:
    """Turns (event, choice) into realised rupees, using oracle outcomes."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None):
        self.cfg = cfg or C.load_config()
        self.economics = Economics(self.cfg)
        self.chargeback_multiplier = float(
            self.cfg["economics"]["chargeback_cost_multiplier"]
        )
        self.outcomes: dict[str, dict[str, bool]] = {}
        self.fraud: dict[str, bool] = {}
        self.true_cause: dict[str, str] = {}
        for surface in EVENT_TYPES:
            frame = dataio.load_surface_df(surface)
            extracted = dataio.extract_outcomes(surface, frame)
            for eid, sim in extracted.items():
                self.outcomes[eid] = dict(sim.outcomes)
            self.fraud.update(dataio.fraud_flags(surface))
            id_col = dataio.SURFACE_SPEC[surface]["id_col"]
            if "true_root_cause" in frame.columns:
                self.true_cause.update({
                    str(r[id_col]): str(r["true_root_cause"])
                    for r in frame.to_dict("records")
                })

    # -- outcome lookup ----------------------------------------------

    def available_keys(self, event_id: str) -> list[str]:
        return sorted(self.outcomes.get(event_id, {}))

    def would_recover(self, event: RiskEvent, choice: Choice) -> Optional[bool]:
        """True/False from the oracle, or None where no counterfactual exists."""
        if choice.action == STOP_AND_FLAG_FRAUD:
            # No attempt is made, so there is nothing to recover. This is a
            # fact about the action, not a missing column.
            return False
        if choice.action == REQUEST_HUMAN_REVIEW:
            return None
        table = self.outcomes.get(event.event_id, {})
        if choice.action_key in table:
            return bool(table[choice.action_key])
        return None

    def realised(self, event: RiskEvent, choice: Choice) -> Realised:
        amount = float(event.amount_inr)
        basis = self.economics.margin_fraction(event)
        cost = self.economics.action_cost(choice.action)

        recovered = self.would_recover(event, choice)
        if recovered is None:
            # The two actions with no counterfactual. Both fall back to the
            # do-nothing outcome and keep their own cost, which is the
            # pessimistic reading in both cases.
            recovered = bool(self.outcomes.get(event.event_id, {}).get(DO_NOTHING, False))

        gross = amount * basis if recovered else 0.0
        discount = (amount * choice.discount_pct / 100.0) if (recovered and choice.discount_pct) else 0.0
        chargeback = (
            amount * self.chargeback_multiplier
            if (recovered and self.fraud.get(event.event_id, False))
            else 0.0
        )
        return Realised(bool(recovered), gross, discount, chargeback, cost)

    def net_vs_nothing(self, event: RiskEvent, choice: Choice) -> float:
        """Realised net, minus what doing nothing on the same event realises.

        Expressed as a difference for the same reason the agent's own scoring
        is: it makes zero the honest reference point, and it means a policy
        that intervenes and gains nothing scores its costs as a loss rather
        than disappearing into a large gross number.
        """
        return self.realised(event, choice).net_inr - self.realised(event, NOTHING).net_inr


# ---------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------

class Policy:
    """A rule for turning events into choices. Subclasses override `choose`."""

    name = "policy"
    description = ""
    oracle_advantage = ""     # non-empty if the policy sees data the agent cannot

    def prepare(self, events: Sequence[RiskEvent]) -> None:
        """Called once before the sweep, in the order events will be seen."""

    def choose(self, event: RiskEvent) -> Choice:
        raise NotImplementedError


class DoNothingPolicy(Policy):
    name = "do_nothing"
    description = "never intervene — the reference point, scores exactly 0 by construction"

    def choose(self, event: RiskEvent) -> Choice:
        return NOTHING


class FixedPolicy(Policy):
    """The same action on every event of a surface. The naive playbooks."""

    def __init__(self, name: str, description: str,
                 per_surface: Mapping[str, Choice]):
        self.name = name
        self.description = description
        self.per_surface = dict(per_surface)

    def choose(self, event: RiskEvent) -> Choice:
        return self.per_surface.get(event.event_type, NOTHING)


CAUSE_RULES: dict[str, dict[str, Choice]] = {
    PAYMENT_FAILURE: {
        "technical_bank_side": Choice(IMMEDIATE_RETRY),
        "insufficient_funds": Choice(DELAYED_RETRY, delay_hours=48),
        "expired_card": Choice(PROMPT_NEW_PAYMENT_METHOD),
        "invalid_details": Choice(PROMPT_NEW_PAYMENT_METHOD),
        "fraud_suspected": Choice(STOP_AND_FLAG_FRAUD),
    },
    CHECKOUT_ABANDONMENT: {
        "technical_friction": Choice(SEND_REMINDER_EMAIL),
        "price_sensitivity": Choice(OFFER_BOUNDED_DISCOUNT, discount_pct=10.0),
        "high_cart_hesitation": Choice(OFFER_BOUNDED_DISCOUNT, discount_pct=5.0),
        "comparison_shopping": Choice(SEND_REMINDER_WHATSAPP),
    },
    OVERDUE_RECEIVABLE: {
        "cash_flow_issue": Choice(AUTOMATED_REMINDER_WITH_PLAN),
        "chronic_late_payer": Choice(ESCALATE_TO_COLLECTIONS),
        "dispute_pending": NOTHING,
        "invoice_error": NOTHING,
    },
}


class CauseRulesPolicy(Policy):
    """A page of if-this-then-that rules over the root cause.

    This is what a competent team writes in an afternoon, and it is the
    baseline the agent has to beat to justify existing. Two variants: one fed
    the ground-truth cause, one fed the classifier's prediction. The gap
    between them is the cost of imperfect classification; the gap between the
    better of them and the agent is what the economics layer adds.
    """

    def __init__(self, name: str, description: str,
                 cause_of: Callable[[RiskEvent], str],
                 oracle_advantage: str = ""):
        self.name = name
        self.description = description
        self.oracle_advantage = oracle_advantage
        self._cause_of = cause_of

    def choose(self, event: RiskEvent) -> Choice:
        cause = self._cause_of(event)
        return CAUSE_RULES[event.event_type].get(cause, NOTHING)


class BestFixedActionPolicy(Policy):
    """The single best fixed action per surface, chosen with hindsight.

    Fitted on the same events it is scored on, which is cheating in the
    baseline's favour. That is the point — it is meant to be hard to beat.
    """

    name = "best_fixed_action"
    description = "one action per surface, picked with hindsight on this very test set"
    oracle_advantage = "chooses its action knowing the test-set outcomes"

    def __init__(self, scorer: Scorer):
        self.scorer = scorer
        self._best: dict[str, Choice] = {}

    def prepare(self, events: Sequence[RiskEvent]) -> None:
        self._best = {}
        for surface in EVENT_TYPES:
            subset = [e for e in events if e.event_type == surface]
            if not subset:
                continue
            best, best_total = NOTHING, 0.0
            for choice in _surface_choices(surface):
                total = sum(self.scorer.net_vs_nothing(e, choice) for e in subset)
                if total > best_total:
                    best, best_total = choice, total
            self._best[surface] = best

    def choose(self, event: RiskEvent) -> Choice:
        return self._best.get(event.event_type, NOTHING)


class OraclePolicy(Policy):
    """Best action per event, knowing the outcome. The ceiling."""

    name = "oracle_per_event"
    description = "per-event best action with full hindsight — an upper bound, not a policy"
    oracle_advantage = "knows every counterfactual outcome for every event"

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def choose(self, event: RiskEvent) -> Choice:
        best, best_net = NOTHING, 0.0
        for choice in _surface_choices(event.event_type):
            net = self.scorer.net_vs_nothing(event, choice)
            if net > best_net:
                best, best_net = choice, net
        return best


class AgentPolicy(Policy):
    """The real thing: models, economics, guardrails, sweep budgets.

    `mode` selects which part is under test:

      * ``full``          — as shipped, gated decisions counted as chosen
      * ``unattended``    — gated decisions counted as inaction, so the score
                            is what the agent achieves with no human help
      * ``no_guardrails`` — unconstrained argmax of expected net recovery, to
                            price the safety layer
    """

    def __init__(self, toolbelt: Toolbelt, mode: str = "full"):
        if mode not in {"full", "unattended", "no_guardrails", "transactional"}:
            raise ValueError(f"unknown agent mode {mode!r}")
        self.toolbelt = toolbelt
        self.mode = mode
        self.name = {"full": "agent", "unattended": "agent_unattended",
                     "no_guardrails": "agent_no_guardrails",
                     "transactional": "agent_transactional"}[mode]
        self.description = {
            "full": "expected-net-recovery ranking with every guardrail and budget applied",
            "unattended": "same, but decisions routed to a human count as inaction",
            "no_guardrails": "expected-net-recovery argmax with policy and budgets removed",
            "transactional": "as shipped, but with the LTV retention weight set to zero",
        }[mode]
        self.oracle_advantage = ""
        self.budget: Optional[SweepBudget] = None
        self.gated_count = 0

    def prepare(self, events: Sequence[RiskEvent]) -> None:
        self.budget = SweepBudget(
            at_risk_total_inr=sum(e.amount_inr for e in events)
        )
        self.gated_count = 0
        # Priming is idempotent and shared between agent variants, so a
        # toolbelt that already holds this sweep is left alone.
        if self.toolbelt.primed_count < len(events):
            self.toolbelt.prime(events)

    def choose(self, event: RiskEvent) -> Choice:
        if self.mode == "no_guardrails":
            return self._choose_unconstrained(event)

        result = self.toolbelt.run_plan(event, self.budget)
        decision = result.decision
        self.budget.record(event, decision.chosen)
        candidate = decision.chosen.candidate
        if decision.requires_human_approval:
            self.gated_count += 1
            if self.mode == "unattended":
                return Choice(DO_NOTHING, gated=True)
        return Choice(candidate.action, candidate.discount_pct,
                      candidate.delay_hours, decision.requires_human_approval)

    def _choose_unconstrained(self, event: RiskEvent) -> Choice:
        cause, _confidence, distribution = self.toolbelt.call(
            "classify_root_cause", event=event
        )
        candidates = self.toolbelt.call("generate_candidates", event=event)
        probabilities = self.toolbelt.call("estimate_recovery", event=event)
        ranked = self.toolbelt.call(
            "score_actions", event=event, probabilities=probabilities,
            p_fraud=float(distribution.get("fraud_suspected", 0.0)),
            candidates=candidates,
        )
        self.toolbelt.reset_trace()
        if not ranked:
            return NOTHING
        top = ranked[0].candidate
        return Choice(top.action, top.discount_pct, top.delay_hours)


def _surface_choices(surface: str) -> list[Choice]:
    """Every distinct choice available on a surface, including variants."""
    out: list[Choice] = []
    for action in _SURFACE_ACTIONS[surface]:
        if action == DELAYED_RETRY:
            out.extend([Choice(action, delay_hours=12), Choice(action, delay_hours=48)])
        elif action == OFFER_BOUNDED_DISCOUNT:
            out.extend([Choice(action, discount_pct=5.0), Choice(action, discount_pct=10.0)])
        else:
            out.append(Choice(action))
    return out


_SURFACE_ACTIONS: dict[str, tuple[str, ...]] = {
    PAYMENT_FAILURE: (DO_NOTHING, IMMEDIATE_RETRY, DELAYED_RETRY,
                      PROMPT_NEW_PAYMENT_METHOD, STOP_AND_FLAG_FRAUD),
    CHECKOUT_ABANDONMENT: (DO_NOTHING, SEND_REMINDER_EMAIL, SEND_REMINDER_WHATSAPP,
                           OFFER_BOUNDED_DISCOUNT),
    OVERDUE_RECEIVABLE: (DO_NOTHING, AUTOMATED_REMINDER, AUTOMATED_REMINDER_WITH_PLAN,
                         ESCALATE_TO_COLLECTIONS),
}


# ---------------------------------------------------------------------
# Would the policy engine allow it?
# ---------------------------------------------------------------------

class Screener:
    """Runs any policy's choice past the shipped guardrails.

    This exists because of the first result this benchmark produced, which was
    that several naive baselines recover more money than the agent. That is
    true, and it stays in the report. The reason is not that the economics are
    wrong — it is that the baselines message customers who withheld consent,
    who are on the DND register, who were already contacted twice this week, or
    at eleven at night. Those recoveries are not available to anyone actually
    operating in India, and a benchmark that scored them as revenue would be
    measuring the wrong thing.

    So every policy is also scored on how many of its choices the policy engine
    would refuse, and the baselines that can be made lawful get a `_compliant`
    twin that is run through the same screen the agent is. That twin is the
    honest comparison, and it is the one to read first.
    """

    def __init__(self, toolbelt: Toolbelt):
        self.toolbelt = toolbelt
        self.guardrails = toolbelt.guardrails
        self.economics = toolbelt.economics

    def evaluate(self, event: RiskEvent, choice: Choice, budget: SweepBudget
                 ) -> tuple[Optional[ScoredAction], list[str]]:
        """Returns the priced action and the guardrails that would refuse it."""
        if choice.action in (DO_NOTHING, REQUEST_HUMAN_REVIEW):
            return None, []
        primed = self.toolbelt._primed.get(event.event_id)
        if primed is None:
            raise RuntimeError(
                f"screener needs primed model output for {event.event_id}; "
                f"call Toolbelt.prime on the full event list first"
            )
        (cause, confidence, distribution), probabilities = primed
        p_fraud = float(distribution.get("fraud_suspected", 0.0))
        p_baseline = float(probabilities.get(DO_NOTHING, 0.0))

        candidate = CandidateAction(
            choice.action, discount_pct=choice.discount_pct,
            delay_hours=choice.delay_hours,
            channel=ACTION_CHANNEL.get(choice.action),
        )
        if choice.action == STOP_AND_FLAG_FRAUD:
            scored = self.economics.score(
                event, candidate, P_RECOVER_STOP_AND_FLAG, p_baseline, p_fraud,
                probability_is_assumed=True,
            )
        else:
            key = candidate_action_key(candidate)
            if key not in probabilities:
                raise KeyError(f"no probability for {key!r} on {event.event_id}")
            scored = self.economics.score(
                event, candidate, float(probabilities[key]), p_baseline, p_fraud,
            )
        ctx = GuardrailContext(root_cause=cause, root_cause_confidence=float(confidence),
                               root_cause_distribution=dict(distribution), budget=budget)
        self.guardrails.screen(event, scored, ctx)
        return scored, list(scored.blocked_by)


class GuardedPolicy(Policy):
    """Any policy, forced through the shipped policy engine.

    When a guardrail refuses the wrapped policy's choice, this falls back to
    doing nothing rather than searching for a permitted substitute. That is
    what a rules table plus a compliance layer actually does in practice: the
    rule fires, the compliance layer says no, and the event is left alone.
    Searching for the best permitted alternative is precisely the thing the
    agent does, so letting the baseline do it too would quietly turn the
    baseline into the agent.
    """

    def __init__(self, inner: Policy, screener: Screener):
        self.inner = inner
        self.screener = screener
        self.name = f"{inner.name}_compliant"
        self.description = f"{inner.description}, forced through the policy engine"
        self.oracle_advantage = inner.oracle_advantage
        self.budget: Optional[SweepBudget] = None
        self.refused = 0

    def prepare(self, events: Sequence[RiskEvent]) -> None:
        self.inner.prepare(events)
        self.budget = SweepBudget(at_risk_total_inr=sum(e.amount_inr for e in events))
        self.refused = 0

    def choose(self, event: RiskEvent) -> Choice:
        choice = self.inner.choose(event)
        scored, reasons = self.screener.evaluate(event, choice, self.budget)
        if reasons:
            self.refused += 1
            return NOTHING
        if scored is not None:
            self.budget.record(event, scored)
            # Quiet hours defer rather than refuse, so carry the deferral
            # through — the contact still happens, just later.
            return Choice(choice.action, choice.discount_pct,
                          scored.candidate.delay_hours, choice.gated)
        return choice


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@dataclass
class PolicyResult:
    name: str
    description: str
    oracle_advantage: str
    per_event_net: np.ndarray
    choices: list[Choice] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    contacts: int = 0
    discount_spend_inr: float = 0.0
    chargeback_inr: float = 0.0
    action_spend_inr: float = 0.0
    recovered_events: int = 0
    gated: int = 0
    per_surface_net: dict[str, float] = field(default_factory=dict)
    would_be_refused: int = 0
    refusal_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def total_net_inr(self) -> float:
        return float(self.per_event_net.sum())

    @property
    def mean_net_inr(self) -> float:
        return float(self.per_event_net.mean()) if len(self.per_event_net) else 0.0

    @property
    def inr_per_contact(self) -> Optional[float]:
        """Recovery per message sent. The cost of a contact is not just money.

        A policy that recovers 20% more by sending three times as many messages
        is not obviously the better policy, and this is the column that makes
        that visible.
        """
        return self.total_net_inr / self.contacts if self.contacts else None

    @property
    def inr_per_discount_rupee(self) -> Optional[float]:
        return (self.total_net_inr / self.discount_spend_inr
                if self.discount_spend_inr else None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "oracle_advantage": self.oracle_advantage,
            "total_net_inr": round(self.total_net_inr, 2),
            "mean_net_inr": round(self.mean_net_inr, 2),
            "events": int(len(self.per_event_net)),
            "recovered_events": self.recovered_events,
            "contacts": self.contacts,
            "discount_spend_inr": round(self.discount_spend_inr, 2),
            "chargeback_inr": round(self.chargeback_inr, 2),
            "action_spend_inr": round(self.action_spend_inr, 2),
            "gated_for_human": self.gated,
            "would_be_refused": self.would_be_refused,
            "refusal_reasons": dict(sorted(self.refusal_reasons.items(),
                                           key=lambda kv: -kv[1])),
            "inr_per_contact": (round(self.inr_per_contact, 2)
                                if self.inr_per_contact is not None else None),
            "inr_per_discount_rupee": (round(self.inr_per_discount_rupee, 3)
                                       if self.inr_per_discount_rupee is not None else None),
            "action_counts": dict(sorted(self.action_counts.items(), key=lambda kv: -kv[1])),
            "per_surface_net_inr": {k: round(v, 2) for k, v in self.per_surface_net.items()},
        }


def evaluate_policy(policy: Policy, events: Sequence[RiskEvent],
                    scorer: Scorer, screener: Optional[Screener] = None) -> PolicyResult:
    policy.prepare(events)
    nets = np.zeros(len(events), dtype=float)
    result = PolicyResult(policy.name, policy.description, policy.oracle_advantage, nets)

    # A separate budget purely for the compliance audit, so that "would this
    # have been refused" reflects the policy's own consumption of the sweep
    # ceilings rather than the agent's.
    audit_budget = SweepBudget(at_risk_total_inr=sum(e.amount_inr for e in events))

    for i, event in enumerate(events):
        choice = policy.choose(event)
        realised = scorer.realised(event, choice)
        baseline = scorer.realised(event, NOTHING)
        nets[i] = realised.net_inr - baseline.net_inr

        if screener is not None:
            scored, reasons = screener.evaluate(event, choice, audit_budget)
            if reasons:
                result.would_be_refused += 1
                for reason in reasons:
                    key = reason.split(" — ")[0].split(":")[0].strip()
                    result.refusal_reasons[key] = result.refusal_reasons.get(key, 0) + 1
            if scored is not None:
                audit_budget.record(event, scored)

        result.choices.append(choice)
        result.action_counts[choice.action] = result.action_counts.get(choice.action, 0) + 1
        if choice.action in OUTREACH_ACTIONS:
            result.contacts += 1
        result.discount_spend_inr += realised.discount_inr
        result.chargeback_inr += realised.chargeback_inr
        result.action_spend_inr += realised.action_cost_inr
        result.recovered_events += int(realised.recovered)
        result.gated += int(choice.gated)
        result.per_surface_net[event.event_type] = (
            result.per_surface_net.get(event.event_type, 0.0) + nets[i]
        )
    result.per_event_net = nets
    return result


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

@dataclass
class Interval:
    point: float
    low: float
    high: float
    replicates: int

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0.0) or (self.high < 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"point": round(self.point, 2), "ci95_low": round(self.low, 2),
                "ci95_high": round(self.high, 2), "replicates": self.replicates,
                "excludes_zero": self.excludes_zero}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *,
                     replicates: int = BOOTSTRAP_REPLICATES,
                     seed: int = BOOTSTRAP_SEED,
                     scale: float = 1.0) -> Interval:
    """95% percentile interval on the mean of (a - b), resampling events.

    The same resampled index is applied to both arrays, which is what makes
    this paired. Resampling the two independently would inflate the interval
    with variance that the shared event set has already removed.

    `scale` multiplies the result, so passing n gives an interval on the total
    rather than the per-event mean. That is a linear transform of the same
    resamples, not a separate estimate.
    """
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal-length arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    idx = rng.integers(0, n, size=(replicates, n))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Interval(float(diff.mean()) * scale, float(low) * scale,
                    float(high) * scale, replicates)


# ---------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------

REFERENCE_POLICY = "agent"

def build_policies(scorer: Scorer, toolbelt: Toolbelt, screener: Screener) -> list[Policy]:
    """Every policy under test, in the order they are reported."""
    aggressive = FixedPolicy(
        "retry_or_chase_everything",
        "the most aggressive action available on each surface, every time",
        {PAYMENT_FAILURE: Choice(IMMEDIATE_RETRY),
         CHECKOUT_ABANDONMENT: Choice(SEND_REMINDER_WHATSAPP),
         OVERDUE_RECEIVABLE: Choice(ESCALATE_TO_COLLECTIONS)},
    )
    drip = FixedPolicy(
        "contact_everyone",
        "one retry, one email, one reminder — the default drip campaign",
        {PAYMENT_FAILURE: Choice(DELAYED_RETRY, delay_hours=12),
         CHECKOUT_ABANDONMENT: Choice(SEND_REMINDER_EMAIL),
         OVERDUE_RECEIVABLE: Choice(AUTOMATED_REMINDER)},
    )
    discounting = FixedPolicy(
        "discount_everyone",
        "maximum permitted discount on every cart; most aggressive elsewhere",
        {PAYMENT_FAILURE: Choice(PROMPT_NEW_PAYMENT_METHOD),
         CHECKOUT_ABANDONMENT: Choice(OFFER_BOUNDED_DISCOUNT, discount_pct=10.0),
         OVERDUE_RECEIVABLE: Choice(AUTOMATED_REMINDER_WITH_PLAN)},
    )
    rules_predicted = CauseRulesPolicy(
        "rules_predicted_cause",
        "if-this-then-that rules over the classifier's predicted root cause",
        cause_of=_predicted_cause_fn(toolbelt),
    )
    rules_true = CauseRulesPolicy(
        "rules_true_cause",
        "the same rules, handed the ground-truth root cause",
        cause_of=lambda e: scorer.true_cause.get(e.event_id, ""),
        oracle_advantage="reads the true root cause, which the agent must infer",
    )

    # A toolbelt that shares the loaded models and the primed scores, but
    # prices with the LTV retention term switched off. Isolates how much of
    # the agent's apparent shortfall on this benchmark is the agent optimising
    # something the benchmark cannot see.
    transactional_cfg = thawed(toolbelt.cfg)
    transactional_cfg["economics"]["ltv_retention_weight"] = 0.0
    transactional_belt = Toolbelt(transactional_cfg,
                                  root_cause=toolbelt.root_cause,
                                  recovery=toolbelt.recovery)
    transactional_belt._primed = toolbelt._primed

    return [
        DoNothingPolicy(),
        aggressive,
        GuardedPolicy(aggressive, screener),
        drip,
        GuardedPolicy(drip, screener),
        discounting,
        GuardedPolicy(discounting, screener),
        rules_predicted,
        GuardedPolicy(rules_predicted, screener),
        rules_true,
        GuardedPolicy(rules_true, screener),
        BestFixedActionPolicy(scorer),
        AgentPolicy(toolbelt, "no_guardrails"),
        AgentPolicy(transactional_belt, "transactional"),
        AgentPolicy(toolbelt, "unattended"),
        AgentPolicy(toolbelt, "full"),
        OraclePolicy(scorer),
    ]


LAWFUL_POLICIES = frozenset({
    "do_nothing", "agent", "agent_unattended", "agent_transactional",
    "retry_or_chase_everything_compliant", "contact_everyone_compliant",
    "discount_everyone_compliant", "rules_predicted_cause_compliant",
    "rules_true_cause_compliant",
})


def _predicted_cause_fn(toolbelt: Toolbelt) -> Callable[[RiskEvent], str]:
    def predicted(event: RiskEvent) -> str:
        primed = toolbelt._primed.get(event.event_id)
        if primed is not None:
            return primed[0][0]
        row = dataio.event_to_feature_row(event)
        return toolbelt.root_cause[event.event_type].predict_one(row)[0]
    return predicted


def run_benchmark(*, split: Optional[str] = "test",
                  limit_per_surface: Optional[int] = None,
                  replicates: int = BOOTSTRAP_REPLICATES,
                  reference: str = REFERENCE_POLICY) -> dict[str, Any]:
    scorer = Scorer()
    toolbelt = Toolbelt(scorer.cfg)

    events: list[RiskEvent] = []
    for surface in EVENT_TYPES:
        events.extend(dataio.load_events(surface, split, limit_per_surface))
    # Same order the agent uses in production, so the sweep budgets bind on
    # the same events here as they would there.
    events.sort(key=lambda e: e.amount_inr * scorer.economics.margin_fraction(e),
                reverse=True)
    # Priming once up front means the rules_predicted_cause policy and the
    # agent policies all read identical model output.
    toolbelt.clear_primed()
    toolbelt.prime(events)

    screener = Screener(toolbelt)
    results = [evaluate_policy(p, events, scorer, screener)
               for p in build_policies(scorer, toolbelt, screener)]
    by_name = {r.name: r for r in results}
    if reference not in by_name:
        raise KeyError(f"reference policy {reference!r} was not evaluated")
    ref = by_name[reference]

    comparisons = []
    for r in results:
        if r.name == reference:
            continue
        total = paired_bootstrap(ref.per_event_net, r.per_event_net,
                                 replicates=replicates, scale=float(len(events)))
        per_event = paired_bootstrap(ref.per_event_net, r.per_event_net,
                                     replicates=replicates)
        comparisons.append({
            "versus": r.name,
            "reference": reference,
            "lawful": r.name in LAWFUL_POLICIES,
            "uplift_total_inr": total.to_dict(),
            "uplift_per_event_inr": per_event.to_dict(),
            "reference_total_inr": round(ref.total_net_inr, 2),
            "versus_total_inr": round(r.total_net_inr, 2),
            "oracle_advantage": r.oracle_advantage,
        })

    unguarded = by_name.get("agent_no_guardrails")
    price_of_compliance = None
    if unguarded is not None and reference == REFERENCE_POLICY:
        price_of_compliance = {
            "inr": round(unguarded.total_net_inr - ref.total_net_inr, 2),
            "pct_of_value_at_risk": round(
                (unguarded.total_net_inr - ref.total_net_inr)
                / max(sum(e.amount_inr * scorer.economics.margin_fraction(e)
                          for e in events), 1.0) * 100.0, 2),
            "extra_contacts_required": unguarded.contacts - ref.contacts,
            "choices_the_policy_engine_would_refuse": unguarded.would_be_refused,
            "note": "what the shipped policy engine costs in realised recovery. The "
                    "unguarded agent earns more only by making choices the engine "
                    "refuses; this is the price of not contacting people who did not "
                    "agree to be contacted, and it is a cost worth paying.",
        }

    policies = []
    for r in results:
        entry = r.to_dict()
        entry["lawful"] = r.name in LAWFUL_POLICIES
        policies.append(entry)

    return {
        "split": split or "all",
        "events": len(events),
        "gross_at_risk_inr": round(sum(e.amount_inr for e in events), 2),
        "value_at_risk_inr": round(
            sum(e.amount_inr * scorer.economics.margin_fraction(e) for e in events), 2),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "reference_policy": reference,
        "policies": policies,
        "comparisons": comparisons,
        "price_of_compliance": price_of_compliance,
        "conventions": {
            "request_human_review": "scored as the do-nothing outcome minus the "
                                    "review cost; the reviewer recovers nothing",
            "stop_and_flag_fraud": "scored as no attempt made; forgoes do-nothing "
                                   "revenue, avoids do-nothing chargeback exposure",
            "unlawful_baselines": "baselines without a `_compliant` suffix ignore "
                                  "consent, DND, quiet hours and frequency caps; their "
                                  "totals include revenue that is not lawfully "
                                  "available to an Indian merchant",
            "value_basis": dict(scorer.cfg["economics"]["value_basis"]),
            "chargeback_multiplier": scorer.chargeback_multiplier,
        },
        "policy_version": C.policy_version(),
        "code_version": C.CODE_VERSION,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_report(report: Mapping[str, Any]) -> None:
    lawful = [p for p in report["policies"] if p["lawful"]]
    unlawful = [p for p in report["policies"] if not p["lawful"]]

    print()
    print("=" * 104)
    print(f"  UPLIFT BENCHMARK — {report['events']:,} held-out events, "
          f"{report['gross_at_risk_inr']:,.0f} INR at risk")
    print("=" * 104)
    print("  Every figure is realised rupees, incremental against doing nothing on the")
    print("  same event, using the simulator's counterfactual outcomes.")

    _print_table("A. LAWFUL POLICIES — the comparison that means something",
                 "Every choice here survives consent, DND, quiet hours and frequency caps.",
                 lawful)
    _print_table("B. POLICIES THAT IGNORE THE RULES — shown for honesty, not as a target",
                 "These out-earn the agent by contacting people it is not permitted to "
                 "contact.\n  The `refused` column counts how many of their choices the "
                 "shipped policy engine\n  would reject. Their totals are not achievable "
                 "revenue.",
                 unlawful)

    starred = [p for p in report["policies"] if p["oracle_advantage"]]
    if starred:
        print()
        print("  * has an advantage the agent does not:")
        for p in starred:
            print(f"      {p['name']:<30} {p['oracle_advantage']}")

    worst = max(unlawful, key=lambda p: p["would_be_refused"], default=None)
    if worst and worst["refusal_reasons"]:
        print()
        print(f"  WHY THOSE CHOICES ARE REFUSED — most-blocked policy, `{worst['name']}`")
        for reason, count in list(worst["refusal_reasons"].items())[:6]:
            print(f"      {count:>6,}  {reason}")

    print()
    print(f"  UPLIFT OF `{report['reference_policy']}` OVER EACH BASELINE")
    print("  Paired bootstrap, 95% percentile interval, "
          f"{report['bootstrap_replicates']:,} replicates. Read the lawful block first.")
    for label, subset in (("lawful", True), ("ignores the rules", False)):
        rows = [c for c in report["comparisons"] if c["lawful"] is subset]
        if not rows:
            continue
        print()
        print(f"  -- {label} " + "-" * (86 - len(label)))
        print(f"  {'baseline':<30}{'uplift INR':>16}{'95% CI':>34}   verdict")
        for c in rows:
            u = c["uplift_total_inr"]
            ci = f"[{u['ci95_low']:>14,.0f} , {u['ci95_high']:>14,.0f}]"
            if u["excludes_zero"]:
                verdict = "significant" if u["point"] > 0 else "SIGNIFICANTLY WORSE"
            else:
                verdict = "not distinguishable from zero"
            print(f"  {c['versus']:<30}{u['point']:>16,.0f}{ci:>34}   {verdict}")

    poc = report.get("price_of_compliance")
    if poc:
        print()
        print("  THE PRICE OF COMPLIANCE")
        print(f"    The same agent with the policy engine removed earns "
              f"{poc['inr']:,.0f} INR more —")
        print(f"    {poc['pct_of_value_at_risk']:.1f}% of value at risk. It buys that with "
              f"{poc['extra_contacts_required']:,} extra contacts and")
        print(f"    {poc['choices_the_policy_engine_would_refuse']:,} choices the engine "
              f"refuses. That is the cost of not messaging")
        print("    people who did not agree to be messaged, and it is worth paying.")

    print()
    print("  CONVENTIONS THAT COST THE AGENT MONEY")
    print(f"    request_human_review  {report['conventions']['request_human_review']}")
    print(f"    stop_and_flag_fraud   {report['conventions']['stop_and_flag_fraud']}")
    print("=" * 104)


def _print_table(title: str, note: str, policies: Sequence[Mapping[str, Any]]) -> None:
    print()
    print(f"  {title}")
    print(f"  {note}")
    print()
    header = (f"  {'policy':<31}{'total INR':>15}{'per event':>11}{'recovered':>11}"
              f"{'contacts':>10}{'INR/contact':>13}{'discount':>11}{'refused':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in policies:
        mark = "*" if p["oracle_advantage"] else " "
        per_contact = p["inr_per_contact"]
        per_contact_s = f"{per_contact:,.0f}" if per_contact is not None else "—"
        print(f"  {p['name']:<30}{mark}{p['total_net_inr']:>15,.0f}"
              f"{p['mean_net_inr']:>11,.0f}{p['recovered_events']:>11,}"
              f"{p['contacts']:>10,}{per_contact_s:>13}"
              f"{p['discount_spend_inr']:>11,.0f}{p['would_be_refused']:>9,}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Score the agent against baselines in rupees, with confidence intervals.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap events per surface, for a quick check")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--reference", default=REFERENCE_POLICY,
                        help="which policy the uplift is measured for")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also write the full report as JSON")
    args = parser.parse_args(argv)

    report = run_benchmark(
        split=None if args.split == "all" else args.split,
        limit_per_surface=args.limit,
        replicates=args.replicates,
        reference=args.reference,
    )
    print_report(report)

    path = args.json_path or C.BENCHMARK_PATH
    C.ensure_dirs()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"\n  full report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
