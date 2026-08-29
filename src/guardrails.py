"""
Guardrails: the layer that can only ever say no.

Economics proposes, guardrails dispose. src/economics.py ranks actions by
expected value and has no power to forbid anything; this module has no power
to choose anything. Every constraint here is a veto or a gate, never a
preference. That asymmetry is the point — it means a miscalibrated model can
produce a poor *ranking* but cannot produce a prohibited *action*, because the
two failure modes are separated by construction.

The distinction between blocking and gating
-------------------------------------------
Two different verdicts, and conflating them would be a design error:

  * **Blocked** — the action is removed from consideration entirely. No
    amount of expected value can bring it back. Messaging a DND customer is
    blocked; there is no rupee figure that makes it acceptable.
  * **Gated** — the action is permitted but cannot execute unattended. It is
    queued for a human to approve. Escalating to collections is gated: it is
    sometimes exactly right, and it is never something a model should do on
    its own authority.

Defence in depth
----------------
Several constraints here duplicate something the economics layer already
discourages, and that redundancy is deliberate rather than sloppy. Retrying
suspected fraud is penalised by the chargeback term in economics *and* hard
blocked here. Contacting a fatigued customer is priced *and* capped. If the
model is wrong, the rule holds; if the rule has a gap, the price still bites.
A single mechanism protecting a money action is one bug away from not
protecting it.

Why the anti-abuse constraints are not tunable knobs
----------------------------------------------------
Retry caps, minimum retry intervals and contact frequency limits are not
optimisation parameters that happen to be set conservatively. Uncapped
retries against a single instrument are the signature of card testing;
rapid-fire retries get a merchant rate-limited or de-platformed; unbounded
messaging is harassment. src/config.py refuses to load a policy file that
sets `max_attempts_per_payment` above 5 for exactly this reason. This system
recovers a merchant's own revenue from its own customers, and every capability
it has is bounded so that it cannot be turned outward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from . import config as C
from .schemas import (
    ACTION_CHANNEL, DO_NOTHING, ESCALATE_TO_COLLECTIONS, HUMAN_GATED_ACTIONS,
    IMMEDIATE_RETRY, OFFER_BOUNDED_DISCOUNT, OUTREACH_ACTIONS, OVERDUE_RECEIVABLE,
    REQUEST_HUMAN_REVIEW, RETRY_ACTIONS, RiskEvent, ScoredAction,
)


# ---------------------------------------------------------------------
# Sweep-level state
# ---------------------------------------------------------------------

@dataclass
class SweepBudget:
    """Constraints that apply across a whole sweep, not to one event.

    Per-event checks cannot catch aggregate failure modes. Every individual
    discount can sit inside the per-event cap while the sweep as a whole
    gives away a fortune, and every individual retry can be spaced correctly
    while the sweep as a whole constitutes a retry storm against an issuer.
    These counters are what make the aggregate bounded too.

    One consequence worth stating plainly, because it is a real property of
    the design rather than an oversight: a sweep-wide cap is *order
    dependent*. Whoever is processed after the budget runs out is refused,
    however valuable they were. On the held-out sweep the retry budget binds
    exactly — 500 of 500 issued, 39 further retries refused — so the order is
    not academic. src/agent.py therefore works events in descending
    at-risk-value order, so a budget that runs out has been spent on the
    largest exposures rather than on whatever happened to be read first, and
    every displaced event says so in its audit record.
    """
    at_risk_total_inr: float = 0.0
    discount_committed_inr: float = 0.0
    retries_issued: int = 0
    contacts_issued: dict[str, int] = field(default_factory=dict)

    def discount_headroom_inr(self, cfg: Mapping[str, Any]) -> float:
        pct = float(cfg["limits"]["max_discount_budget_pct_of_at_risk"])
        ceiling = self.at_risk_total_inr * pct / 100.0
        return max(0.0, ceiling - self.discount_committed_inr)

    def record(self, event: RiskEvent, scored: ScoredAction) -> None:
        """Commit an action's consumption against the sweep budget."""
        action = scored.candidate.action
        if action == OFFER_BOUNDED_DISCOUNT:
            # Charged at face value, not expected value. A committed offer is
            # a liability for its full amount regardless of how likely it is
            # to be redeemed, so budgeting it at p x value would understate
            # the exposure by roughly half.
            self.discount_committed_inr += event.amount_inr * scored.candidate.discount_pct / 100.0
        if action in RETRY_ACTIONS:
            self.retries_issued += 1
        if action in OUTREACH_ACTIONS:
            cid = event.customer.customer_id
            self.contacts_issued[cid] = self.contacts_issued.get(cid, 0) + 1


@dataclass
class GuardrailContext:
    """Everything the guardrails need that is not on the event itself."""
    root_cause: str
    root_cause_confidence: float
    root_cause_distribution: dict[str, float] = field(default_factory=dict)
    budget: SweepBudget = field(default_factory=SweepBudget)


@dataclass
class Verdict:
    """The outcome of screening one event."""
    chosen: ScoredAction
    considered: list[ScoredAction]
    requires_human_approval: bool
    approval_reason: str
    guardrails_applied: list[str]
    rejected_reasons: dict[str, str]


# ---------------------------------------------------------------------

class Guardrails:
    """Stateless with respect to events; all mutable state is in SweepBudget."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, economics: Any = None):
        self.cfg = cfg or C.load_config()
        # Only used to re-price `request_human_review` once the permitted set
        # is known. See `_reprice_review` for why that has to happen here and
        # not in the ranker.
        if economics is None:
            from .economics import Economics
            economics = Economics(self.cfg)
        self.economics = economics

    # =================================================================
    # Per-action screening
    # =================================================================

    def screen(self, event: RiskEvent, scored: ScoredAction, ctx: GuardrailContext) -> ScoredAction:
        """Annotate one scored action with its guardrail verdict.

        Mutates and returns `scored`. Blocked actions are kept in the list
        rather than discarded so the audit trail records what was considered
        and refused — a guardrail that leaves no trace is indistinguishable
        from a guardrail that never ran.
        """
        action = scored.candidate.action
        blocked: list[str] = []
        notes: list[str] = []

        if action == DO_NOTHING:
            # Always available, by definition. Inaction is the one thing that
            # can never be unsafe, and it must always remain reachable as a
            # fallback when everything else is blocked.
            scored.allowed = True
            return scored

        self._check_consent(event, action, blocked, notes)
        self._check_contact_frequency(event, action, ctx, blocked, notes)
        self._check_quiet_hours(event, scored, notes)
        self._check_retries(event, action, ctx, blocked, notes)
        self._check_receivables(event, action, ctx, blocked, notes)
        self._check_discount(event, scored, ctx, blocked, notes)
        self._check_human_review(event, action, ctx, blocked, notes)

        scored.blocked_by.extend(blocked)
        scored.notes.extend(notes)
        scored.allowed = not blocked
        return scored

    # -- human review --------------------------------------------------

    def _check_human_review(self, event: RiskEvent, action: str, ctx: GuardrailContext,
                            blocked: list[str], notes: list[str]) -> None:
        """Escalate to a person only where a person can actually do more.

        This check exists because of a failure mode found by running the
        guardrails over the whole dataset: review was the only action never
        screened by anything, so whenever consent, DND and the never-retry
        rules blocked everything else, review won by elimination. It was
        selected on 30% of events — 556 analyst touches out of 1,844 — and
        most of those were cases where a human had no more latitude than the
        agent did. A DND customer with an expired card cannot be messaged by
        a person either. Booking 40 INR of someone's time to tell them that
        is worse than doing nothing, because it also buries the cases that
        genuinely need attention.

        So review has to *add capability*, not merely add cost. It is
        permitted when a human has real latitude the agent lacks:

          * genuine ambiguity — confidence below the acting threshold, which
            is a judgement call and exactly what people are for;
          * size — above the auto-approve ceiling, where a second pair of
            eyes is warranted regardless of confidence;
          * a process fix — a disputed or erroneous invoice needs a corrected
            document or a conversation, neither of which is in the agent's
            action space at all.

        Collections is not on that list because it does not need to be: it is
        gated directly by the approval check below, so it already reaches a
        human without being routed through review.

        Absent all three, the honest answer is that there is nothing to be
        done, and the agent says so rather than passing the buck.
        """
        if action != REQUEST_HUMAN_REVIEW:
            return

        limits = self.cfg["limits"]
        grounds: list[str] = []

        if ctx.root_cause_confidence < float(limits["min_confidence_to_act"]):
            grounds.append(f"cause is uncertain (confidence {ctx.root_cause_confidence:.2f})")
        if event.amount_inr > float(limits["max_auto_approve_amount_inr"]):
            grounds.append(f"amount {event.amount_inr:,.0f} INR is above the auto-approve ceiling")
        if event.event_type == OVERDUE_RECEIVABLE:
            if ctx.root_cause in set(self.cfg["receivables"]["never_auto_chase_root_causes"]):
                grounds.append(
                    f"{ctx.root_cause} needs a process fix (corrected invoice or dispute "
                    f"resolution), which is outside the agent's action space"
                )

        if grounds:
            notes.append("review is warranted: " + "; ".join(grounds))
            return

        blocked.append(
            "a human has no more latitude here than the agent does — the cause is "
            "clear, the amount is within the auto-approve ceiling, and the blocking "
            "constraints (consent, DND, never-retry) bind a person equally"
        )

    # -- consent ------------------------------------------------------

    def _check_consent(self, event: RiskEvent, action: str,
                       blocked: list[str], notes: list[str]) -> None:
        if action not in OUTREACH_ACTIONS:
            return
        channel = ACTION_CHANNEL.get(action)
        if channel is None:
            blocked.append("outreach action has no declared channel — cannot verify consent")
            return

        cust = event.customer
        if cust.dnd_flagged and self.cfg["contact"]["honour_dnd"]:
            blocked.append("customer is DND-flagged — no outreach on any channel")
            return

        required = set(self.cfg["contact"]["consent_required_channels"])
        if channel in required and not cust.has_consent(channel):
            blocked.append(f"no opt-in recorded for {channel}")
            return
        if not cust.has_consent(channel):
            # Even for channels not on the explicit opt-in list, an absent
            # consent record blocks. The safe reading of "we have no record"
            # is "no", never "probably fine".
            blocked.append(f"no consent recorded for {channel}")

    # -- contact frequency --------------------------------------------

    def _check_contact_frequency(self, event: RiskEvent, action: str, ctx: GuardrailContext,
                                 blocked: list[str], notes: list[str]) -> None:
        if action not in OUTREACH_ACTIONS:
            return
        contact = self.cfg["contact"]
        cust = event.customer

        already = cust.contacts_last_7d + ctx.budget.contacts_issued.get(cust.customer_id, 0)
        cap = int(contact["max_contacts_per_customer_per_7d"])
        if already >= cap:
            blocked.append(
                f"contact cap reached ({already}/{cap} in 7d, including this sweep)"
            )
        min_gap = float(contact["min_hours_between_contacts"])
        if cust.hours_since_last_contact < min_gap:
            blocked.append(
                f"last contacted {cust.hours_since_last_contact:.0f}h ago, "
                f"minimum gap is {min_gap:.0f}h"
            )

    # -- quiet hours ---------------------------------------------------

    def _check_quiet_hours(self, event: RiskEvent, scored: ScoredAction,
                           notes: list[str]) -> None:
        """Defer outreach out of quiet hours rather than blocking it.

        A reminder that would land at 03:00 is not an illegitimate reminder,
        it is a badly timed one. Blocking it would silently lose recoverable
        revenue; sending it would be obnoxious. So the action survives with a
        delay attached, and the adapter honours it.
        """
        if scored.candidate.action not in OUTREACH_ACTIONS:
            return
        contact = self.cfg["contact"]
        start = int(contact["quiet_hours_start"])
        end = int(contact["quiet_hours_end"])

        send_hour = (event.occurred_at_hour + scored.candidate.delay_hours) % 24
        if not in_quiet_hours(send_hour, start, end):
            return

        defer_by = (end - send_hour) % 24
        scored.candidate.delay_hours += defer_by
        notes.append(
            f"would land at {send_hour:02d}:00, inside quiet hours "
            f"({start:02d}:00-{end:02d}:00) — deferred {defer_by}h to {end:02d}:00"
        )

    # -- retry discipline ---------------------------------------------

    def _check_retries(self, event: RiskEvent, action: str, ctx: GuardrailContext,
                       blocked: list[str], notes: list[str]) -> None:
        if action not in RETRY_ACTIONS:
            return
        retries = self.cfg["retries"]

        if ctx.root_cause in set(retries["never_retry_root_causes"]):
            blocked.append(
                f"root cause {ctx.root_cause!r} is on the never-retry list — "
                f"retrying cannot succeed and looks like card testing"
            )

        # Read the whole posterior, not just its argmax. See the comment on
        # retries.max_fraud_probability_for_retry in config/policy.yaml.
        p_fraud = float(ctx.root_cause_distribution.get("fraud_suspected", 0.0))
        fraud_cap = float(retries.get("max_fraud_probability_for_retry", 0.15))
        if p_fraud > fraud_cap:
            blocked.append(
                f"P(fraud)={p_fraud:.0%} exceeds the {fraud_cap:.0%} retry ceiling "
                f"(predicted cause was {ctx.root_cause!r}, but the posterior is what matters)"
            )

        attempts = int(event.features.get("retry_count", 0) or 0)
        max_attempts = int(retries["max_attempts_per_payment"])
        # `attempts` have already happened, so this one would be attempts+1.
        if attempts + 1 > max_attempts:
            blocked.append(
                f"already attempted {attempts}x, cap is {max_attempts} per payment"
            )

        # An immediate retry after a previous attempt breaches the minimum
        # spacing. A delayed retry satisfies it by construction, because the
        # shortest delay variant offered (12h) already exceeds the configured
        # minimum — asserted in tests so lowering one without the other fails
        # loudly rather than quietly permitting rapid-fire attempts.
        min_gap = float(retries["min_hours_between_attempts"])
        if attempts > 0 and action == IMMEDIATE_RETRY:
            blocked.append(
                f"an immediate retry would breach the {min_gap:.0f}h minimum "
                f"interval after {attempts} prior attempt(s)"
            )

        sweep_cap = int(retries["max_retries_per_sweep"])
        if ctx.budget.retries_issued >= sweep_cap:
            blocked.append(
                f"sweep retry budget exhausted ({ctx.budget.retries_issued}/{sweep_cap})"
            )

    # -- receivables ---------------------------------------------------

    def _check_receivables(self, event: RiskEvent, action: str, ctx: GuardrailContext,
                           blocked: list[str], notes: list[str]) -> None:
        if event.event_type != OVERDUE_RECEIVABLE:
            return
        rec = self.cfg["receivables"]

        if ctx.root_cause in set(rec["never_auto_chase_root_causes"]) and action != DO_NOTHING:
            if action != REQUEST_HUMAN_REVIEW:
                blocked.append(
                    f"root cause {ctx.root_cause!r} must not be chased automatically — "
                    f"the fix is a corrected invoice or a resolved dispute, not pressure"
                )

        if action == ESCALATE_TO_COLLECTIONS:
            days = float(event.features.get("days_overdue", 0) or 0)
            min_days = float(rec["min_days_overdue_for_collections"])
            if days < min_days:
                blocked.append(
                    f"only {days:.0f} days overdue, collections requires {min_days:.0f}"
                )
            if event.features.get("dispute_flagged_in_ar"):
                blocked.append("invoice is flagged as disputed in AR — cannot escalate")

    # -- discounts -----------------------------------------------------

    def _check_discount(self, event: RiskEvent, scored: ScoredAction, ctx: GuardrailContext,
                        blocked: list[str], notes: list[str]) -> None:
        if scored.candidate.action != OFFER_BOUNDED_DISCOUNT:
            return
        pct = float(scored.candidate.discount_pct)
        cap = float(self.cfg["limits"]["max_discount_pct"])
        if pct > cap:
            blocked.append(f"discount {pct:g}% exceeds the {cap:g}% cap")

        face_value = event.amount_inr * pct / 100.0
        headroom = ctx.budget.discount_headroom_inr(self.cfg)
        if face_value > headroom:
            blocked.append(
                f"sweep discount budget: this offer commits {face_value:,.0f} INR "
                f"but only {headroom:,.0f} INR of headroom remains"
            )

    # =================================================================
    # Selection
    # =================================================================

    def _reprice_review(self, event: RiskEvent, screened: list[ScoredAction],
                        best_permitted: Optional[ScoredAction]) -> Optional[ScoredAction]:
        """Re-price `request_human_review` against the *permitted* field.

        This closes a hole the uplift benchmark found, and it is worth spelling
        out because the fix is small and the consequence was not.

        `Economics.rank` prices review as a haircut on the best alternative,
        and its own docstring claims that "by construction this cannot outrank
        the best automated action". That holds only while the comparison set is
        the same one the price was derived from. It is not: the ranker sees
        every candidate, and selection happens *after* screening. So a
        high-value option that policy then refuses still sets review's price,
        and review walks into the permitted set carrying a number it has no
        claim to.

        The effect was not academic. On the 1,844-event held-out split the
        agent handed 48 events to a person while a permitted automated action
        was sitting right there — 5,681,844 INR of face value, and 1,878,460
        INR of realised recovery forgone, which was most of the gap between the
        agent and the compliant rules baseline. The agent was not being
        cautious; it was mispricing caution.

        So review is re-priced here, off the best option that actually survived
        screening, and excluded from the selection contest outright. Two
        mechanisms for one property, which is the pattern used everywhere else
        in this module: the arithmetic now makes review unable to win, and the
        selection code would not let it win even if the arithmetic changed.

        When nothing is permitted, `best_permitted` is None and review prices
        at minus the cost of the analyst's time. That is the honest number —
        there is no automated upside for it to be a fraction of — and it is why
        the "nothing permitted" branch selects review on the grounds that a
        person has latitude the agent lacks, rather than on value.
        """
        from .economics import is_review_pricing_note

        for i, s in enumerate(screened):
            if s.candidate.action != REQUEST_HUMAN_REVIEW:
                continue
            repriced = self.economics.score_human_review(event, s.candidate, best_permitted)
            # The screening verdict was reached on the action, not on its
            # price, so it carries over untouched.
            repriced.allowed = s.allowed
            repriced.blocked_by = list(s.blocked_by)
            # The old arithmetic notes are dropped rather than kept alongside
            # the new ones — they describe a price derived from options that
            # policy has since refused, and leaving both in the record leaves a
            # reviewer to work out which of two derivations produced the number
            # they are looking at. Everything else survives: the notes saying
            # *why* a person is warranted were never about the price and are
            # still true.
            kept = [n for n in s.notes if not is_review_pricing_note(n)]
            repriced.notes = kept + [n for n in repriced.notes if n not in kept]
            screened[i] = repriced
            return repriced
        return None

    def select(self, event: RiskEvent, ranked: list[ScoredAction],
               ctx: GuardrailContext) -> Verdict:
        """Screen every candidate, then pick the best permitted one.

        Returns the full considered set alongside the choice, because
        improvements.md item 7 asks the audit trail to be decision *evidence*.
        Recording only the winner tells a reviewer nothing about whether the
        decision was close, whether a cheaper option was nearly as good, or
        whether a guardrail is quietly blocking most of the action space.
        """
        applied: list[str] = []
        rejected: dict[str, str] = {}

        if C.kill_switch_engaged(self.cfg):
            # Nothing is chosen and nothing is executed. Deliberately checked
            # here as well as in the adapter: the halt file should stop
            # decisions being *made*, not merely stop them being sent.
            do_nothing = _find_do_nothing(ranked)
            do_nothing.notes.append("kill switch engaged — sweep halted")
            return Verdict(
                chosen=do_nothing, considered=ranked,
                requires_human_approval=False,
                approval_reason="",
                guardrails_applied=["kill_switch: HALT file present, no action taken"],
                rejected_reasons={s.candidate.action: "kill switch engaged"
                                  for s in ranked if s.candidate.action != DO_NOTHING},
            )

        screened = [self.screen(event, s, ctx) for s in ranked]

        for s in screened:
            label = _label(s)
            if s.blocked_by:
                rejected[label] = "; ".join(s.blocked_by)
                for reason in s.blocked_by:
                    entry = f"{label}: {reason}"
                    if entry not in applied:
                        applied.append(entry)

        limits = self.cfg["limits"]
        min_enr = float(limits["min_expected_net_recovery_inr"])
        min_conf = float(limits["min_confidence_to_act"])
        max_auto = float(limits["max_auto_approve_amount_inr"])

        # Review is excluded from the contest on purpose. See
        # `_reprice_review` — it is a fallback and a mandate, never a winner.
        permitted = [s for s in screened if s.allowed
                     and s.candidate.action not in (DO_NOTHING, REQUEST_HUMAN_REVIEW)]
        best = max(permitted, key=lambda s: s.expected_net_recovery_inr, default=None)
        review = self._reprice_review(event, screened, best)

        # --- Nothing automated is permitted. ---
        if best is None:
            if review is not None and review.allowed:
                # Everything the agent could do itself was refused, but a
                # person has latitude the agent lacks — a disputed invoice
                # needs the dispute resolved, an uncertain cause needs a
                # judgement call. `_review_adds_capability` has already
                # established that; if it had not, review would be blocked
                # here and this branch would not be taken.
                review.notes.append(
                    "no automated action was permitted; escalated to a person because "
                    "review adds capability the agent does not have"
                )
                applied.append(f"{_label(review)}: selected as the only permitted option")
                return Verdict(review, screened, True,
                               "every automated option was refused by policy; "
                               "this event needs a person",
                               applied, rejected)
            chosen = _find_do_nothing(screened)
            chosen.notes.append("every alternative was blocked by policy")
            return Verdict(chosen, screened, False, "", applied, rejected)

        if best.expected_net_recovery_inr < min_enr:
            label = _label(best)
            rejected[label] = (
                f"expected net recovery {best.expected_net_recovery_inr:,.2f} INR is "
                f"below the {min_enr:,.0f} INR floor — not worth the customer-experience cost"
            )
            applied.append(
                f"min_expected_net_recovery: best option {label} at "
                f"{best.expected_net_recovery_inr:,.2f} INR < {min_enr:,.0f} INR floor"
            )
            chosen = _find_do_nothing(screened)
            chosen.notes.append(
                f"deliberately taking no action — the best available option was worth only "
                f"{best.expected_net_recovery_inr:,.2f} INR"
            )
            return Verdict(chosen, screened, False, "", applied, rejected)

        # --- Approval gates. The action stands; a human must release it. ---
        gates: list[str] = []

        if event.amount_inr > max_auto:
            gates.append(
                f"amount {event.amount_inr:,.0f} INR exceeds the "
                f"{max_auto:,.0f} INR auto-approve ceiling"
            )
        if ctx.root_cause_confidence < min_conf:
            gates.append(
                f"root-cause confidence {ctx.root_cause_confidence:.2f} is below the "
                f"{min_conf:.2f} threshold to act unattended"
            )
        if best.candidate.action in HUMAN_GATED_ACTIONS:
            gates.append(
                f"{best.candidate.action} always requires sign-off — it is not an "
                f"action a model may take on its own authority"
            )
        if (best.candidate.action == ESCALATE_TO_COLLECTIONS
                and self.cfg["receivables"]["collections_requires_human_signoff"]):
            gates.append("collections escalation requires human sign-off by policy")

        for g in gates:
            entry = f"{_label(best)}: gated — {g}"
            if entry not in applied:
                applied.append(entry)

        return Verdict(
            chosen=best,
            considered=screened,
            requires_human_approval=bool(gates),
            approval_reason="; ".join(gates),
            guardrails_applied=applied,
            rejected_reasons=rejected,
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def in_quiet_hours(hour: int, start: int, end: int) -> bool:
    """Quiet hours wrap midnight, so a naive `start <= h < end` is wrong.

    Public because src/adapters/messaging.py re-checks quiet hours at the
    egress boundary and must use exactly this implementation. Two copies of a
    midnight-wrapping comparison is two chances to get it wrong in different
    ways, and a quiet-hours bug means a 3am payment reminder.
    """
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _label(scored: ScoredAction) -> str:
    c = scored.candidate
    label = c.action
    if c.discount_pct:
        label += f"@{c.discount_pct:g}%"
    if c.delay_hours:
        label += f"+{c.delay_hours}h"
    return label


def _find_do_nothing(scored: list[ScoredAction]) -> ScoredAction:
    for s in scored:
        if s.candidate.action == DO_NOTHING:
            return s
    raise RuntimeError(
        "no do_nothing candidate was generated — inaction must always remain "
        "available as a fallback. This is a bug in src/economics.build_candidates."
    )
