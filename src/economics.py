"""
Expected Net Recovery: turning probabilities into rupees, and rupees into a
ranking.

This module is the answer to improvements.md item 1 — "optimise for money
recovered, not classification accuracy". v2 predicted a root cause and looked
up a fixed action for it. That is a classifier with a dictionary attached, and
it cannot answer the only question that matters commercially: *is this
intervention worth doing at all?*

The core quantity
-----------------
Every action is scored as its incremental value against doing nothing:

    ENR(a) = incremental_margin(a)
           - action_cost(a)
           - expected_failure_cost(a)
           - incremental_chargeback_cost(a)
           - contact_fatigue_penalty(a)
           + incremental_retained_ltv(a)

Because every term is differenced against the do-nothing baseline,
`do_nothing` scores exactly 0.0 by construction. That is a deliberate and
useful property: ENR reads directly as "rupees better than leaving this
alone", and an action with negative ENR is not merely suboptimal, it is
actively worse than inaction. The agent doing nothing is therefore a
*positive* result rather than a failure to decide.

Why margin is differenced rather than just the uplift
-----------------------------------------------------
The naive version of this calculation is `uplift x amount x margin`. It is
wrong for any action that changes the price, and wrongly flattering to
discounts specifically.

A discount is given to *everyone who converts*, including the customers who
would have converted anyway. So the correct comparison is:

    with the discount:  p_a  x amount x (margin - discount)
    doing nothing:      p_0  x amount x  margin

Differencing those two charges the discount against the organic converters as
well as the incremental ones. This is what makes the agent correctly value a
discount on a technical-friction cart as *negative*: those customers already
wanted to buy, a free reminder recovers them at almost the same rate, so the
discount buys ~2 points of conversion and gives away 10 points of margin on
the whole cohort. The naive formula would have called that a win.

Two probabilities that are assumed, not learned
-----------------------------------------------
`stop_and_flag_fraud` and `request_human_review` have no fitted model, and
they never will, because no sane merchant would randomise them into a live
exploration log in order to generate training data. They are modelled
explicitly here and flagged with `probability_is_assumed=True` so that an
auditor can see exactly which numbers came from data and which came from a
stated assumption. See `P_RECOVER_ASSUMPTIONS` below.

Economics can only ever propose
-------------------------------
Nothing in this module blocks anything. It computes and ranks; src/guardrails.py
disposes. The separation matters because it means a miscalibrated probability
can produce a bad *ranking* but cannot produce a prohibited *action*. Fraud is
the clearest case: the chargeback term below makes stopping suspected fraud
the highest-ENR action on the arithmetic alone, and
`retries.never_retry_root_causes` independently forbids retrying it. Two
mechanisms, reached independently, same conclusion.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from . import config as C
from .ml.uplift import BASELINE_ACTION, SURFACE_ACTION_KEYS, make_action_key, split_action_key
from .schemas import (
    ACTION_CHANNEL, ACTIONS_BY_SURFACE, CHECKOUT_ABANDONMENT, DELAYED_RETRY,
    DO_NOTHING, ESCALATE_TO_COLLECTIONS, OFFER_BOUNDED_DISCOUNT,
    OUTREACH_ACTIONS, OVERDUE_RECEIVABLE, PAYMENT_FAILURE, REQUEST_HUMAN_REVIEW,
    STOP_AND_FLAG_FRAUD, CandidateAction, RiskEvent, ScoredAction,
)

# ---------------------------------------------------------------------
# Actions with no fitted model.
#
# Each entry is documented with its justification, because an assumed
# probability that nobody can see is indistinguishable from a made-up one.
# ---------------------------------------------------------------------

# Blocking a payment means it definitively does not settle. There is no
# uncertainty to model: the recovery probability is zero by definition. The
# *value* of doing it comes entirely from the chargeback exposure it avoids.
P_RECOVER_STOP_AND_FLAG = 0.0

# A human reviewing a case does better than any automated action — they can
# phone the customer, read the account history, spot that the invoice is
# wrong — but not perfectly, and not always in time.
#
# It is scored as a haircut on the best automated option's **expected net
# recovery**, not on its probability. Scoring it on probability alone was the
# first implementation and it was wrong in an instructive way: review inherits
# the probability of, say, a 10%-discount action but pays no discount, so it
# captured the same conversion at full margin and beat discounting on almost
# every cart. That is margin arbitrage against the model, not a real
# operational option, and it routed 17% of all carts to a human.
#
# Taking the haircut on ENR instead means review can never outrank the best
# automated play on economics alone. It is selected when *policy* demands a
# person — amount over the auto-approve ceiling, confidence under the
# threshold, a guardrail fencing every alternative — which is precisely its
# architectural job. Its ENR still varies with the size of the opportunity,
# so a forced review on a large invoice is correctly worth more than on a
# small cart, and the benchmark can measure whether the routing pays.
HUMAN_REVIEW_EFFICACY = 0.85

P_RECOVER_ASSUMPTIONS = {
    STOP_AND_FLAG_FRAUD: "zero by definition — a blocked payment does not settle",
    REQUEST_HUMAN_REVIEW: f"ENR is {HUMAN_REVIEW_EFFICACY:.2f} x the best automated option's ENR",
}

# Review is priced twice: once here during ranking, and again in
# `Guardrails._reprice_review` once screening has established which options are
# actually permitted. Only the second price is real. These prefixes mark the
# notes that state the arithmetic, so the re-pricing can *replace* them instead
# of appending to them, while keeping the notes that explain why a person is
# needed at all.
#
# Without this the audit record carried both claims — "85% of
# automated_reminder_with_payment_plan_offer, ENR -375" and "85% of nothing,
# ENR 0" — against a single stored figure of -40. The figure was right and the
# first note was stale, but a reviewer cannot tell which is which from the
# record, and a decision record that contains two contradictory derivations of
# its own number is not evidence. It is a thing that has to be adjudicated
# before it can be used, which defeats the point of writing it down.
#
# Matched by prefix because these notes are prose for a human to read. The
# alternative is a parallel structured field holding the same arithmetic, and
# then there are two places for it to disagree with itself rather than one.
REVIEW_PRICING_NOTE_PREFIXES = ("ENR = ", "this is strictly below")


def is_review_pricing_note(note: str) -> bool:
    """True for notes stating review's arithmetic, which re-pricing supersedes.

    Kept next to the code that writes those notes, so the two cannot drift
    apart in separate files.
    """
    return note.startswith(REVIEW_PRICING_NOTE_PREFIXES)


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------

def candidate_action_key(candidate: CandidateAction) -> str:
    """Map a candidate onto the recovery model's action-key vocabulary.

    The models are keyed by variant (`delayed_retry@12`, not `delayed_retry`)
    because a 12-hour and a 48-hour retry have genuinely different response
    curves — the second one crosses a payday. Collapsing them would throw
    that away.
    """
    action = candidate.action
    if action == DELAYED_RETRY:
        return make_action_key(action, candidate.delay_hours)
    if action == OFFER_BOUNDED_DISCOUNT:
        return make_action_key(action, candidate.discount_pct)
    return action


def build_candidates(event: RiskEvent, cfg: Optional[Mapping[str, Any]] = None
                     ) -> list[CandidateAction]:
    """Every action worth considering for this event.

    Generated from the closed vocabulary in schemas.ACTIONS_BY_SURFACE, so
    it is structurally impossible to produce an action this surface does not
    support — a receivables invoice can never be handed "retry the card".

    Note this generates the *consideration set*, not the permitted set. A
    candidate that will certainly be blocked (outreach to a DND customer,
    say) is still generated and still scored, because the audit trail should
    show what was considered and why it was refused. Filtering here instead
    would make the guardrails invisible.
    """
    cfg = cfg or C.load_config()
    allowed = ACTIONS_BY_SURFACE[event.event_type]
    max_discount = float(cfg["limits"]["max_discount_pct"])
    out: list[CandidateAction] = []

    for action in allowed:
        if action == DELAYED_RETRY:
            # Two horizons: same-day, and after the next likely salary credit.
            out.append(CandidateAction(
                action, delay_hours=12,
                rationale="retry after 12h — clears short-lived balance and issuer faults",
            ))
            out.append(CandidateAction(
                action, delay_hours=48,
                rationale="retry after 48h — more likely to land after a salary credit",
            ))
        elif action == OFFER_BOUNDED_DISCOUNT:
            for pct in (5.0, 10.0):
                if pct > max_discount:
                    # Respects the configured ceiling at generation time so
                    # a lowered cap cannot be reached even by mis-ranking.
                    continue
                out.append(CandidateAction(
                    action, discount_pct=pct, channel=ACTION_CHANNEL.get(action),
                    rationale=f"{pct:g}% off, costed against the margin it gives away",
                ))
        else:
            out.append(CandidateAction(
                action,
                channel=ACTION_CHANNEL.get(action),
                rationale=_default_rationale(action),
            ))
    return out


def _default_rationale(action: str) -> str:
    return {
        DO_NOTHING: "leave it alone — the baseline every other option is measured against",
        STOP_AND_FLAG_FRAUD: "block settlement and refer; avoids chargeback exposure",
        REQUEST_HUMAN_REVIEW: "hand to a person — too large, too uncertain, or fenced by policy",
        ESCALATE_TO_COLLECTIONS: "refer to collections; expensive and requires sign-off",
    }.get(action, action.replace("_", " "))


# ---------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------

class Economics:
    """Prices actions in rupees. Holds no state beyond configuration."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None):
        self.cfg = cfg or C.load_config()
        self.econ = self.cfg["economics"]

    # -- component terms ---------------------------------------------

    def margin_fraction(self, event: RiskEvent) -> float:
        """Fraction of the at-risk amount that recovery actually returns.

        See `economics.value_basis` in config/policy.yaml — receivables
        recover their full face value because the cost is already sunk,
        whereas a cart recovers only its margin.
        """
        basis = self.econ["value_basis"].get(event.event_type, "margin")
        if basis == "full":
            return 1.0
        pct = float(event.customer.gross_margin_pct or 0.0)
        if pct <= 0:
            pct = float(self.econ["default_gross_margin_pct"])
        return pct / 100.0

    def action_cost(self, action: str) -> float:
        """Cost to attempt, win or lose. Unknown actions cost nothing to
        attempt only if explicitly listed as free; otherwise this raises,
        because an unpriced action would silently look like a bargain."""
        costs = self.econ["action_cost_inr"]
        if action not in costs:
            raise KeyError(
                f"action {action!r} has no entry in economics.action_cost_inr. "
                "Refusing to score it — an unpriced action would rank as free."
            )
        return float(costs[action])

    def failure_penalty(self, action: str) -> float:
        """Extra cost incurred when the action is attempted and fails.

        Defaults to zero, so this dict only needs entries for actions where
        failure genuinely costs something beyond the attempt fee: a failed
        retry carries issuer-reputation risk, an unwanted WhatsApp carries
        goodwill cost, an unwarranted collections referral carries real
        relationship damage.
        """
        return float(self.econ["failure_penalty_inr"].get(action, 0.0))

    def contact_fatigue(self, event: RiskEvent, action: str) -> float:
        """Cost of contacting someone who has already been contacted lately.

        Applies only to outreach. Scales with prior contacts in the window,
        so the third message costs more than the first. This is the term
        that stops the agent from being technically-within-policy and
        practically a nuisance: the frequency cap in guardrails is a hard
        wall, and this is the gradient that discourages walking into it.
        """
        if action not in OUTREACH_ACTIONS:
            return 0.0
        per_contact = float(self.econ["contact_fatigue_penalty_inr"])
        return per_contact * max(0, int(event.customer.contacts_last_7d))

    def retained_ltv(self, event: RiskEvent, uplift: float, p_fraud: float = 0.0) -> float:
        """Incremental future value from keeping the relationship intact.

        Weighted by repeat-purchase probability so it does not inflate the
        value of one-off buyers, and by the configured retention weight so
        it cannot dominate the in-period money. This is improvements.md
        item 8: a 40,000 INR failure on a five-year account with a high
        repeat rate deserves more effort than the same amount on a first
        purchase, and this term is what expresses that.

        Scaled by `1 - p_fraud`, which matters more than it looks. Customer
        lifetime value is a property of a *legitimate* customer. If the
        transaction is fraudulent then the party transacting is not the
        account holder, so there is no relationship to retain and no future
        value to gain or lose. Without this factor the term treats a stolen
        credential as a loyal customer and, because annual value routinely
        exceeds a single transaction by 20x or more, it swamps the
        chargeback saving and makes blocking fraud look expensive. That is
        how a well-intentioned LTV term silently switches off fraud
        prevention.

        It remains signed, which produces a property worth pointing out:
        because `stop_and_flag_fraud` has negative uplift, this term is a
        *cost* for it, in proportion to how valuable the customer is and to
        how likely it is that the fraud call is wrong. The agent therefore
        demands stronger evidence before blocking a long-tenured, high-value
        account than a brand-new one — the false-positive cost of a fraud
        block, priced. That was not designed in; it falls out of scoring both
        sides of the trade-off with the same arithmetic.
        """
        weight = float(self.econ["ltv_retention_weight"])
        annual = float(event.customer.estimated_annual_value_inr or 0.0)
        repeat = float(event.customer.repeat_purchase_probability or 0.0)
        legitimate = max(0.0, 1.0 - p_fraud)
        return uplift * weight * annual * repeat * legitimate

    def chargeback_exposure(self, event: RiskEvent, uplift: float, p_fraud: float) -> float:
        """Incremental expected chargeback cost.

        A recovered fraudulent payment is not revenue. It is a refund, plus
        a fee, plus a contribution to the dispute ratio that governs whether
        the merchant keeps its processing terms at all.

        `p_fraud` is the root-cause model's posterior on `fraud_suspected` —
        a model output, never a ground-truth flag. The ground-truth fraud
        markers in the dataset are used only by src/benchmark.py to score
        outcomes; if this term could read them the agent would be marking
        its own homework.

        Signed via uplift, so an action that *reduces* settlement probability
        (blocking) shows a negative exposure, i.e. a benefit. With a 1.5x
        multiplier and a 35% margin, blocking becomes ENR-positive once
        P(fraud) passes roughly 0.35/1.5 = 0.23. That threshold is arithmetic,
        not a magic number in a config file.
        """
        if p_fraud <= 0.0:
            return 0.0
        multiplier = float(self.econ["chargeback_cost_multiplier"])
        return uplift * p_fraud * event.amount_inr * multiplier

    # -- the scorer ---------------------------------------------------

    def score(
        self,
        event: RiskEvent,
        candidate: CandidateAction,
        p_action: float,
        p_baseline: float,
        p_fraud: float = 0.0,
        probability_is_assumed: bool = False,
    ) -> ScoredAction:
        """Price one candidate. Pure function of its arguments."""
        action = candidate.action

        if action == DO_NOTHING:
            # Zero by identity, not by arithmetic. Every money field in this
            # system is incremental against inaction, so inaction is the origin
            # of the scale and cannot be anywhere else on it.
            #
            # The general path below would already land on zero whenever
            # `p_action == p_baseline`, and `rank` does pass the same number
            # twice, because the recovery model's do-nothing arm *is* the
            # baseline arm. But that makes the invariant hold by a coincidence
            # of two dict lookups in a different module. Anyone who later fits
            # a separate estimate for the do-nothing arm — a reasonable thing
            # to want — would price inaction at some small non-zero figure, and
            # nothing would complain: the dashboard would keep rendering every
            # ENR as "rupees better than doing nothing" while the reference
            # point had quietly moved. This is the second mechanism, and it is
            # the one that does not depend on how the caller behaves.
            return ScoredAction(
                candidate=candidate,
                p_recover=p_baseline,
                p_recover_baseline=p_baseline,
                uplift=0.0,
                gross_value_inr=0.0,
                action_cost_inr=0.0,
                expected_failure_cost_inr=0.0,
                expected_chargeback_cost_inr=0.0,
                cx_penalty_inr=0.0,
                ltv_component_inr=0.0,
                expected_net_recovery_inr=0.0,
                probability_is_assumed=probability_is_assumed,
                notes=[f"P(recover) assumed: {P_RECOVER_ASSUMPTIONS.get(action, 'see economics.py')}"]
                if probability_is_assumed else [],
            )

        amount = float(event.amount_inr)
        margin = self.margin_fraction(event)
        discount = float(candidate.discount_pct) / 100.0
        uplift = p_action - p_baseline

        # Incremental margin, differenced properly so that a discount is
        # charged against organic converters too. See the module docstring.
        value_with_action = p_action * amount * max(0.0, margin - discount)
        value_do_nothing = p_baseline * amount * margin
        gross_value = value_with_action - value_do_nothing

        cost = self.action_cost(action)
        failure_cost = (1.0 - p_action) * self.failure_penalty(action)
        chargeback = self.chargeback_exposure(event, uplift, p_fraud)
        cx_penalty = self.contact_fatigue(event, action)
        ltv = self.retained_ltv(event, uplift, p_fraud)

        enr = gross_value - cost - failure_cost - chargeback - cx_penalty + ltv

        notes: list[str] = []
        if discount > 0:
            given_away = p_action * amount * discount
            notes.append(
                f"discount gives away ~{given_away:,.0f} INR of margin, "
                f"including on the {p_baseline:.0%} who would have converted anyway"
            )
        if probability_is_assumed:
            notes.append(f"P(recover) assumed: {P_RECOVER_ASSUMPTIONS.get(action, 'see economics.py')}")
        if chargeback < -1.0:
            notes.append(
                f"avoids ~{-chargeback:,.0f} INR of expected chargeback exposure "
                f"at P(fraud)={p_fraud:.0%}"
            )
        if cx_penalty > 0:
            notes.append(
                f"customer already contacted {event.customer.contacts_last_7d}x in 7d "
                f"({cx_penalty:,.0f} INR fatigue cost)"
            )

        return ScoredAction(
            candidate=candidate,
            p_recover=p_action,
            p_recover_baseline=p_baseline,
            uplift=uplift,
            gross_value_inr=gross_value,
            action_cost_inr=cost,
            expected_failure_cost_inr=failure_cost,
            expected_chargeback_cost_inr=chargeback,
            cx_penalty_inr=cx_penalty,
            ltv_component_inr=ltv,
            expected_net_recovery_inr=enr,
            probability_is_assumed=probability_is_assumed,
            notes=notes,
        )

    def score_human_review(
        self,
        event: RiskEvent,
        candidate: CandidateAction,
        best_automated: Optional[ScoredAction],
    ) -> ScoredAction:
        """Price a hand-off to a person, relative to the best automated play.

        See the HUMAN_REVIEW_EFFICACY note above for why this is a haircut on
        expected net recovery rather than on probability. `p_recover` is still
        reported, for the dashboard and the audit line, as the same haircut
        applied to the best option's probability — it is informative but it is
        not what the ENR is computed from, and `probability_is_assumed` is set
        so nobody mistakes it for a fitted estimate.

        The `min` on the next line is what makes the invariant in the notes
        below unconditionally true. A multiplicative haircut moves a positive
        number down but a negative one *up*: 85% of -3,259 is -2,770, so on an
        event where every option loses money, a bare multiplication would price
        review above the thing it is supposed to be a discount on. That was
        real — 53 events on the held-out split hit it. It could not change a
        decision, because `Guardrails.select` excludes review from the
        selection contest outright, but a docstring claiming an invariant the
        arithmetic does not hold is a trap for whoever later decides the
        exclusion looks redundant and removes it. Taking the worse of the two
        keeps both mechanisms independently sufficient, which is the only
        reason having two is worth anything.
        """
        cost = self.action_cost(candidate.action)
        if best_automated is None:
            base_enr, base_p, base_baseline = 0.0, 0.0, 0.0
        else:
            base_enr = best_automated.expected_net_recovery_inr
            base_p = best_automated.p_recover
            base_baseline = best_automated.p_recover_baseline

        enr = min(HUMAN_REVIEW_EFFICACY * base_enr, base_enr) - cost
        p_action = HUMAN_REVIEW_EFFICACY * base_p

        label = best_automated.candidate.action if best_automated else "nothing"
        return ScoredAction(
            candidate=candidate,
            p_recover=p_action,
            p_recover_baseline=base_baseline,
            uplift=p_action - base_baseline,
            # The value is carried entirely in the ENR haircut, so the
            # component breakdown would be misleading if populated — a
            # reviewer reading these fields should see that they are not the
            # basis of the number.
            gross_value_inr=0.0,
            action_cost_inr=cost,
            expected_failure_cost_inr=0.0,
            expected_chargeback_cost_inr=0.0,
            cx_penalty_inr=0.0,
            ltv_component_inr=0.0,
            expected_net_recovery_inr=enr,
            probability_is_assumed=True,
            notes=[
                # Both of these are matched by REVIEW_PRICING_NOTE_PREFIXES, so
                # that re-pricing after screening replaces them. Changing how
                # either one opens means changing that tuple too; the test suite
                # asserts they still match.
                f"ENR = {HUMAN_REVIEW_EFFICACY:.0%} of the best automated option "
                f"({label}, ENR {base_enr:,.0f} INR) less {cost:,.0f} INR of analyst time, "
                f"floored so it can never exceed that option even when the option loses money",
                "this is strictly below the best automated action for every input; "
                "it is selected when policy requires a person, not when it wins on value",
            ],
        )

    # -- ranking ------------------------------------------------------

    def rank(
        self,
        event: RiskEvent,
        p_by_action_key: Mapping[str, float],
        p_fraud: float = 0.0,
        candidates: Optional[list[CandidateAction]] = None,
    ) -> list[ScoredAction]:
        """Score every candidate and sort by expected net recovery, descending.

        `p_by_action_key` is the recovery model's output for this event,
        keyed by action variant. Actions absent from it are the two
        documented assumptions; anything else absent is a bug and raises
        rather than defaulting to a made-up probability.
        """
        candidates = candidates if candidates is not None else build_candidates(event, self.cfg)
        p_baseline = float(p_by_action_key.get(BASELINE_ACTION, 0.0))

        # Pass one: everything that can be priced on its own terms.
        scored: list[ScoredAction] = []
        deferred: list[CandidateAction] = []

        for cand in candidates:
            if cand.action == REQUEST_HUMAN_REVIEW:
                # Needs the rest of the field scored first.
                deferred.append(cand)
                continue

            key = candidate_action_key(cand)
            if cand.action == STOP_AND_FLAG_FRAUD:
                p_action, assumed = P_RECOVER_STOP_AND_FLAG, True
            elif key in p_by_action_key:
                p_action, assumed = float(p_by_action_key[key]), False
            else:
                raise KeyError(
                    f"no recovery probability for action variant {key!r} on event "
                    f"{event.event_id} (surface {event.event_type}). Known keys: "
                    f"{sorted(p_by_action_key)}. Refusing to invent one — see the "
                    f"'does not extrapolate' note in src/ml/uplift.py."
                )
            scored.append(self.score(
                event, cand, p_action, p_baseline, p_fraud, probability_is_assumed=assumed
            ))

        # Pass two: price human review against the best genuine alternative.
        # Do-nothing is excluded as the reference — a person asked to review an
        # event they will not act on has no upside to be a fraction of.
        alternatives = [s for s in scored if s.candidate.action != DO_NOTHING]
        best = max(alternatives, key=lambda s: s.expected_net_recovery_inr, default=None)
        for cand in deferred:
            scored.append(self.score_human_review(event, cand, best))

        # Ties broken toward the cheaper action, then toward doing nothing.
        # Without this a 5% and a 10% discount that score identically would
        # be ordered arbitrarily by dict iteration, and the agent would give
        # away more margin than it needed to on a coin flip.
        scored.sort(key=lambda s: (
            -s.expected_net_recovery_inr,
            s.action_cost_inr + s.candidate.discount_pct,
            s.candidate.action != DO_NOTHING,
        ))
        return scored


# ---------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------

def explain(scored: ScoredAction) -> str:
    """One-line arithmetic breakdown, for the audit trail and the dashboard.

    Written out term by term rather than as a single number because
    improvements.md item 7 asks for the audit trail to be *decision
    evidence*. "ENR 412 INR" is not evidence; "uplift 14pp x 38,000 INR x
    35% margin, less 2.50 attempt cost, less 1.20 expected failure cost" is
    something a reviewer can check by hand and disagree with.
    """
    c = scored.candidate
    label = c.action
    if c.discount_pct:
        label += f"@{c.discount_pct:g}%"
    if c.delay_hours:
        label += f"+{c.delay_hours}h"
    parts = [
        f"uplift {scored.uplift:+.1%} (p={scored.p_recover:.3f} vs "
        f"baseline {scored.p_recover_baseline:.3f})",
        f"value {scored.gross_value_inr:+,.0f}",
        f"cost {-scored.action_cost_inr:,.0f}",
    ]
    if abs(scored.expected_failure_cost_inr) > 0.005:
        parts.append(f"failure risk {-scored.expected_failure_cost_inr:,.0f}")
    if abs(scored.expected_chargeback_cost_inr) > 0.005:
        parts.append(f"chargeback {-scored.expected_chargeback_cost_inr:+,.0f}")
    if abs(scored.cx_penalty_inr) > 0.005:
        parts.append(f"fatigue {-scored.cx_penalty_inr:,.0f}")
    if abs(scored.ltv_component_inr) > 0.005:
        parts.append(f"ltv {scored.ltv_component_inr:+,.0f}")
    return (f"{label}: " + ", ".join(parts) +
            f"  =>  ENR {scored.expected_net_recovery_inr:+,.2f} INR")
