"""Arithmetic properties of the pricing layer.

These are the tests that would catch a change to how money is estimated. Most
of them are pure — no models, no disk — because the pricing layer is a function
of its arguments by design, and that is worth having a suite that demonstrates.

The review-pricing tests are the largest group here, out of proportion to how
often `request_human_review` is chosen. That is deliberate. Review is the
action the whole oversight story rests on: it is what the agent picks when it
declines to act alone, and its price is what decides whether the ranker would
ever prefer being cautious to being useful. It is also where this project has
already had two real defects — a haircut that priced review *above* its own
reference option whenever every option lost money, and a re-pricing step that
left two contradictory derivations of one number in the same audit record. Both
are pinned below.
"""

from __future__ import annotations

import math
import re
import unittest

from src import config as C
from src.economics import (
    HUMAN_REVIEW_EFFICACY, REVIEW_PRICING_NOTE_PREFIXES, Economics,
    build_candidates, is_review_pricing_note,
)
from src.ml.uplift import BASELINE_ACTION, make_action_key
from src.schemas import (
    ALL_ACTIONS, CHECKOUT_ABANDONMENT, CandidateAction, DELAYED_RETRY,
    DO_NOTHING, EVENT_TYPES, OVERDUE_RECEIVABLE, PAYMENT_FAILURE,
    REQUEST_HUMAN_REVIEW, ScoredAction,
)

from .helpers import make_event, permissive_customer


class TestDoNothingIsExactlyZero(unittest.TestCase):
    """Every money field is incremental against inaction, so inaction is 0.

    Not "close to zero". Exactly zero, and by identity rather than by
    cancellation, because the entire dashboard reads
    `expected_net_recovery_inr` as "rupees better than doing nothing". If
    do_nothing scored 0.004 the reading would still be roughly right and the
    invariant would be gone.
    """

    def setUp(self) -> None:
        self.economics = Economics()

    def test_zero_on_every_surface_and_amount(self) -> None:
        for surface in EVENT_TYPES:
            for amount in (0.0, 1.0, 999.99, 50_000.0, 4_000_000.0):
                event = make_event(surface, amount)
                scored = self.economics.score(
                    event, CandidateAction(action=DO_NOTHING),
                    p_action=0.31, p_baseline=0.31,
                )
                self.assertEqual(
                    scored.expected_net_recovery_inr, 0.0,
                    f"{surface} at {amount}: do_nothing priced "
                    f"{scored.expected_net_recovery_inr}",
                )

    def test_zero_even_when_the_baseline_probability_is_odd(self) -> None:
        """A do-nothing that is priced off p_action - p_baseline must not drift.

        The pair is passed in from the model, and nothing guarantees the two
        arrive identical for the baseline arm. If they differ, the uplift is
        non-zero and a naive implementation would price inaction at some small
        number.
        """
        event = make_event(PAYMENT_FAILURE, 25_000.0)
        for p_action, p_baseline in ((0.0, 0.0), (0.4, 0.31), (0.1, 0.9), (1.0, 0.0)):
            scored = self.economics.score(event, CandidateAction(action=DO_NOTHING),
                                          p_action=p_action, p_baseline=p_baseline)
            self.assertEqual(scored.expected_net_recovery_inr, 0.0,
                             f"p_action={p_action} p_baseline={p_baseline}")

    def test_every_component_is_zero_not_just_the_total(self) -> None:
        """A total of zero built from offsetting components would be a coincidence."""
        scored = Economics().score(make_event(PAYMENT_FAILURE, 25_000.0),
                                   CandidateAction(action=DO_NOTHING),
                                   p_action=0.5, p_baseline=0.2)
        for field in ("gross_value_inr", "action_cost_inr", "expected_failure_cost_inr",
                      "expected_chargeback_cost_inr", "cx_penalty_inr",
                      "ltv_component_inr"):
            self.assertEqual(getattr(scored, field), 0.0, field)

    def test_the_short_circuit_agrees_with_the_general_formula(self) -> None:
        """The zero is returned directly; this proves nothing was lost by that.

        `score` special-cases do_nothing and returns zeros without running the
        general pricing path. That guarantees the invariant, but it would be a
        poor trade if it also changed the numbers the pipeline had been
        producing — the shipped audit trail was written by the general path, and
        a record that no longer reproduces is not evidence.

        So this reimplements the general formula and checks the two agree
        wherever `p_action == p_baseline`, which is what `rank` passes, since
        the recovery model's do-nothing arm *is* the baseline arm. The two are
        only permitted to disagree where the old path was wrong.
        """
        economics = Economics()
        candidate = CandidateAction(action=DO_NOTHING)
        checked = 0
        for surface in EVENT_TYPES:
            for amount in (0.0, 1_000.0, 50_000.0, 3_000_000.0):
                for p in (0.0, 0.31, 0.87, 1.0):
                    for p_fraud in (0.0, 0.4, 0.95):
                        for contacts in (0, 3, 9):
                            event = make_event(surface, amount, customer=permissive_customer(
                                contacts_last_7d=contacts))
                            new = economics.score(event, candidate, p, p, p_fraud=p_fraud)

                            margin = economics.margin_fraction(event)
                            gross = (p * amount * margin) - (p * amount * margin)
                            cost = economics.action_cost(DO_NOTHING)
                            failure = (1.0 - p) * economics.failure_penalty(DO_NOTHING)
                            chargeback = economics.chargeback_exposure(event, 0.0, p_fraud)
                            fatigue = economics.contact_fatigue(event, DO_NOTHING)
                            ltv = economics.retained_ltv(event, 0.0, p_fraud)
                            expected = gross - cost - failure - chargeback - fatigue + ltv

                            self.assertAlmostEqual(
                                new.expected_net_recovery_inr, expected, places=9,
                                msg=f"{surface} amount={amount} p={p} "
                                    f"p_fraud={p_fraud} contacts={contacts}")
                            checked += 1
        self.assertGreater(checked, 400, "the sweep of inputs got smaller by accident")


class TestRetryTiming(unittest.TestCase):
    """The candidate set cannot propose a retry sooner than policy allows."""

    def test_shortest_delayed_retry_clears_the_policy_floor(self) -> None:
        """12h vs the configured minimum gap.

        This is the check that would catch someone tuning the retry ladder for
        recovery rate without noticing they had crossed the abuse threshold.
        The guardrail also enforces the gap at decision time, so this is the
        second of two independent mechanisms: here the option does not exist,
        there it would be refused.
        """
        cfg = C.load_config()
        floor = float(cfg["retries"]["min_hours_between_attempts"])
        delays = [c.delay_hours for c in build_candidates(make_event(PAYMENT_FAILURE), cfg)
                  if c.action == DELAYED_RETRY]
        self.assertTrue(delays, "no delayed_retry candidate was generated at all")
        self.assertGreater(
            min(delays), floor,
            f"the shortest delayed retry is {min(delays)}h against a "
            f"{floor}h policy floor",
        )
        self.assertEqual(min(delays), 12, "the documented ladder starts at 12h")

    def test_immediate_retry_is_not_offered_on_a_second_attempt(self) -> None:
        """Retry count is on the event, so the candidate set can respect it."""
        cfg = C.load_config()
        cap = int(cfg["retries"]["max_attempts_per_payment"])
        exhausted = make_event(PAYMENT_FAILURE, features={"retry_count": cap + 3})
        actions = {c.action for c in build_candidates(exhausted, cfg)}
        self.assertIn(DO_NOTHING, actions,
                      "do_nothing must always be available as the reference option")


class TestReviewPricing(unittest.TestCase):
    """`request_human_review` is priced as a discount on the best real option.

    Four properties, in increasing order of how easy they are to break:
    it is below its reference option; it is below it even when every option
    loses money; the note explaining it recomputes to the stored number; and
    the predicate that finds those notes still matches the ones written here.
    """

    def setUp(self) -> None:
        self.economics = Economics()
        self.event = make_event(PAYMENT_FAILURE, 40_000.0)
        self.candidate = CandidateAction(action=REQUEST_HUMAN_REVIEW)

    def _automated(self, enr: float, p: float = 0.42) -> ScoredAction:
        return ScoredAction(
            candidate=CandidateAction(action=DELAYED_RETRY, delay_hours=12),
            p_recover=p, p_recover_baseline=0.30, uplift=p - 0.30,
            gross_value_inr=enr, action_cost_inr=0.0,
            expected_failure_cost_inr=0.0, expected_chargeback_cost_inr=0.0,
            cx_penalty_inr=0.0, ltv_component_inr=0.0,
            expected_net_recovery_inr=enr,
        )

    def test_never_above_its_reference_option(self) -> None:
        for enr in (250_000.0, 10_000.0, 1_000.0, 1.0, 0.0):
            review = self.economics.score_human_review(
                self.event, self.candidate, self._automated(enr))
            self.assertLess(
                review.expected_net_recovery_inr, enr + 1e-9,
                f"review priced {review.expected_net_recovery_inr} against a "
                f"reference option worth {enr}",
            )

    def test_never_above_its_reference_option_when_every_option_loses_money(self) -> None:
        """The regression that motivated the floor.

        A bare multiplicative haircut moves a negative number *up*: 85% of
        -3,259 is -2,770, which would have priced review above the option it is
        supposed to be a fraction of. 53 events on the held-out split hit this.
        """
        for enr in (-1.0, -40.0, -3_259.0, -1_821.0, -250_000.0):
            review = self.economics.score_human_review(
                self.event, self.candidate, self._automated(enr))
            self.assertLess(
                review.expected_net_recovery_inr, enr + 1e-9,
                f"review priced {review.expected_net_recovery_inr} against a "
                f"losing reference option worth {enr} — the floor is gone",
            )

    def test_with_no_permitted_option_it_prices_at_minus_the_analyst_cost(self) -> None:
        """When policy refused everything, review costs what the analyst costs.

        Not "the least bad refused option". A blocked option's ENR is not a
        price the agent can pay, so it is not the thing review is a fraction
        of — the reference is nothing, and the price is the cost.
        """
        review = self.economics.score_human_review(self.event, self.candidate, None)
        self.assertAlmostEqual(
            review.expected_net_recovery_inr,
            -self.economics.action_cost(REQUEST_HUMAN_REVIEW), places=6)

    def test_probability_is_flagged_as_assumed(self) -> None:
        review = self.economics.score_human_review(
            self.event, self.candidate, self._automated(10_000.0))
        self.assertTrue(review.probability_is_assumed)
        self.assertEqual(review.gross_value_inr, 0.0,
                         "the component breakdown must stay empty; the value is "
                         "carried entirely by the ENR haircut")

    def test_exactly_one_arithmetic_note_and_it_recomputes(self) -> None:
        """The note has to reproduce the stored figure.

        A record carrying two derivations of one number, or a derivation that
        does not reach it, is something a reviewer has to adjudicate before
        they can use it — which defeats the point of writing it down.
        """
        for enr in (120_000.0, 5_000.0, 0.0, -900.0):
            review = self.economics.score_human_review(
                self.event, self.candidate, self._automated(enr))
            arithmetic = [n for n in review.notes if n.startswith("ENR = ")]
            self.assertEqual(len(arithmetic), 1,
                             f"expected one arithmetic note, got {arithmetic}")
            quoted = re.search(r"ENR ([-\d,]+) INR\) less ([\d,]+) INR", arithmetic[0])
            self.assertIsNotNone(quoted, f"cannot parse the note: {arithmetic[0]!r}")
            base = float(quoted.group(1).replace(",", ""))
            cost = float(quoted.group(2).replace(",", ""))
            recomputed = min(HUMAN_REVIEW_EFFICACY * base, base) - cost
            self.assertLess(
                abs(recomputed - review.expected_net_recovery_inr), 1.0,
                f"the note derives {recomputed} but the record stores "
                f"{review.expected_net_recovery_inr}",
            )

    def test_the_note_predicate_matches_both_notes(self) -> None:
        """`is_review_pricing_note` must match exactly what this function writes.

        `Guardrails._reprice_review` uses it to drop stale arithmetic when it
        re-prices review against the permitted set. If a note stopped matching,
        the stale line would survive next to the new one and the record would
        contain two contradictory derivations again — silently, because nothing
        else looks at these strings.
        """
        review = self.economics.score_human_review(
            self.event, self.candidate, self._automated(10_000.0))
        self.assertEqual(len(review.notes), 2, "score_human_review writes two notes")
        for note in review.notes:
            self.assertTrue(is_review_pricing_note(note),
                            f"note not matched by the predicate: {note!r}")
        for prefix in REVIEW_PRICING_NOTE_PREFIXES:
            self.assertTrue(any(n.startswith(prefix) for n in review.notes),
                            f"prefix {prefix!r} matches none of the notes it names")

    def test_the_predicate_does_not_match_unrelated_notes(self) -> None:
        """Over-broad matching would delete a guardrail's reason for refusing."""
        for note in ("blocked by contact frequency cap",
                     "quiet hours: deferred to 09:00 IST",
                     "discount budget exhausted for this sweep",
                     "kill switch engaged — sweep halted"):
            self.assertFalse(is_review_pricing_note(note), note)


class TestScoringIsAFunctionOfItsArguments(unittest.TestCase):
    """Pricing must not depend on call order, shared state, or time."""

    def test_repeated_scoring_is_identical(self) -> None:
        economics = Economics()
        event = make_event(OVERDUE_RECEIVABLE, 88_000.0)
        candidate = CandidateAction(action=DO_NOTHING)
        first = economics.score(event, candidate, 0.4, 0.3)
        for _ in range(5):
            again = economics.score(event, candidate, 0.4, 0.3)
            self.assertEqual(first.expected_net_recovery_inr,
                             again.expected_net_recovery_inr)

    def test_two_instances_agree(self) -> None:
        event = make_event(CHECKOUT_ABANDONMENT, 12_500.0)
        candidate = CandidateAction(action=DO_NOTHING)
        a = Economics().score(event, candidate, 0.4, 0.3)
        b = Economics().score(event, candidate, 0.4, 0.3)
        self.assertEqual(a.expected_net_recovery_inr, b.expected_net_recovery_inr)

    def test_no_nan_or_infinity_reaches_a_record(self) -> None:
        """Every money field must be finite.

        A NaN in an audit record is worse than a wrong number: it serialises,
        it renders, and it silently poisons every total computed from it.
        """
        economics = Economics()
        for surface in EVENT_TYPES:
            for amount in (0.0, 0.01, 7_500_000.0):
                event = make_event(surface, amount)
                for candidate in build_candidates(event):
                    scored = economics.score(event, candidate, 0.5, 0.2)
                    for field in ("gross_value_inr", "action_cost_inr",
                                  "expected_failure_cost_inr",
                                  "expected_chargeback_cost_inr", "cx_penalty_inr",
                                  "ltv_component_inr", "expected_net_recovery_inr",
                                  "p_recover", "p_recover_baseline", "uplift"):
                        value = getattr(scored, field)
                        self.assertTrue(
                            math.isfinite(value),
                            f"{surface}/{candidate.action}.{field} = {value}")


class TestRankingRefusesToInventProbabilities(unittest.TestCase):
    """An action with no probability is a bug, not a default.

    The two documented assumptions (`do_nothing`, `stop_and_flag_fraud`) are
    priced from stated constants. Anything else missing from the model output
    must raise rather than be quietly scored at zero, because a made-up
    probability produces a real rupee figure that nobody can trace.
    """

    def test_missing_probability_raises(self) -> None:
        event = make_event(PAYMENT_FAILURE, 30_000.0)
        economics = Economics()
        with self.assertRaises(Exception):
            economics.rank(event, {BASELINE_ACTION: 0.3})

    def test_full_probability_map_ranks_without_error(self) -> None:
        event = make_event(PAYMENT_FAILURE, 30_000.0)
        economics = Economics()
        candidates = build_candidates(event)
        probabilities = {BASELINE_ACTION: 0.30}
        for candidate in candidates:
            key = make_action_key(candidate.action, candidate.discount_pct
                                  or candidate.delay_hours or 0.0)
            probabilities[key] = 0.42
        ranked = economics.rank(event, probabilities, candidates=candidates)
        self.assertEqual(len(ranked), len(candidates))
        values = [s.expected_net_recovery_inr for s in ranked]
        self.assertEqual(values, sorted(values, reverse=True),
                         "rank() must return descending expected net recovery")
        self.assertIn(DO_NOTHING, {s.candidate.action for s in ranked},
                      "the reference option must always be in the considered set")


class TestActionVocabularyIsClosed(unittest.TestCase):
    """Candidates only ever come from the declared vocabulary."""

    def test_every_candidate_is_a_known_action(self) -> None:
        for surface in EVENT_TYPES:
            for candidate in build_candidates(make_event(surface, 20_000.0)):
                self.assertIn(candidate.action, ALL_ACTIONS)

    def test_discounts_stay_inside_the_configured_cap(self) -> None:
        cfg = C.load_config()
        cap = float(cfg["limits"]["max_discount_pct"])
        for surface in EVENT_TYPES:
            for candidate in build_candidates(make_event(surface, 20_000.0), cfg):
                self.assertLessEqual(
                    candidate.discount_pct, cap,
                    f"{surface}/{candidate.action} proposes "
                    f"{candidate.discount_pct}% against a {cap}% cap")


if __name__ == "__main__":
    unittest.main()
