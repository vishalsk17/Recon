"""
Tests for the test scaffolding.

Every other file in this suite builds its cases out of `tests/helpers.py`, so a
fixture that has drifted from the real data contract does not fail — it quietly
tests a shape nothing in the system produces, and reports that everything is
fine. That is the worst failure mode available to a test suite, and it is the
reason this file exists.

Two of the checks here are about a bug that already happened once.
`ranked_options` used to write the flat probability map *after* the baseline
entry, and since `uplift.BASELINE_ACTION` is the string `"do_nothing"` — which
is also a real candidate's key — the baseline came out equal to every other arm
and every uplift in the ranked set was zero. Nothing raised. The symptom was a
selection test cheerfully reporting that the agent had declined to act. So the
helper's contract is now asserted directly: the baseline is what the caller
asked for, and the uplifts are not all zero.

Nothing here needs trained weights. That is the whole point of the helpers: a
guardrail or pricing test should be able to state its inputs rather than
discover them, and that only works if the fixtures involve no model.
"""

from __future__ import annotations

import unittest

from src import audit as A
from src import dataio
from src.ml import features as F
from src.ml.uplift import BASELINE_ACTION
from src.schemas import (
    ACTIONS_BY_SURFACE,
    CHECKOUT_ABANDONMENT,
    EVENT_TYPES,
    OVERDUE_RECEIVABLE,
    PAYMENT_FAILURE,
    SEGMENTS,
    default_customer,
)

from .helpers import (
    FEATURE_VALUES,
    AuditCase,
    clear_context,
    make_decision,
    make_event,
    needs_data,
    permissive_customer,
    ranked_options,
)


class TestTheEventFixtureMatchesTheRealContract(unittest.TestCase):
    """`make_event` assembles features from `dataio.SURFACE_SPEC`, not by hand.

    A hand-written feature dict is the thing that drifts: the pipeline grows a
    column, the fixture does not, and every test that uses it keeps passing
    against a shape the production code no longer sees. Building from the spec
    turns that drift into a `KeyError` at fixture-construction time, which is
    what the last test in this class checks.
    """

    def test_every_surface_can_be_built(self) -> None:
        for surface in EVENT_TYPES:
            with self.subTest(surface=surface):
                event = make_event(surface, 12_345.0)
                self.assertEqual(event.event_type, surface)
                self.assertEqual(event.amount_inr, 12_345.0)

    def test_the_fixture_declares_every_column_the_spec_does(self) -> None:
        for surface in EVENT_TYPES:
            expected = (set(dataio.SURFACE_SPEC[surface]["feature_cols"])
                        | set(dataio.CUSTOMER_COLUMNS))
            with self.subTest(surface=surface):
                self.assertEqual(set(make_event(surface).features), expected)

    @needs_data
    def test_the_fixture_carries_the_same_columns_as_a_real_event(self) -> None:
        """The check that would actually catch drift, run against the real CSVs.

        The test above compares the fixture to the spec, and the loader builds
        real events from the same spec, so both could be wrong together if the
        spec itself fell behind the generated data. This one compares the
        fixture to an event that came off disk.
        """
        for surface in EVENT_TYPES:
            real = dataio.load_events(surface, None, 1)
            if not real:
                self.skipTest(f"no rows for {surface}")
            with self.subTest(surface=surface):
                self.assertEqual(set(make_event(surface).features),
                                 set(real[0].features))

    def test_a_new_column_with_no_test_value_raises(self) -> None:
        """Adding a signal to the pipeline must break the fixture loudly.

        The alternative — filling the gap with `None` or skipping the column —
        is how a fixture ends up testing a shape nothing produces. The error
        names the file and the dict to add to, because the person who hits this
        is mid-way through adding a feature and does not want a puzzle.
        """
        spec = dataio.SURFACE_SPEC[PAYMENT_FAILURE]["feature_cols"]
        spec.append("brand_new_signal")
        try:
            with self.assertRaises(KeyError) as caught:
                make_event(PAYMENT_FAILURE)
            message = str(caught.exception)
            self.assertIn("brand_new_signal", message)
            self.assertIn("FEATURE_VALUES", message)
        finally:
            spec.remove("brand_new_signal")
        # And the fixture works again once the column is gone, so the test
        # above cannot pass by having left the spec broken.
        make_event(PAYMENT_FAILURE)

    def test_no_fixture_value_is_dead(self) -> None:
        """Every entry in FEATURE_VALUES is a column some surface declares.

        A leftover entry is harmless but misleading: it reads as documentation
        of a field the system uses, and it is the first thing someone copies
        when adding a surface.
        """
        declared = set(dataio.CUSTOMER_COLUMNS)
        for spec in dataio.SURFACE_SPEC.values():
            declared |= set(spec["feature_cols"])
        self.assertEqual(set(FEATURE_VALUES) - declared, set())

    def test_no_oracle_column_is_in_the_fixture(self) -> None:
        """Outcome labels are quarantined, and the fixtures must not smuggle one in.

        Uses the same prefix list the feature builder screens with, so a new
        forbidden prefix covers this file automatically.
        """
        for column in FEATURE_VALUES:
            with self.subTest(column=column):
                self.assertFalse(column.startswith(F.FORBIDDEN_FEATURE_PREFIXES),
                                 f"{column} looks like ground truth")

    def test_the_amount_lands_in_the_column_the_surface_uses(self) -> None:
        for surface, column in ((PAYMENT_FAILURE, "amount"),
                                (CHECKOUT_ABANDONMENT, "cart_value"),
                                (OVERDUE_RECEIVABLE, "invoice_amount")):
            with self.subTest(surface=surface):
                event = make_event(surface, 7_500.0)
                self.assertEqual(event.features[column], 7_500.0)
                self.assertEqual(event.amount_inr, 7_500.0)

    def test_the_hour_reaches_both_places_that_read_it(self) -> None:
        """Quiet hours are checked from the attribute; the models read the column.

        If these two disagreed, a fixture could be inside quiet hours for the
        guardrail and outside it for the model, and a test aimed at one would
        silently be testing neither.
        """
        event = make_event(PAYMENT_FAILURE, occurred_at_hour=3)
        self.assertEqual(event.occurred_at_hour, 3)
        self.assertEqual(event.features["hour_of_day"], 3)

    def test_an_override_wins(self) -> None:
        event = make_event(PAYMENT_FAILURE, features={"retry_count": 4,
                                                     "decline_code": "expired_card"})
        self.assertEqual(event.features["retry_count"], 4)
        self.assertEqual(event.features["decline_code"], "expired_card")

    def test_the_default_event_id_is_stable(self) -> None:
        """Two fixtures built the same way are the same event.

        Idempotency and ledger tests depend on this: an event id that varied
        per call would make a duplicate impossible to construct.
        """
        self.assertEqual(make_event(PAYMENT_FAILURE).event_id,
                         make_event(PAYMENT_FAILURE).event_id)


class TestTheTwoCustomerStandInsAreOpposites(unittest.TestCase):
    """`permissive_customer` and `default_customer` must never be confused.

    One is a test scaffold that starts from "everything is allowed" so a
    guardrail test can switch off exactly one thing and prove which rule did
    the blocking. The other is the production stand-in for a missing customer
    record, and it is pessimistic on purpose: ignorance must not widen what the
    agent may do. Using the first in production code, or the second in a
    guardrail test, would be a serious mistake in opposite directions — so the
    difference is asserted rather than left to the docstring.
    """

    def test_the_fixture_customer_consents_to_everything(self) -> None:
        customer = permissive_customer()
        for channel in ("email", "whatsapp", "sms"):
            with self.subTest(channel=channel):
                self.assertTrue(customer.has_consent(channel))
        self.assertFalse(customer.dnd_flagged)
        self.assertEqual(customer.contacts_last_7d, 0)

    def test_the_production_stand_in_consents_to_nothing(self) -> None:
        customer = default_customer()
        for channel in ("email", "whatsapp", "sms"):
            with self.subTest(channel=channel):
                self.assertFalse(customer.has_consent(channel))
        self.assertEqual(customer.estimated_annual_value_inr, 0.0)
        self.assertEqual(customer.repeat_purchase_probability, 0.0)

    def test_they_disagree_on_every_consent_flag(self) -> None:
        fixture, production = permissive_customer(), default_customer()
        for flag in ("email_consent", "whatsapp_consent", "sms_consent"):
            with self.subTest(flag=flag):
                self.assertNotEqual(getattr(fixture, flag), getattr(production, flag))

    def test_an_override_can_switch_exactly_one_thing_off(self) -> None:
        customer = permissive_customer(whatsapp_consent=False)
        self.assertFalse(customer.has_consent("whatsapp"))
        self.assertTrue(customer.has_consent("email"))

    def test_the_fixture_segment_is_one_the_system_recognises(self) -> None:
        """Otherwise the narrator's fallback would be exercised by every test.

        `build_fact_sheet` renders `segment` through `SEGMENTS` and falls back
        to "unspecified" for anything else. A fixture with an unrecognised
        segment would take that path silently, and the fallback would never be
        tested against a case that is supposed to pass through.
        """
        self.assertIn(permissive_customer().segment, SEGMENTS)
        self.assertIn(default_customer().segment, SEGMENTS)

    def test_the_feature_row_and_the_profile_agree(self) -> None:
        """The same fourteen customer values are written out twice; they must match.

        `FEATURE_VALUES` feeds the model-facing feature dict and
        `permissive_customer` feeds the guardrail-facing profile. When they
        disagree, a test can be blocked by a consent flag the model never saw,
        and the failure looks like a guardrail bug.
        """
        customer = permissive_customer()
        for column in dataio.CUSTOMER_COLUMNS:
            with self.subTest(column=column):
                self.assertEqual(getattr(customer, column), FEATURE_VALUES[column])


class TestRankedOptionsPricesEveryArm(unittest.TestCase):
    """The candidate set, priced with the model held constant.

    Flat probabilities are the point: with every arm given the same chance of
    recovering, any difference that shows up between options is the work of the
    pricing and the guardrails, which are the things the tests using this
    helper are about.
    """

    def test_every_action_the_surface_allows_is_priced(self) -> None:
        for surface in EVENT_TYPES:
            with self.subTest(surface=surface):
                actions = {s.candidate.action
                           for s in ranked_options(make_event(surface, 20_000.0))}
                self.assertEqual(actions, set(ACTIONS_BY_SURFACE[surface]))

    def test_no_action_from_another_surface_appears(self) -> None:
        """A receivables invoice can never be handed a "retry the card" option."""
        for surface in EVENT_TYPES:
            others = set()
            for other in EVENT_TYPES:
                if other != surface:
                    others |= set(ACTIONS_BY_SURFACE[other])
            exclusive = others - set(ACTIONS_BY_SURFACE[surface])
            actions = {s.candidate.action
                       for s in ranked_options(make_event(surface, 20_000.0))}
            with self.subTest(surface=surface):
                self.assertEqual(actions & exclusive, set())

    def test_the_baseline_is_the_one_the_caller_asked_for(self) -> None:
        """The regression test for the bug in this file's docstring.

        `BASELINE_ACTION` collides with a real candidate key, so writing the
        two probability maps in the wrong order silently sets the baseline to
        the treatment probability. Both halves are asserted: the baseline is
        0.30 on every option, and the uplift is the difference — not zero.
        """
        ranked = ranked_options(make_event(PAYMENT_FAILURE, 20_000.0),
                                p=0.55, p_baseline=0.30)
        self.assertEqual(BASELINE_ACTION, "do_nothing",
                         "if this changes, the collision this guards may be gone")
        for scored in ranked:
            with self.subTest(action=scored.candidate.action):
                self.assertAlmostEqual(scored.p_recover_baseline, 0.30, places=6)
        acting = [s for s in ranked if s.candidate.action != "do_nothing"]
        self.assertTrue(any(s.uplift > 0 for s in acting),
                        "every uplift came out zero — the baseline was overwritten")

    def test_doing_nothing_scores_exactly_zero(self) -> None:
        """Every money field is incremental against inaction, so this is the origin.

        It is what makes `expected_net_recovery_inr` readable as "rupees better
        than leaving it alone", and it is worth pinning here because a pricing
        change that broke it would make every comparison in the suite mean
        something slightly different.
        """
        for surface in EVENT_TYPES:
            ranked = ranked_options(make_event(surface, 20_000.0))
            nothing = [s for s in ranked if s.candidate.action == "do_nothing"]
            with self.subTest(surface=surface):
                self.assertEqual(len(nothing), 1)
                self.assertEqual(nothing[0].expected_net_recovery_inr, 0.0)
                self.assertEqual(nothing[0].uplift, 0.0)

    def test_the_result_is_sorted_by_expected_net_recovery(self) -> None:
        values = [s.expected_net_recovery_inr
                  for s in ranked_options(make_event(PAYMENT_FAILURE, 20_000.0))]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_a_fraud_probability_reprices_in_both_directions(self) -> None:
        """The helper's `p_fraud` has to reach the pricing, or tests using it lie.

        The sign convention makes this less obvious than it looks, and the first
        version of this test got it backwards. Every money field is incremental
        against doing nothing, so at `p_fraud=0.9` the actions that would push
        the payment through pick up a positive expected chargeback cost, while
        `stop_and_flag_fraud` gets a *negative* one — the chargeback it avoids
        is a saving relative to inaction. Both are correct and they point
        opposite ways, so both are asserted.

        The consequence is the part worth having: with fraud likely, the
        arithmetic alone drives every retry below zero and lifts stopping to the
        top of the list, before any guardrail is consulted. The guardrail that
        refuses those retries outright is in `test_policy.py` — two independent
        mechanisms, which is the pattern this project uses for anything that
        moves money.
        """
        clean = {s.candidate.action: s for s in
                 ranked_options(make_event(PAYMENT_FAILURE, 50_000.0), p_fraud=0.0)}
        risky = {s.candidate.action: s for s in
                 ranked_options(make_event(PAYMENT_FAILURE, 50_000.0), p_fraud=0.9)}

        for action in ("immediate_retry", "delayed_retry", "prompt_new_payment_method"):
            with self.subTest(action=action):
                self.assertEqual(clean[action].expected_chargeback_cost_inr, 0.0)
                self.assertGreater(risky[action].expected_chargeback_cost_inr, 0.0)
                self.assertLess(risky[action].expected_net_recovery_inr,
                                clean[action].expected_net_recovery_inr)
                self.assertLess(risky[action].expected_net_recovery_inr, 0.0,
                                "chasing a likely-fraudulent payment must price "
                                "worse than leaving it alone")

        stop = "stop_and_flag_fraud"
        self.assertEqual(clean[stop].expected_chargeback_cost_inr, 0.0)
        self.assertLess(risky[stop].expected_chargeback_cost_inr, 0.0)
        self.assertGreater(risky[stop].expected_net_recovery_inr,
                           clean[stop].expected_net_recovery_inr)
        self.assertEqual(max(risky.values(),
                             key=lambda s: s.expected_net_recovery_inr
                             ).candidate.action, stop)

    def test_doing_nothing_is_unmoved_by_the_fraud_probability(self) -> None:
        """It is the origin the other options are measured from, so it cannot move."""
        for p_fraud in (0.0, 0.5, 0.9):
            ranked = ranked_options(make_event(PAYMENT_FAILURE, 50_000.0),
                                    p_fraud=p_fraud)
            nothing = next(s for s in ranked if s.candidate.action == "do_nothing")
            with self.subTest(p_fraud=p_fraud):
                self.assertEqual(nothing.expected_net_recovery_inr, 0.0)
                self.assertEqual(nothing.expected_chargeback_cost_inr, 0.0)


class TestMakeDecisionMirrorsTheAgent(AuditCase):
    """A fully decided case, assembled the way `Toolbelt.run_plan` assembles one.

    Minus the two model calls, which is what makes it usable without trained
    weights — and, more usefully, what makes the decision's contents something
    a test states rather than something it has to discover.
    """

    def test_the_chosen_action_is_one_the_surface_allows(self) -> None:
        for surface in EVENT_TYPES:
            decision = make_decision(make_event(surface, 20_000.0))
            with self.subTest(surface=surface):
                self.assertIn(decision.action, ACTIONS_BY_SURFACE[surface])

    def test_the_chosen_option_is_in_the_considered_set(self) -> None:
        decision = make_decision()
        self.assertIn(decision.chosen.candidate.action,
                      {s.candidate.action for s in decision.considered})
        self.assertGreaterEqual(len(decision.considered), 2)

    def test_the_root_cause_comes_from_the_context_the_test_passed(self) -> None:
        decision = make_decision(ctx=clear_context(root_cause="expired_card",
                                                   root_cause_confidence=0.77))
        self.assertEqual(decision.root_cause, "expired_card")
        self.assertAlmostEqual(decision.root_cause_confidence, 0.77, places=6)

    def test_a_gated_decision_always_carries_a_reason(self) -> None:
        """Conditional, and still worth asserting.

        Whether the gate fires depends on the amount and the policy, which is
        `test_policy.py`'s subject. What must hold for any amount is that a
        decision withheld for approval says why — a queue of unexplained items
        is not a review workflow.
        """
        for amount in (500.0, 20_000.0, 250_000.0, 5_000_000.0):
            decision = make_decision(make_event(PAYMENT_FAILURE, amount))
            with self.subTest(amount=amount):
                if decision.requires_human_approval:
                    self.assertTrue(decision.approval_reason.strip())

    def test_the_chosen_option_was_allowed(self) -> None:
        for surface in EVENT_TYPES:
            decision = make_decision(make_event(surface, 20_000.0))
            with self.subTest(surface=surface):
                self.assertTrue(decision.chosen.allowed)
                self.assertEqual(decision.chosen.blocked_by, [])

    def test_deciding_writes_no_audit_record(self) -> None:
        """The helper builds a decision; recording one is the agent's job.

        Counted as records rather than trusted to the size guard in `AuditCase`,
        for the same reason as everywhere else in this suite: a write followed
        by a rewrite of the same length would pass a byte comparison.
        """
        before = len(A.AuditStore())
        for surface in EVENT_TYPES:
            make_decision(make_event(surface, 20_000.0))
        self.assertEqual(len(A.AuditStore()), before)
        self.assertEqual(self.audit.rows(), [],
                         "the temp trail should be empty too — nothing here writes")


class TestTheShippedTrailGuardActuallyFires(AuditCase):
    """The guard in `AuditCase` is load-bearing, so it gets a test of its own.

    It exists because the suite once appended 576 records to
    `data/audit/decisions.jsonl` and nothing noticed for a whole run. A guard
    that had quietly stopped working would put the suite back in exactly that
    position, and the failure would again be invisible — so rather than trust
    it, this feeds it a false "before" reading and checks that it fails. No
    file is touched: the mismatch is manufactured, which is the only safe way
    to test a check whose real trigger is damage.
    """

    def test_a_mismatch_is_reported_as_a_failure(self) -> None:
        class Probe(AuditCase):
            def runTest(self) -> None:  # pragma: no cover - never run
                pass

        probe = Probe()
        with self.assertRaises(probe.failureException) as caught:
            probe._assert_shipped_trail_untouched(0, 0, 0)
        message = str(caught.exception)
        self.assertIn("shipped evidence", message)
        self.assertIn("truthiness-based fallback", message,
                      "the failure should name the cause it was written for")

    def test_the_temp_trail_is_not_the_shipped_one(self) -> None:
        from src import config as C
        self.assertNotEqual(self.audit.store.path, C.AUDIT_LOG_PATH)
        self.assertNotEqual(self.audit.runs.path, C.RUN_INDEX_PATH)

    def test_an_unchanged_trail_passes(self) -> None:
        """The other half: the guard must not fail every test it is attached to."""
        from .helpers import _shipped_trail_sizes
        self._assert_shipped_trail_untouched(*_shipped_trail_sizes())


if __name__ == "__main__":
    unittest.main()
