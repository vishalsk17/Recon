"""The guardrails, and the config file that parameterises them.

Two groups, and the split matters.

The first group tests the *config loader*, because several of the safety
properties in this project are enforced by refusing to start. A policy file
that permits eight automated retries against one card is not a policy this
agent will run under, and the check for that belongs with the loader rather
than at the point of use — by the time a sweep is running, a bad limit has
already been read.

The second group tests the guardrails one variable at a time. Each test starts
from `permissive_customer()`, where nothing is blocked, switches off exactly
one thing, and asserts both that the action is refused *and* that the refusal
names the right reason. Asserting only `allowed is False` would pass just as
happily if some unrelated guardrail were doing the blocking, which would leave
the guardrail under test dead and the suite green — the failure mode this
project is least able to afford, since the audit record shows the reason to a
reviewer who will act on it.
"""

from __future__ import annotations


import os
import tempfile
import unittest

import yaml

from src import config as C
from src.adapters import Dispatcher, unrouted_actions
from src.economics import Economics
from src.guardrails import Guardrails, SweepBudget, in_quiet_hours
from src.schemas import (
    ALL_ACTIONS, AUTOMATED_REMINDER, CHECKOUT_ABANDONMENT, CandidateAction,
    DELAYED_RETRY, DO_NOTHING, ESCALATE_TO_COLLECTIONS, EVENT_TYPES,
    HUMAN_GATED_ACTIONS, IMMEDIATE_RETRY, OFFER_BOUNDED_DISCOUNT,
    OUTREACH_ACTIONS, OVERDUE_RECEIVABLE, PAYMENT_FAILURE,
    REQUEST_HUMAN_REVIEW, SEND_REMINDER_WHATSAPP,
)

from .helpers import clear_context, make_event, permissive_customer, ranked_options


class TestActionRoutingIsTotal(unittest.TestCase):
    """Every action in the vocabulary reaches exactly one adapter.

    This is what makes "the agent cannot invent a new intervention" a
    structural claim rather than a hope. Two failure directions, and both are
    checked: an action with no adapter would raise at execution time deep
    inside a sweep, and an adapter handling something outside the vocabulary
    would be a capability nothing in the decision path knows exists.
    """

    def test_no_action_is_unrouted(self) -> None:
        self.assertEqual(unrouted_actions(), [],
                         "actions with no adapter — src/adapters/__init__.py")

    def test_no_adapter_handles_an_unknown_action(self) -> None:
        extras = set(Dispatcher().routing_table) - set(ALL_ACTIONS)
        self.assertEqual(extras, set(),
                         f"adapters route actions absent from ALL_ACTIONS: {extras}")

    def test_routing_is_unambiguous(self) -> None:
        """Two adapters claiming one action would make behaviour order-dependent.

        `Dispatcher.__init__` raises on a collision, so constructing it at all
        is the assertion. Stated as a test anyway, because the constructor
        raising is the mechanism and a test is what notices if it stops.
        """
        table = Dispatcher().routing_table
        self.assertEqual(len(table), len(ALL_ACTIONS))

    def test_do_nothing_has_a_real_adapter(self) -> None:
        """Inaction is a decision and must produce an execution record."""
        self.assertEqual(Dispatcher().adapter_for(DO_NOTHING).name, "none")


class TestConfigRefusesAbusiveSettings(unittest.TestCase):
    """Some configurations are refused at load, not warned about.

    The retry limits are the ones that matter most: a high automated attempt
    count against a single instrument is the signature of card testing, and no
    amount of downstream care makes it acceptable. So the loader refuses, and
    the agent cannot be run in that configuration by editing one number.
    """

    def _load_variant(self, mutate) -> None:
        """Write a modified copy of the shipped policy and load it.

        A copy, never the real file: a test that edited config/policy.yaml in
        place and then failed would leave the repo holding a policy nobody
        wrote.
        """
        with open(C.POLICY_PATH, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        mutate(raw)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                         encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh)
            path = fh.name
        self.addCleanup(os.unlink, path)
        C.load_config(path)

    def test_more_than_five_attempts_per_payment_is_refused(self) -> None:
        for attempts in (6, 10, 50):
            with self.assertRaises(ValueError) as caught:
                self._load_variant(
                    lambda raw: raw["retries"].__setitem__(
                        "max_attempts_per_payment", attempts))
            self.assertIn("card testing", str(caught.exception).lower(),
                          "the refusal should say why, not just that")

    def test_zero_attempts_is_refused(self) -> None:
        """Nonsense in the other direction still has to fail loudly.

        A cap of zero would silently disable retries everywhere rather than
        raise, and "the recovery agent stopped recovering" is a bug that could
        run unnoticed for a long time.
        """
        with self.assertRaises(ValueError):
            self._load_variant(
                lambda raw: raw["retries"].__setitem__("max_attempts_per_payment", 0))

    def test_sub_hour_retry_spacing_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._load_variant(
                lambda raw: raw["retries"].__setitem__("min_hours_between_attempts", 0))
        self.assertIn("abuse pattern", str(caught.exception).lower())

    def test_out_of_range_discount_cap_is_refused(self) -> None:
        for pct in (-1, 101, 1000):
            with self.assertRaises(ValueError):
                self._load_variant(
                    lambda raw: raw["limits"].__setitem__("max_discount_pct", pct))

    def test_a_missing_section_is_refused(self) -> None:
        """Not defaulted. A missing `limits` block must not mean "no limits"."""
        for section in ("limits", "retries", "contact", "receivables", "llm"):
            with self.assertRaises(ValueError) as caught:
                self._load_variant(lambda raw: raw.pop(section))
            self.assertIn(section, str(caught.exception))

    def test_a_missing_individual_limit_is_refused(self) -> None:
        for key in ("max_auto_approve_amount_inr", "max_discount_pct",
                    "min_confidence_to_act", "min_expected_net_recovery_inr"):
            with self.assertRaises(ValueError) as caught:
                self._load_variant(lambda raw: raw["limits"].pop(key))
            self.assertIn(key, str(caught.exception))


class TestTheShippedPolicyIsTheOneDescribed(unittest.TestCase):
    """The documented numbers and the file agree, and cannot be edited at runtime."""

    def setUp(self) -> None:
        self.cfg = C.load_config()

    def test_it_loads_and_the_documented_limits_match(self) -> None:
        limits = self.cfg["limits"]
        self.assertEqual(float(limits["max_auto_approve_amount_inr"]), 50_000.0)
        self.assertEqual(float(limits["max_discount_pct"]), 10.0)
        self.assertEqual(float(limits["min_confidence_to_act"]), 0.55)
        self.assertEqual(int(self.cfg["retries"]["max_attempts_per_payment"]), 3)
        self.assertEqual(int(self.cfg["contact"]["max_contacts_per_customer_per_7d"]), 2)

    def test_dry_run_is_the_default(self) -> None:
        """The shipped config must never be the one that acts for real."""
        self.assertIs(self.cfg["execution"]["dry_run"], True)

    def test_the_loaded_config_cannot_be_mutated(self) -> None:
        """A limit any module can widen at runtime is not a limit.

        Nested, because freezing only the top level would leave every actual
        number writable — and the numbers are the whole point.
        """
        with self.assertRaises(TypeError):
            self.cfg["limits"]["max_discount_pct"] = 100  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.cfg["retries"]["max_attempts_per_payment"] = 99  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.cfg["execution"]["dry_run"] = False  # type: ignore[index]

    def test_every_action_has_a_price(self) -> None:
        """An unpriced action would look free, and free options win contests."""
        economics = Economics(self.cfg)
        for action in ALL_ACTIONS:
            economics.action_cost(action)  # raises KeyError if unpriced

    def test_there_is_no_halt_file_in_the_checkout(self) -> None:
        """A stray kill switch would make every selection test trivially pass.

        `select` short-circuits to do_nothing when the halt file exists. If one
        had been left behind by a previous run, the whole group below would go
        green while asserting nothing at all — so it is worth one test.
        """
        halt = os.path.join(C.PROJECT_ROOT, self.cfg["execution"]["kill_switch_file"])
        self.assertFalse(os.path.exists(halt), f"remove {halt} before running the suite")


class GuardrailCase(unittest.TestCase):
    """Screens one action against one customer, so the reason can be asserted."""

    def setUp(self) -> None:
        self.cfg = C.load_config()
        self.economics = Economics(self.cfg)
        self.guardrails = Guardrails(self.cfg, self.economics)

    def screen(self, action: str, *, event=None, ctx=None, customer=None,
               features=None, amount: float = 20_000.0, hour: int = 12,
               surface: str = PAYMENT_FAILURE, discount_pct: float = 0.0,
               delay_hours: int = 0, budget=None):
        event = event or make_event(surface, amount, customer=customer,
                                    features=features, occurred_at_hour=hour)
        ctx = ctx or clear_context()
        if budget is not None:
            ctx.budget = budget
        candidate = CandidateAction(action=action, discount_pct=discount_pct,
                                    delay_hours=delay_hours)
        scored = self.economics.score(event, candidate, 0.55, 0.30)
        return self.guardrails.screen(event, scored, ctx)

    def assertBlockedBecause(self, scored, fragment: str) -> None:
        self.assertFalse(scored.allowed,
                         f"{scored.candidate.action} was permitted; expected a block "
                         f"mentioning {fragment!r}")
        joined = " | ".join(scored.blocked_by).lower()
        self.assertIn(fragment.lower(), joined,
                      f"blocked, but for the wrong reason: {scored.blocked_by}")

    def assertPermitted(self, scored) -> None:
        self.assertTrue(scored.allowed,
                        f"{scored.candidate.action} was blocked by {scored.blocked_by}")


class TestConsentAndContactDiscipline(GuardrailCase):
    """Nothing reaches a customer who has not agreed to be reached."""

    def test_dnd_blocks_every_outreach_action(self) -> None:
        dnd = permissive_customer(dnd_flagged=True)
        for action in sorted(OUTREACH_ACTIONS):
            surface = (OVERDUE_RECEIVABLE if action.startswith("automated_")
                       else PAYMENT_FAILURE)
            self.assertBlockedBecause(
                self.screen(action, customer=dnd, surface=surface), "DND")

    def test_a_missing_channel_consent_blocks_only_that_channel(self) -> None:
        """The check has to be per-channel, or it is not a consent check.

        A customer who opted into email and not WhatsApp is a normal customer,
        and blocking both would be over-broad in a way nobody would notice —
        the agent would simply appear less effective.
        """
        no_whatsapp = permissive_customer(whatsapp_consent=False)
        self.assertBlockedBecause(
            self.screen(SEND_REMINDER_WHATSAPP, customer=no_whatsapp), "whatsapp")
        self.assertPermitted(self.screen(AUTOMATED_REMINDER, customer=no_whatsapp,
                                         surface=OVERDUE_RECEIVABLE))

    def test_absent_consent_is_read_as_no(self) -> None:
        """Email is not on `consent_required_channels`, and still blocks.

        The two branches in `_check_consent` reach the same answer by different
        routes; this pins the second one, which is the one that says "we have
        no record" must never be read as "probably fine".
        """
        self.assertNotIn("email", set(self.cfg["contact"]["consent_required_channels"]))
        self.assertBlockedBecause(
            self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                        customer=permissive_customer(email_consent=False)),
            "no consent recorded for email")

    def test_the_seven_day_contact_cap_blocks(self) -> None:
        cap = int(self.cfg["contact"]["max_contacts_per_customer_per_7d"])
        self.assertBlockedBecause(
            self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                        customer=permissive_customer(contacts_last_7d=cap)),
            "contact cap reached")

    def test_contacts_issued_earlier_in_the_same_sweep_count(self) -> None:
        """The cap is a property of the customer, not of one decision.

        Without this, a sweep could send the second and third message itself:
        every individual decision would see `contacts_last_7d=1` and permit
        one more. The historical count and the in-flight count have to be added
        together, and this is the only test that would catch them being read
        separately.
        """
        cap = int(self.cfg["contact"]["max_contacts_per_customer_per_7d"])
        customer = permissive_customer(contacts_last_7d=cap - 1)
        budget = SweepBudget(contacts_issued={customer.customer_id: 1})
        self.assertBlockedBecause(
            self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                        customer=customer, budget=budget),
            "including this sweep")

    def test_the_minimum_gap_between_contacts_blocks(self) -> None:
        gap = float(self.cfg["contact"]["min_hours_between_contacts"])
        self.assertBlockedBecause(
            self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                        customer=permissive_customer(hours_since_last_contact=gap - 1)),
            "minimum gap")

    def test_quiet_hours_defer_rather_than_block(self) -> None:
        """A 3am reminder is badly timed, not illegitimate.

        Blocking it would quietly lose recoverable revenue and look like
        caution. So the action survives with a delay, and lands at the hour
        quiet hours end.
        """
        end = int(self.cfg["contact"]["quiet_hours_end"])
        for hour in (22, 23, 0, 3, 6):
            scored = self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                                 hour=hour)
            self.assertPermitted(scored)
            self.assertEqual((hour + scored.candidate.delay_hours) % 24, end,
                             f"an outreach at {hour}:00 was rescheduled to "
                             f"{(hour + scored.candidate.delay_hours) % 24}:00")

    def test_quiet_hours_leave_a_daytime_send_alone(self) -> None:
        for hour in (9, 12, 17, 20):
            scored = self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                                 hour=hour)
            self.assertEqual(scored.candidate.delay_hours, 0,
                             f"{hour}:00 is inside the permitted window")

    def test_the_quiet_hours_window_wraps_midnight(self) -> None:
        """21:00-09:00 is not `start <= h < end`, and getting it backwards
        would invert the rule — sending only at night."""
        start, end = 21, 9
        for hour in (21, 22, 23, 0, 4, 8):
            self.assertTrue(in_quiet_hours(hour, start, end), f"{hour}:00")
        for hour in (9, 10, 15, 20):
            self.assertFalse(in_quiet_hours(hour, start, end), f"{hour}:00")


class TestRetryDiscipline(GuardrailCase):
    """Retries are where a recovery agent most resembles an attacker."""

    def test_a_never_retry_root_cause_blocks(self) -> None:
        for cause in self.cfg["retries"]["never_retry_root_causes"]:
            self.assertBlockedBecause(
                self.screen(DELAYED_RETRY, ctx=clear_context(root_cause=cause)),
                "never-retry list")

    def test_the_fraud_posterior_blocks_even_when_fraud_is_not_the_top_label(self) -> None:
        """The check reads the distribution, not its argmax.

        {insufficient_funds: 0.41, fraud_suspected: 0.39} predicts a cause that
        is not on the never-retry list, so a top-1 check would permit the
        retry. The model is very nearly as suspicious as it is confident, and
        retrying a card with a two-in-five chance of being stolen is not a
        recovery strategy.
        """
        ctx = clear_context(
            root_cause="insufficient_funds",
            root_cause_confidence=0.41,
            root_cause_distribution={"insufficient_funds": 0.41, "fraud_suspected": 0.39},
        )
        scored = self.screen(DELAYED_RETRY, ctx=ctx)
        self.assertBlockedBecause(scored, "exceeds the")
        self.assertIn("posterior", " ".join(scored.blocked_by).lower(),
                      "the reason should explain that the argmax was not what decided")

    def test_the_fraud_ceiling_is_where_the_config_says(self) -> None:
        cap = float(self.cfg["retries"]["max_fraud_probability_for_retry"])
        just_under = clear_context(root_cause_distribution={"fraud_suspected": cap - 0.01})
        just_over = clear_context(root_cause_distribution={"fraud_suspected": cap + 0.01})
        self.assertPermitted(self.screen(DELAYED_RETRY, ctx=just_under))
        self.assertBlockedBecause(self.screen(DELAYED_RETRY, ctx=just_over), "P(fraud)")

    def test_the_per_payment_attempt_cap_blocks(self) -> None:
        cap = int(self.cfg["retries"]["max_attempts_per_payment"])
        self.assertPermitted(
            self.screen(DELAYED_RETRY, features={"retry_count": cap - 1}))
        self.assertBlockedBecause(
            self.screen(DELAYED_RETRY, features={"retry_count": cap}),
            f"cap is {cap}")

    def test_an_immediate_retry_after_a_prior_attempt_breaches_spacing(self) -> None:
        """Blocked for the spacing, not for the count.

        One prior attempt is inside the per-payment cap, so if this test ever
        started passing for the other reason the spacing rule would be dead.
        Hence asserting the reason.
        """
        scored = self.screen(IMMEDIATE_RETRY, features={"retry_count": 1})
        self.assertBlockedBecause(scored, "minimum")
        self.assertNotIn("cap is", " ".join(scored.blocked_by))

    def test_a_first_immediate_retry_is_permitted(self) -> None:
        self.assertPermitted(self.screen(IMMEDIATE_RETRY, features={"retry_count": 0}))

    def test_the_sweep_wide_retry_budget_blocks(self) -> None:
        """An aggregate cap that per-event logic cannot see.

        Every individual retry here is correctly spaced and inside its cap. The
        sweep as a whole is still a retry storm, and only the budget catches it.
        """
        cap = int(self.cfg["retries"]["max_retries_per_sweep"])
        self.assertBlockedBecause(
            self.screen(DELAYED_RETRY, budget=SweepBudget(retries_issued=cap)),
            "sweep retry budget exhausted")


class TestReceivablesRestraint(GuardrailCase):
    """Collections is the most damaging action available, so it is the most gated."""

    def test_collections_below_the_ageing_threshold_is_blocked(self) -> None:
        min_days = float(self.cfg["receivables"]["min_days_overdue_for_collections"])
        self.assertBlockedBecause(
            self.screen(ESCALATE_TO_COLLECTIONS, surface=OVERDUE_RECEIVABLE,
                        features={"days_overdue": min_days - 30}),
            f"collections requires {min_days:.0f}")

    def test_a_disputed_invoice_is_never_escalated(self) -> None:
        """Even when it is old enough. Two independent conditions, and being
        old does not resolve being wrong."""
        min_days = float(self.cfg["receivables"]["min_days_overdue_for_collections"])
        self.assertBlockedBecause(
            self.screen(ESCALATE_TO_COLLECTIONS, surface=OVERDUE_RECEIVABLE,
                        features={"days_overdue": min_days + 60,
                                  "dispute_flagged_in_ar": True}),
            "disputed")

    def test_a_never_auto_chase_cause_blocks_chasing(self) -> None:
        """A wrong invoice is fixed by correcting it, not by pressure."""
        for cause in self.cfg["receivables"]["never_auto_chase_root_causes"]:
            self.assertBlockedBecause(
                self.screen(AUTOMATED_REMINDER, surface=OVERDUE_RECEIVABLE,
                            ctx=clear_context(root_cause=cause)),
                "must not be chased automatically")


class TestDiscountsAndInaction(GuardrailCase):
    def test_a_discount_above_the_cap_is_blocked(self) -> None:
        """The second of two mechanisms.

        `build_candidates` never proposes a discount above the cap — asserted
        in test_economics — so this path is only reached by a caller
        constructing one directly. It is checked anyway, because the ranker is
        not the only thing that can produce a candidate.
        """
        cap = float(self.cfg["limits"]["max_discount_pct"])
        self.assertBlockedBecause(
            self.screen(OFFER_BOUNDED_DISCOUNT, surface=CHECKOUT_ABANDONMENT,
                        discount_pct=cap + 15),
            "exceeds the")

    def test_do_nothing_survives_when_everything_else_is_blocked(self) -> None:
        """Inaction must stay reachable, or a fully-blocked event has no answer.

        A DND customer, contacted twice already, an hour ago, with a fraud
        posterior over the ceiling: every other option is refused. do_nothing
        is permitted regardless, because it cannot be unsafe.
        """
        hostile = permissive_customer(dnd_flagged=True, email_consent=False,
                                      whatsapp_consent=False, sms_consent=False,
                                      contacts_last_7d=9, hours_since_last_contact=1.0)
        scored = self.screen(DO_NOTHING, customer=hostile,
                             ctx=clear_context(
                                 root_cause="fraud_suspected",
                                 root_cause_confidence=0.2,
                                 root_cause_distribution={"fraud_suspected": 0.9}))
        self.assertPermitted(scored)
        self.assertEqual(scored.blocked_by, [])


class TestReviewMustAddCapability(GuardrailCase):
    """Escalation has to give a person latitude the agent lacks.

    The failure this encodes was measured: review was the only action nothing
    screened, so whenever consent, DND and the never-retry rules blocked
    everything else it won by elimination — 556 of 1,844 held-out events. Most
    were cases where a human could do no more than the agent. A DND customer
    with an expired card cannot be messaged by a person either, and booking an
    analyst to discover that buries the cases that genuinely need one.
    """

    def test_blocked_when_the_agent_is_confident_and_the_amount_is_small(self) -> None:
        self.assertBlockedBecause(
            self.screen(REQUEST_HUMAN_REVIEW, amount=5_000.0,
                        ctx=clear_context(root_cause_confidence=0.95)),
            "no more latitude")

    def test_permitted_when_the_cause_is_uncertain(self) -> None:
        """Ambiguity is exactly what people are for."""
        below = float(self.cfg["limits"]["min_confidence_to_act"]) - 0.1
        scored = self.screen(REQUEST_HUMAN_REVIEW, amount=5_000.0,
                             ctx=clear_context(root_cause_confidence=below))
        self.assertPermitted(scored)
        self.assertTrue(any("uncertain" in n for n in scored.notes), scored.notes)

    def test_permitted_above_the_auto_approve_ceiling(self) -> None:
        """Size alone warrants a second pair of eyes, however confident the model."""
        ceiling = float(self.cfg["limits"]["max_auto_approve_amount_inr"])
        scored = self.screen(REQUEST_HUMAN_REVIEW, amount=ceiling + 1,
                             ctx=clear_context(root_cause_confidence=0.99))
        self.assertPermitted(scored)
        self.assertTrue(any("auto-approve ceiling" in n for n in scored.notes),
                        scored.notes)

    def test_permitted_for_a_receivable_that_needs_a_process_fix(self) -> None:
        scored = self.screen(REQUEST_HUMAN_REVIEW, surface=OVERDUE_RECEIVABLE,
                             amount=5_000.0,
                             ctx=clear_context(root_cause="dispute_pending",
                                               root_cause_confidence=0.99))
        self.assertPermitted(scored)
        self.assertTrue(any("process fix" in n for n in scored.notes), scored.notes)


class TestSelectionInvariants(GuardrailCase):
    """Properties of the whole verdict, over many events rather than one.

    `select` is where screening, pricing and the approval gates meet, and the
    interesting properties are relationships between its outputs — which is
    what a per-check test cannot see.
    """

    def _verdicts(self):
        for surface in EVENT_TYPES:
            for amount in (2_000.0, 20_000.0, 49_000.0, 51_000.0, 400_000.0):
                for confidence in (0.35, 0.6, 0.95):
                    event = make_event(surface, amount)
                    ctx = clear_context(root_cause_confidence=confidence)
                    ctx.budget.at_risk_total_inr = 5_000_000.0
                    ranked = ranked_options(event, p=0.6, cfg=self.cfg,
                                            economics=self.economics)
                    yield event, ctx, self.guardrails.select(event, ranked, ctx)

    def test_review_never_wins_when_an_automated_option_is_permitted(self) -> None:
        """Review is a fallback and a mandate, never a winner on value.

        It is priced as a haircut on the best automated option, so it can never
        be worth more than that option — but the arithmetic is not what this
        relies on. `select` excludes it from the contest outright. Two
        independent mechanisms, and this asserts the outcome both produce.
        """
        seen = 0
        for event, ctx, verdict in self._verdicts():
            automated = [s for s in verdict.considered if s.allowed
                         and s.candidate.action not in (DO_NOTHING, REQUEST_HUMAN_REVIEW)]
            if not automated:
                continue
            seen += 1
            self.assertNotEqual(
                verdict.chosen.candidate.action, REQUEST_HUMAN_REVIEW,
                f"{event.event_type} at {event.amount_inr}: review beat "
                f"{len(automated)} permitted automated option(s)")
        self.assertGreater(seen, 10, "no verdict had a permitted automated option")

    def test_the_chosen_option_is_always_one_that_was_considered(self) -> None:
        """The dashboard renders the choice out of the considered set.

        A winner constructed outside that list would render as an option nobody
        can see the reasoning for.
        """
        for event, _ctx, verdict in self._verdicts():
            self.assertIn(verdict.chosen, verdict.considered,
                          f"{event.event_type}: chosen option is not in `considered`")

    def test_a_blocked_option_always_records_why(self) -> None:
        for _event, _ctx, verdict in self._verdicts():
            for scored in verdict.considered:
                if not scored.allowed:
                    self.assertTrue(
                        scored.blocked_by,
                        f"{scored.candidate.action} is not allowed and gives no reason")

    def test_nothing_gated_is_ever_marked_auto_approvable(self) -> None:
        for _event, _ctx, verdict in self._verdicts():
            if verdict.chosen.candidate.action in HUMAN_GATED_ACTIONS:
                self.assertTrue(
                    verdict.requires_human_approval,
                    f"{verdict.chosen.candidate.action} was selected without a gate")
                self.assertTrue(verdict.approval_reason)

    def test_an_amount_above_the_ceiling_gates_the_action(self) -> None:
        ceiling = float(self.cfg["limits"]["max_auto_approve_amount_inr"])
        event = make_event(PAYMENT_FAILURE, ceiling + 100_000.0)
        ctx = clear_context()
        verdict = self.guardrails.select(
            event, ranked_options(event, p=0.8, cfg=self.cfg, economics=self.economics),
            ctx)
        if verdict.chosen.candidate.action == DO_NOTHING:
            self.skipTest("nothing was chosen for this event, so there is no gate to check")
        self.assertTrue(verdict.requires_human_approval)
        self.assertIn("auto-approve ceiling", verdict.approval_reason)

    def test_low_confidence_gates_the_action(self) -> None:
        threshold = float(self.cfg["limits"]["min_confidence_to_act"])
        event = make_event(PAYMENT_FAILURE, 20_000.0)
        ctx = clear_context(root_cause_confidence=threshold - 0.2)
        verdict = self.guardrails.select(
            event, ranked_options(event, p=0.8, cfg=self.cfg, economics=self.economics),
            ctx)
        self.assertTrue(verdict.requires_human_approval,
                        f"chose {verdict.chosen.candidate.action} unattended at "
                        f"confidence {ctx.root_cause_confidence}")
        self.assertIn("below the", verdict.approval_reason)

    def test_a_low_value_best_option_resolves_to_deliberate_inaction(self) -> None:
        """Below the floor the agent does nothing, and says that is what it did.

        Distinguishing "nothing was worth doing" from "nothing was permitted"
        matters to whoever reads the trail: the first is the agent working
        correctly, the second is a policy that may be too tight.

        The customer has to be stripped of future value to reach this branch,
        which is worth explaining rather than hiding in a fixture. On a one
        rupee cart the in-period money is nil, but the default synthetic
        customer is worth 120,000 INR a year at a 0.6 repeat probability, and
        the retention term alone prices a free reminder at around 107 INR —
        comfortably over the floor. So the agent acts, correctly. Zeroing the
        relationship is what isolates the floor. The behaviour that got in the
        way is the subject of the next test.
        """
        floor = float(self.cfg["limits"]["min_expected_net_recovery_inr"])
        worthless = permissive_customer(estimated_annual_value_inr=0.0,
                                        repeat_purchase_probability=0.0)
        event = make_event(CHECKOUT_ABANDONMENT, 1.0, customer=worthless)
        verdict = self.guardrails.select(
            event, ranked_options(event, p=0.31, cfg=self.cfg, economics=self.economics),
            clear_context())
        self.assertEqual(verdict.chosen.candidate.action, DO_NOTHING)
        self.assertFalse(verdict.requires_human_approval)
        self.assertTrue(
            any("worth only" in n for n in verdict.chosen.notes),
            f"inaction gives no reason: {verdict.chosen.notes}")
        self.assertTrue(any(f"{floor:,.0f} INR floor" in g
                            for g in verdict.guardrails_applied),
                        verdict.guardrails_applied)

    def test_retention_alone_can_justify_a_free_reminder(self) -> None:
        """A trivial cart belonging to a valuable customer is still worth an email.

        This is improvements.md item 8 reduced to a single event, and it is the
        behaviour that a purely in-period ledger gets wrong: one rupee of gross
        margin cannot pay for anything, so an agent that only counted the
        transaction would file this under "not worth it" and quietly let a
        long-standing customer walk. The retention term is what makes contact
        the right answer.

        Two halves, and the second is the one that keeps this honest: the value
        has to come from the relationship rather than the sale, *and* the agent
        must not reach for a discount to protect it. Giving away margin to
        defend a one rupee cart would be the same mistake in the opposite
        direction.
        """
        event = make_event(CHECKOUT_ABANDONMENT, 1.0)  # default: a valuable customer
        verdict = self.guardrails.select(
            event, ranked_options(event, p=0.31, cfg=self.cfg, economics=self.economics),
            clear_context())
        chosen = verdict.chosen
        self.assertNotEqual(chosen.candidate.action, DO_NOTHING,
                            "a valuable relationship was written off over one rupee")
        self.assertGreater(chosen.ltv_component_inr, chosen.gross_value_inr,
                           "the case for acting should rest on retention here")
        self.assertEqual(chosen.candidate.discount_pct, 0,
                         f"{chosen.candidate.action} gives away "
                         f"{chosen.candidate.discount_pct}% to protect a 1 INR cart")


class TestKillSwitchStopsDecisionsNotJustSends(GuardrailCase):
    """The halt file stops actions being *chosen*, not merely being sent.

    Checked in the adapter too, but that is the wrong place to rely on: a
    decision recorded as taken and then silently not executed is a trail that
    disagrees with reality. So `select` short-circuits first.

    The switch is a path joined to the project root, so pointing it at a
    temporary file both exercises the real code path and keeps the test from
    creating a HALT file in a working checkout — where, if the process died
    mid-test, it would silently halt the agent for whoever ran it next.
    """

    def _halted_cfg(self, halt_path: str):
        with open(C.POLICY_PATH, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw["execution"]["kill_switch_file"] = halt_path
        return raw

    def setUp(self) -> None:
        super().setUp()
        handle = tempfile.NamedTemporaryFile("w", prefix="HALT_", delete=False)
        handle.write("halted by tests/test_policy.py\n")
        handle.close()
        self.halt_path = handle.name
        self.addCleanup(os.unlink, self.halt_path)
        self.halted = Guardrails(self._halted_cfg(self.halt_path))

    def test_the_absolute_path_really_is_seen_as_engaged(self) -> None:
        """Guard the guard: if this were false the next two tests prove nothing."""
        self.assertTrue(C.kill_switch_engaged(self.halted.cfg))
        self.assertFalse(C.kill_switch_engaged(self.cfg))

    def test_nothing_is_chosen_while_halted(self) -> None:
        event = make_event(PAYMENT_FAILURE, 40_000.0)
        verdict = self.halted.select(event, ranked_options(event, p=0.9), clear_context())
        self.assertEqual(verdict.chosen.candidate.action, DO_NOTHING)
        self.assertFalse(verdict.requires_human_approval,
                         "a halted sweep must not queue work for a human either")
        self.assertTrue(any("HALT file present" in g
                            for g in verdict.guardrails_applied),
                        verdict.guardrails_applied)

    def test_every_other_option_is_recorded_as_refused(self) -> None:
        """Silence would be indistinguishable from an empty candidate set."""
        event = make_event(PAYMENT_FAILURE, 40_000.0)
        ranked = ranked_options(event, p=0.9)
        verdict = self.halted.select(event, ranked, clear_context())
        expected = {s.candidate.action for s in ranked
                    if s.candidate.action != DO_NOTHING}
        self.assertEqual(set(verdict.rejected_reasons), expected)
        for reason in verdict.rejected_reasons.values():
            self.assertIn("kill switch", reason.lower())


if __name__ == "__main__":
    unittest.main()
