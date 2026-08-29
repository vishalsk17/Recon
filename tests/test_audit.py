"""The audit trail: what it guarantees, and what it refuses to hold.

The trail is the deliverable this project is really built around. Every claim
in the docs about what the agent did is a claim about this file, so the tests
here are less about "the writer works" and more about the properties a reader
is entitled to assume:

  * it only ever grows, and an earlier run's records are still byte-identical
    after a later one;
  * every record hashes to its stored digest and links to its predecessor, so
    an edit or a deletion is detectable rather than merely discouraged;
  * cardholder data and contact details cannot get into it, because the write
    is refused rather than sanitised;
  * running the same sweep twice does not act twice.

The tests that need a sweep write into a throwaway directory. The shipped
trail is demo evidence whose record count and chain head are quoted in the
docs, and a test run that appended to it would make those figures wrong.
"""

from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timezone

from src import audit as A
from src import config as C
from src.agent import RecoveryAgent, idempotency_window
from src.schemas import (
    DO_NOTHING, OUTREACH_ACTIONS, REQUEST_HUMAN_REVIEW, RETRY_ACTIONS,
)

from .helpers import AuditCase, needs_audit, needs_data, needs_models


class TestTheChainDetectsTampering(AuditCase):
    """Hash chaining is only worth having if a broken chain is actually noticed.

    Each test edits a copy of a trail on disk — the store itself has no method
    that rewrites or deletes a record, so tampering has to come from outside,
    which is exactly the threat model. The interesting part is not that
    `verify_chain` returns False; it is that it says *where* and *how*.
    """

    def _three_records(self) -> None:
        for i in range(3):
            self.audit.store.append("test", {"event_id": f"evt_{i}", "n": i})

    def test_a_clean_trail_verifies(self) -> None:
        self._three_records()
        result = self.audit.store.verify_chain()
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertEqual(result["records"], 3)

    def test_an_empty_trail_verifies(self) -> None:
        """Vacuously true, and worth pinning: a fresh install must not look broken."""
        result = self.audit.store.verify_chain()
        self.assertTrue(result["ok"])
        self.assertEqual(result["records"], 0)

    def test_editing_a_record_in_place_is_detected(self) -> None:
        """The most tempting tamper: change one number and leave everything else.

        This is the case the chain exists for. Changing the amount on a decision
        record and leaving its own digest untouched breaks the record's own
        hash, and the report names the record rather than just the line.
        """
        self._three_records()
        lines = self.audit.rows()
        record = json.loads(lines[1])
        record["n"] = 999
        lines[1] = A.canonical_json(record) + "\n"
        with open(self.audit.store.path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        result = self.audit.store.verify_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at_line"], 2)
        self.assertIn("modified after it was written", result["reason"])

    def test_deleting_a_record_is_detected(self) -> None:
        """Removing a line breaks the *link*, not the record's own hash.

        Worth separating from the test above, because it is caught by a
        different check and produces a different message — and because deleting
        an inconvenient record is the likelier tamper of the two.
        """
        self._three_records()
        lines = self.audit.rows()
        del lines[1]
        with open(self.audit.store.path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        result = self.audit.store.verify_chain()
        self.assertFalse(result["ok"])
        self.assertIn("inserted or deleted", result["reason"])

    def test_re_hashing_the_tail_is_still_detected_at_the_head(self) -> None:
        """A thorough tamper has to rewrite every subsequent hash.

        Doing so produces a file that verifies internally but no longer matches
        the chain head recorded in the run index at the time — which is why
        `RunIndex.finish` stores it, and why the head is the one value worth
        publishing somewhere the agent cannot write. This test pins the
        relationship: internal consistency alone is not evidence.
        """
        self._three_records()
        head_before = self.audit.store.chain_head()

        lines = self.audit.rows()
        records = [json.loads(line) for line in lines]
        records[1]["n"] = 999
        prev = records[0]["record_hash"]
        for record in records[1:]:
            body = {k: v for k, v in record.items()
                    if k not in ("prev_hash", "record_hash")}
            record["prev_hash"] = prev
            record["record_hash"] = A.record_hash(prev, body)
            prev = record["record_hash"]
        with open(self.audit.store.path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(A.canonical_json(record) + "\n")

        rebuilt = A.AuditStore(self.audit.store.path)
        self.assertTrue(rebuilt.verify_chain()["ok"],
                        "a fully re-hashed file verifies internally — that is the point")
        self.assertNotEqual(rebuilt.chain_head(), head_before,
                            "the head must move, or the tamper would be invisible")

    def test_a_truncated_line_is_reported_as_such(self) -> None:
        """Half a line is a corrupt file, and must not be silently skipped."""
        self._three_records()
        with open(self.audit.store.path, "a", encoding="utf-8") as fh:
            fh.write('{"record_type": "test", "event_id": "evt_x"')
        with self.assertRaises(A.AuditIntegrityError) as caught:
            self.audit.store.verify_chain()
        self.assertIn("truncated or hand-edited", str(caught.exception))


class TestForbiddenFieldsCannotBeWritten(AuditCase):
    """The screen refuses the write; it does not strip the field and continue.

    Stripping would be worse than refusing. A record that silently lost a field
    is a record whose hash covers something other than what the caller meant to
    say, and the caller would never find out. So the write raises, at any depth,
    and nothing lands on disk.
    """

    def test_a_forbidden_key_at_the_top_level_raises(self) -> None:
        for key in ("email", "phone", "card_number", "cvv", "api_key", "address"):
            with self.assertRaises(A.AuditIntegrityError, msg=key):
                self.audit.store.append("test", {"event_id": "evt_1", key: "x"})

    def test_a_forbidden_key_nested_in_a_dict_raises(self) -> None:
        with self.assertRaises(A.AuditIntegrityError) as caught:
            self.audit.store.append("test", {
                "event_id": "evt_1",
                "execution_context": {"channel": "email", "recipient": {"email": "a@b.c"}},
            })
        self.assertIn("email", str(caught.exception))

    def test_a_forbidden_key_inside_a_list_raises(self) -> None:
        """`considered` is a list of dicts, so depth-through-lists is the real case."""
        with self.assertRaises(A.AuditIntegrityError):
            self.audit.store.append("test", {
                "event_id": "evt_1",
                "considered": [{"action": DO_NOTHING}, {"action": "x", "phone_number": "9"}],
            })

    def test_the_key_is_matched_case_insensitively(self) -> None:
        for key in ("Email", "EMAIL", "Card_Number", "CVV"):
            with self.assertRaises(A.AuditIntegrityError, msg=key):
                self.audit.store.append("test", {key: "x"})

    def test_a_refused_write_leaves_the_file_untouched(self) -> None:
        """Partial writes would break the chain for every later record."""
        self.audit.store.append("test", {"event_id": "evt_0"})
        before_rows, before_size = self.audit.rows(), self.audit.size()
        head_before = self.audit.store.chain_head()

        with self.assertRaises(A.AuditIntegrityError):
            self.audit.store.append("test", {"event_id": "evt_1", "email": "a@b.c"})

        self.assertEqual(self.audit.rows(), before_rows)
        self.assertEqual(self.audit.size(), before_size)
        self.assertEqual(self.audit.store.chain_head(), head_before)
        self.assertTrue(self.audit.store.verify_chain()["ok"])

    def test_a_value_that_looks_like_an_address_is_not_blocked(self) -> None:
        """The list bans keys, not values, and the docstring says so.

        Worth a test because the two are easy to confuse when reading
        FORBIDDEN_KEYS, and a reviewer's expectation should match the code.
        A decline reason that happens to contain an @ is not a leak.
        """
        record = self.audit.store.append("test", {
            "event_id": "evt_1",
            "detail": "issuer returned do_not_honour for a@b.c style alias",
        })
        self.assertIn("a@b.c", record["detail"])


class TestCanonicalSerialisation(AuditCase):
    """Hashes must not depend on incidental dict ordering."""

    def test_key_order_does_not_change_the_hash(self) -> None:
        first = A.canonical_json({"a": 1, "b": {"c": 2, "d": 3}})
        second = A.canonical_json({"b": {"d": 3, "c": 2}, "a": 1})
        self.assertEqual(first, second)

    def test_the_written_line_round_trips(self) -> None:
        record = self.audit.store.append("test", {"event_id": "evt_1", "n": 1.5})
        on_disk = json.loads(self.audit.rows()[0])
        self.assertEqual(on_disk, record)

    def test_a_records_stored_hash_covers_its_body(self) -> None:
        record = self.audit.store.append("test", {"event_id": "evt_1"})
        body = {k: v for k, v in record.items()
                if k not in ("prev_hash", "record_hash")}
        self.assertEqual(A.record_hash(record["prev_hash"], body),
                         record["record_hash"])


class TestAnEmptyStoreIsStillAStore(AuditCase):
    """The regression that made this file's own guard necessary.

    `AuditStore` defines `__len__`, which is useful — `len(store)` is the record
    count. It also makes an empty store *falsy*, and every dependency injection
    in this codebase was written as `store or AuditStore()`. The two combined
    mean that passing a brand-new store is the exact case where the injection is
    silently ignored and the production trail is used instead. That is the worst
    possible place for the failure to hide: a store is only empty on its first
    use, so anything that had already written a record worked fine.

    It cost 576 records appended to the shipped audit trail before an unrelated
    assertion noticed the temp file was empty. The trail was restorable only
    because it is append-only and `RunIndex.finish` had recorded the chain head
    from before the damage — which is a good argument for both of those designs,
    and no argument at all for leaving the bug in place.

    Fixed at every call site by testing `is not None`, and belt-and-braces by
    `__bool__` returning True. Both are asserted here, because the call-site fix
    is the real repair and `__bool__` alone would paper over a future one.
    """

    def test_an_empty_store_is_truthy(self) -> None:
        self.assertEqual(len(self.audit.store), 0)
        self.assertTrue(self.audit.store,
                        "an empty AuditStore must not be falsy, or every "
                        "`store or default` fallback discards it")

    @needs_models
    def test_the_agent_writes_where_it_was_told_to(self) -> None:
        agent = RecoveryAgent(store=self.audit.store, runs=self.audit.runs,
                              approvals=self.audit.approvals)
        self.assertIs(agent.store, self.audit.store)
        self.assertIs(agent.runs, self.audit.runs)
        self.assertIs(agent.approvals, self.audit.approvals)
        self.assertIsNot(agent.store.path, C.AUDIT_LOG_PATH)
        self.assertNotEqual(agent.store.path, C.AUDIT_LOG_PATH)

    def test_the_ledger_reads_where_it_was_told_to(self) -> None:
        """`ExecutionLedger.load(empty_store)` must not fall back to the real trail.

        This one is worse than the agent's, because it fails *open*: a ledger
        built from the wrong file reports contact history for customers the
        caller never asked about, and the frequency caps it feeds are then
        enforced against someone else's numbers.
        """
        ledger = A.ExecutionLedger.load(self.audit.store)
        self.assertEqual(ledger.executed_keys, set())
        self.assertEqual(ledger.contacts_by_customer, {})
        self.assertEqual(ledger.attempts_by_event, {})

    def test_the_approval_queue_reads_where_it_was_told_to(self) -> None:
        queue = A.ApprovalQueue(os.path.join(self.audit.dir, "approvals2.jsonl"),
                                store=self.audit.store)
        self.assertIs(queue.store, self.audit.store)
        self.assertEqual(queue.pending(), [])

    def test_the_run_scoped_lookups_read_where_they_were_told_to(self) -> None:
        self.assertEqual(A.decisions_for_run("run_nope", self.audit.store), [])
        self.assertEqual(A.executions_for_run("run_nope", self.audit.store), [])
        self.assertIsNone(A.find_decision("nope", self.audit.store))
        self.assertIsNone(A.latest_decision_for_event("nope", self.audit.store))

    def test_no_injectable_collaborator_is_falsy_when_empty(self) -> None:
        """A sweep of the classes that get injected, so the next one is caught.

        `AuditStore` was the only class in `src/` defining `__len__`, which is
        why it was the only one affected. This fails if another collaborator
        grows a length and becomes falsy when empty, instead of waiting for the
        symptom to show up as records in the wrong file.
        """
        for obj in (self.audit.store, self.audit.runs, self.audit.approvals):
            self.assertTrue(obj, f"{type(obj).__name__} is falsy when empty")


class TestIdempotencyWindows(unittest.TestCase):
    """The window decides what "already done" means, so it is policy, not plumbing."""

    def test_money_and_messages_get_a_calendar_day_window(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for action in sorted(RETRY_ACTIONS | OUTREACH_ACTIONS):
            self.assertEqual(idempotency_window(action, "run_abc"), today, action)

    def test_everything_else_is_scoped_to_the_run(self) -> None:
        """A second do_nothing record is harmless; a second charge is not."""
        for action in (DO_NOTHING, REQUEST_HUMAN_REVIEW):
            self.assertEqual(idempotency_window(action, "run_abc"), "run_abc", action)

    def test_the_key_separates_event_action_and_window(self) -> None:
        base = A.idempotency_key("evt_1", "delayed_retry", "2026-08-28")
        self.assertNotEqual(base, A.idempotency_key("evt_2", "delayed_retry", "2026-08-28"))
        self.assertNotEqual(base, A.idempotency_key("evt_1", "immediate_retry", "2026-08-28"))
        self.assertNotEqual(base, A.idempotency_key("evt_1", "delayed_retry", "2026-08-29"))
        self.assertEqual(base, A.idempotency_key("evt_1", "delayed_retry", "2026-08-28"))

    def test_decision_ids_are_deterministic(self) -> None:
        """A replay of a run must produce the same ids, or idempotency is
        uncheckable after the fact."""
        self.assertEqual(A.decision_id("run_1", "evt_1"), A.decision_id("run_1", "evt_1"))
        self.assertNotEqual(A.decision_id("run_1", "evt_1"), A.decision_id("run_2", "evt_1"))


@needs_data
@needs_models
class TestRunningTwiceIsSafe(AuditCase):
    """Two sweeps, in one throwaway trail. The properties are about the pair.

    A single sweep cannot demonstrate any of this: append-only, durable
    idempotency and cross-run frequency capping are all statements about what
    the second run does with what the first one left behind.
    """

    LIMIT = 6

    def setUp(self) -> None:
        super().setUp()
        self.agent = RecoveryAgent(store=self.audit.store, runs=self.audit.runs,
                                   approvals=self.audit.approvals)
        self.first = self.agent.run(split="test", limit_per_surface=self.LIMIT,
                                    execute=True)
        self.rows_after_first = self.audit.rows()
        # A fresh agent, because a real second run is a new process: its ledger
        # has to come from the trail on disk, not from memory carried over.
        self.second_agent = RecoveryAgent(store=self.audit.store, runs=self.audit.runs,
                                         approvals=self.audit.approvals)
        self.second = self.second_agent.run(split="test", limit_per_surface=self.LIMIT,
                                            execute=True)

    def test_the_trail_only_grows(self) -> None:
        self.assertGreater(len(self.audit.rows()), len(self.rows_after_first))

    def test_the_first_runs_records_are_byte_identical_afterwards(self) -> None:
        """Stronger than "it got longer", and the property readers rely on.

        A trail that grew while quietly rewriting an earlier line would pass a
        length check and still be worthless as evidence.
        """
        after = self.audit.rows()
        self.assertEqual(after[:len(self.rows_after_first)], self.rows_after_first)

    def test_the_chain_still_verifies_across_both_runs(self) -> None:
        result = self.audit.store.verify_chain()
        self.assertTrue(result["ok"], result.get("reason"))

    def test_the_second_run_reports_duplicates_rather_than_acting_twice(self) -> None:
        """Same day, same events: every retry and message is already done.

        This is the behaviour the demo instructions warn about — a second sweep
        on the same day shows `skipped_duplicate` — and it is worth a test
        because it looks like a failure and is the single most important thing
        the ledger buys.
        """
        counted = {
            str(r["idempotency_key"])
            for r in self.audit.store.read()
            if r.get("record_type") == A.RECORD_EXECUTION
            and r.get("run_id") == self.first.run_id
            and r.get("status") in A.ExecutionLedger.COUNTED_STATUSES
        }
        if not counted:
            self.skipTest("the first sweep consumed no allowance, so there is "
                          "nothing for the second to skip")

        repeats = [r for r in self.audit.store.read()
                   if r.get("record_type") == A.RECORD_EXECUTION
                   and r.get("run_id") == self.second.run_id
                   and str(r.get("idempotency_key")) in counted]
        self.assertTrue(repeats, "the second run reused no idempotency key at all")
        for record in repeats:
            self.assertEqual(
                record["status"], "skipped_duplicate",
                f"{record['action']} on {record['event_id']} was executed twice")

    def test_an_action_that_did_not_happen_consumes_no_allowance(self) -> None:
        """Refusals and duplicate-skips must not count against the customer.

        Otherwise re-running the sweep would exhaust everyone's contact budget
        without a single message being sent, and the agent would fall silent for
        a week for a reason no reader of the trail could reconstruct. The check
        is a count rather than a membership test, because a skipped duplicate's
        idempotency key *is* in `executed_keys` — put there by the run that
        really did the work, which is how the skip was recognised in the first
        place. What must not grow is the contact and attempt tally.
        """
        executions = [r for r in self.audit.store.read()
                      if r.get("record_type") == A.RECORD_EXECUTION]
        uncounted = [r for r in executions
                     if r.get("status") not in A.ExecutionLedger.COUNTED_STATUSES]
        if not uncounted:
            self.skipTest("neither sweep produced a refusal or a duplicate skip")

        ledger = A.ExecutionLedger.load(self.audit.store)
        counted_outreach = sum(
            1 for r in executions
            if r["action"] in OUTREACH_ACTIONS
            and r["status"] in A.ExecutionLedger.COUNTED_STATUSES)
        tallied = sum(len(v) for v in ledger.contacts_by_customer.values())
        self.assertEqual(
            tallied, counted_outreach,
            f"the ledger counts {tallied} contacts against "
            f"{counted_outreach} that actually went out")

        counted_retries = sum(
            1 for r in executions
            if r["action"] in RETRY_ACTIONS
            and r["status"] in A.ExecutionLedger.COUNTED_STATUSES)
        self.assertEqual(sum(ledger.attempts_by_event.values()), counted_retries,
                         "the ledger counts retries that were never attempted")

    def test_every_decision_has_exactly_one_execution_record(self) -> None:
        """A decision with no execution record is an action nobody can account for."""
        decisions = [r for r in self.audit.store.read()
                     if r.get("record_type") == A.RECORD_DECISION]
        executions = [r for r in self.audit.store.read()
                      if r.get("record_type") == A.RECORD_EXECUTION]
        self.assertEqual(len(decisions), len(executions))
        by_decision: dict[str, int] = {}
        for record in executions:
            by_decision[record["decision_id"]] = by_decision.get(record["decision_id"], 0) + 1
        for record in decisions:
            self.assertEqual(by_decision.get(record["decision_id"], 0), 1,
                             f"decision {record['decision_id']} has "
                             f"{by_decision.get(record['decision_id'], 0)} executions")

    def test_both_runs_are_in_the_run_index_with_their_chain_head(self) -> None:
        runs = {r["run_id"]: r for r in self.audit.runs.read()
                if r.get("chain_head")}
        for run_id in (self.first.run_id, self.second.run_id):
            self.assertIn(run_id, runs, "a finished run must record its chain head")
        self.assertEqual(runs[self.second.run_id]["chain_head"],
                         self.audit.store.chain_head())

    def test_nothing_executed_for_real(self) -> None:
        """Every sweep in this suite is a dry run, and the records say so."""
        self.assertTrue(self.first.dry_run)
        for record in self.audit.store.read():
            if record.get("record_type") in (A.RECORD_DECISION, A.RECORD_EXECUTION):
                self.assertTrue(record["dry_run"],
                                f"record {record.get('record_hash', '')[:8]} is not dry-run")


@needs_data
@needs_models
class TestBatchedAndPerEventScoringAgree(AuditCase):
    """Priming is a speed fix that must not also be a behaviour change.

    `Toolbelt.prime` scores a whole sweep in one pandas pass, which is roughly
    70ms an event faster and — more importantly — prices every event in a run
    against identical model state. Both of those are only worth having if the
    batched path produces the same numbers as the per-event path, because the
    per-event path is what a reader reproducing one decision by hand will use.
    """

    LIMIT = 5

    def test_the_two_paths_agree_to_floating_point_noise(self) -> None:
        from src import dataio
        from src.schemas import EVENT_TYPES

        agent = RecoveryAgent(store=self.audit.store, runs=self.audit.runs,
                              approvals=self.audit.approvals)
        events = []
        for surface in EVENT_TYPES:
            events.extend(dataio.load_events(surface, "test", self.LIMIT))
        self.assertTrue(events, "no events loaded")

        agent.toolbelt.clear_primed()
        per_event = {e.event_id: agent.decide(e) for e in events}

        agent.toolbelt.clear_primed()
        agent.toolbelt.prime(events)
        self.assertEqual(agent.toolbelt.primed_count, len(events))
        batched = {e.event_id: agent.decide(e) for e in events}

        for event_id, one in per_event.items():
            other = batched[event_id]
            self.assertEqual(one.chosen.candidate.action, other.chosen.candidate.action,
                             f"{event_id}: batching changed the chosen action")
            self.assertAlmostEqual(one.expected_net_recovery_inr,
                                   other.expected_net_recovery_inr, places=6,
                                   msg=f"{event_id}: ENR differs")
            self.assertEqual(one.root_cause, other.root_cause, event_id)
            self.assertAlmostEqual(one.root_cause_confidence,
                                   other.root_cause_confidence, places=9,
                                   msg=f"{event_id}: confidence differs")

    def test_priming_leaves_no_state_behind_after_clearing(self) -> None:
        """A stale cache would price a later sweep against an earlier one's state.

        `run` clears before priming for exactly this reason; the test pins the
        method it relies on.
        """
        from src import dataio

        agent = RecoveryAgent(store=self.audit.store, runs=self.audit.runs,
                              approvals=self.audit.approvals)
        events = dataio.load_events("payment_failure", "test", self.LIMIT)
        agent.toolbelt.prime(events)
        self.assertGreater(agent.toolbelt.primed_count, 0)
        agent.toolbelt.clear_primed()
        self.assertEqual(agent.toolbelt.primed_count, 0)


@needs_audit
class TestTheShippedTrailHoldsUp(unittest.TestCase):
    """Properties of the artefact that ships, not of the code that writes it.

    Everything above tests the writer. These read `data/audit/decisions.jsonl`
    as a reviewer would, because the documented figures are claims about this
    file and a writer that is correct today says nothing about a file written
    weeks ago.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.store = A.AuditStore()
        cls.records = list(cls.store.read())
        cls.decisions = [r for r in cls.records
                         if r.get("record_type") == A.RECORD_DECISION]

    def test_the_chain_verifies(self) -> None:
        result = self.store.verify_chain()
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertEqual(result["records"], len(self.records))

    def test_no_record_carries_a_forbidden_field(self) -> None:
        """Re-screening what is on disk, not what the writer would allow now.

        The forbidden list can grow. If it does, this fails on the shipped
        artefact rather than quietly applying only to future writes.
        """
        for record in self.records:
            A._screen_payload(record)

    def test_every_decision_names_the_policy_it_was_made_under(self) -> None:
        """Replayability is the whole reason both versions are stamped."""
        for record in self.decisions:
            self.assertTrue(record.get("policy_version"))
            self.assertTrue(record.get("code_version"))

    def test_review_never_won_on_value(self) -> None:
        """The selection invariant, checked against decisions actually recorded.

        `test_policy` proves `select` excludes review from the contest on
        synthetic events. This asserts the same property held over the real
        sweep: wherever the agent escalated, there was genuinely nothing it was
        permitted to do itself. If this ever fails, the 21% gating figure in
        the docs is describing something other than what it claims to.
        """
        escalated = [r for r in self.decisions if r.get("action") == REQUEST_HUMAN_REVIEW]
        for record in escalated:
            automated = [
                option for option in record.get("considered", [])
                if option.get("allowed")
                and option.get("action") not in (DO_NOTHING, REQUEST_HUMAN_REVIEW)
            ]
            self.assertEqual(
                automated, [],
                f"decision {record['decision_id']} escalated to a person while "
                f"{[o['action'] for o in automated]} were permitted")

    def test_every_recorded_decision_shows_what_it_refused(self) -> None:
        """A record with no considered set is a receipt, not evidence."""
        for record in self.decisions:
            self.assertTrue(record.get("considered"),
                            f"decision {record['decision_id']} lists no alternatives")
            self.assertIn(record["action"],
                          {o["action"] for o in record["considered"]})

    def test_a_gated_decision_always_gives_its_reason(self) -> None:
        for record in self.decisions:
            if record.get("requires_human_approval"):
                self.assertTrue(record.get("approval_reason"),
                                f"decision {record['decision_id']} is gated silently")

    def test_the_trail_is_the_size_the_docs_describe(self) -> None:
        """One rough cost figure, so the docs' ~8KB per decision claim is checked.

        Loose bounds on purpose: the point is to catch the record growing by an
        order of magnitude — an embedded model output, a whole feature frame —
        not to pin a byte count.
        """
        size = os.path.getsize(C.AUDIT_LOG_PATH)
        per_record = size / max(1, len(self.records))
        self.assertLess(per_record, 32_000,
                        f"records average {per_record:,.0f} bytes; something large "
                        f"is being embedded in the trail")
        self.assertGreater(per_record, 200,
                           "records are suspiciously small for decision evidence")


if __name__ == "__main__":
    unittest.main()
