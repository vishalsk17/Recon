"""
Append-only, hash-chained decision record.

This module is the answer to improvement #7: turn a log into evidence. The
distinction matters. A log tells you what a program printed. Evidence has to
survive someone asking three uncomfortable questions:

  1. *What exactly did you decide, and against which rules?*  Every record
     carries the full ranked action set — including the options that were
     blocked and why — plus the policy and code versions in force at the
     time. A decision from six months ago can be replayed against the rules
     that actually produced it rather than against today's config.

  2. *Did anyone quietly edit this afterwards?*  Each record stores the hash
     of the one before it, so the file is a chain. Change a number in line
     400 and every hash from 400 onward stops verifying. `verify_chain()`
     reports the first line that breaks and the reason.

  3. *Did you do the same thing twice?*  Every executed action derives a
     deterministic idempotency key from the event, the action and the run
     window. Re-running a sweep replays decisions but refuses duplicate
     executions, so an operator who runs the agent twice by accident does
     not double-charge a customer or double-send a reminder.

Two honest limitations, stated here rather than buried:

  * A hash chain gives tamper **evidence**, not tamper **proofing**. Someone
    with write access to the file can recompute the entire chain from the
    genesis record. The defensible claim is narrower and still useful: you
    cannot alter, insert or delete a single record and leave the file
    internally consistent. Real immutability needs an anchor outside the
    writer's control — WORM object storage, or shipping the head hash
    somewhere the agent cannot reach. `chain_head()` exists so that anchor
    is a two-line integration rather than a rewrite.

  * The chain says nothing about whether the *inputs* were honest. It proves
    the record has not moved since it was written, not that the model was
    right.

Version 2 opened this file in write mode on every run, so each sweep erased
the last one's history. That is the specific bug this module exists to
retire, and it is why every writer here is append-only and why there is a
test asserting a second run does not shorten the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping, Optional

from . import config as C
from .schemas import Decision, RETRY_ACTIONS, OUTREACH_ACTIONS

GENESIS_HASH = "0" * 64

# Record types written to the decision log.
RECORD_DECISION = "decision"
RECORD_EXECUTION = "execution"
RECORD_RUN = "run"
RECORD_APPROVAL = "approval"

# Keys that must never appear anywhere in an audit payload. The audit file is
# the most-copied, longest-lived artefact the system produces — it gets
# attached to tickets, pasted into reviews and shipped to analysts. Cardholder
# data and contact details do not belong in it, and "we were careful" is not
# an enforcement mechanism, so writes are screened structurally instead.
#
# Note this bans the *keys*, not values that happen to look like them. A
# reviewer can read this list and know what the file cannot contain.
FORBIDDEN_KEYS = frozenset({
    "card_number", "card_no", "cardnumber", "pan", "primary_account_number",
    "cvv", "cvc", "card_cvv", "expiry", "expiry_month", "expiry_year",
    "cardholder_name", "account_number", "bank_account_number", "iban",
    "email", "email_address", "phone", "phone_number", "mobile", "msisdn",
    "address", "postal_address", "upi_id", "vpa", "token", "api_key",
    "authorization", "password", "secret",
})


class AuditIntegrityError(Exception):
    """Raised when a write would violate the audit file's invariants."""


# ---------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------

def canonical_json(payload: Mapping[str, Any]) -> str:
    """Byte-stable JSON, so the same payload always hashes the same way.

    Sorted keys and no incidental whitespace. Without this, a dict that
    happened to be built in a different order would produce a different
    hash for identical content, and chain verification would fail on a
    file nobody had touched.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def record_hash(prev_hash: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(payload)).encode("utf-8")).hexdigest()


def _screen_payload(payload: Any, path: str = "") -> None:
    """Refuse to write a payload containing a forbidden key, at any depth."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS:
                raise AuditIntegrityError(
                    f"refusing to write audit record: field {path}{key!r} is on the "
                    f"forbidden-key list. The audit trail must not carry cardholder "
                    f"data or contact details — record an identifier and resolve it "
                    f"in the adapter instead."
                )
            _screen_payload(value, f"{path}{key}.")
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _screen_payload(item, f"{path[:-1]}[]." if path.endswith(".") else path)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """A run identifier that survives restarts and sorts chronologically.

    Timestamp first so `sort` is `sort by time`, then four random hex chars
    so two sweeps started in the same second do not collide.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid.uuid4().hex[:4]}"


def decision_id(run_id: str, event_id: str) -> str:
    """Deterministic id for one (run, event) pair.

    Deterministic rather than random so that a replay of the same run
    produces the same identifiers, which is what makes execution
    idempotency checkable after the fact.
    """
    return hashlib.sha256(f"{run_id}:{event_id}".encode()).hexdigest()[:16]


def idempotency_key(event_id: str, action: str, window: str) -> str:
    """Key identifying "this action, on this event, in this window".

    `window` is deliberately a caller-supplied string rather than a
    timestamp. For retries the agent passes a date bucket, so the same
    payment cannot be retried twice for the same intent on the same day
    even across separate runs; for one-shot actions it passes the run id.
    The choice of window is a policy decision, so it is made where the
    policy lives rather than hidden in a hash function.
    """
    return hashlib.sha256(f"{event_id}|{action}|{window}".encode()).hexdigest()[:24]


# ---------------------------------------------------------------------
# The append-only store
# ---------------------------------------------------------------------

class AuditStore:
    """Hash-chained JSONL writer and reader.

    Opened in append mode, always. There is no method on this class that
    truncates, rewrites or deletes a record, and that is the point.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or C.AUDIT_LOG_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._head: Optional[str] = None

    # -- reading -----------------------------------------------------

    def __len__(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, "r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def __bool__(self) -> bool:
        """A store is a destination, not a collection — always truthy.

        Without this, `__len__` above makes an empty store falsy, and every
        `store or AuditStore()` fallback in this codebase silently discards an
        injected store that has not been written to yet. That is not a
        hypothetical: it happened, and the symptom was a test suite writing 576
        records into the shipped audit trail while its own temporary trail
        stayed empty. The call sites were fixed to test `is not None`, which is
        the real repair; this method is here so that a site anyone adds later
        cannot reintroduce the same failure.
        """
        return True

    def read(self) -> Iterator[dict[str, Any]]:
        """Yield every record in write order, skipping blank lines."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditIntegrityError(
                        f"{self.path}:{lineno} is not valid JSON ({exc}). The audit "
                        f"file has been truncated or hand-edited."
                    ) from None

    def chain_head(self) -> str:
        """Hash of the most recent record, or the genesis hash if empty.

        Publishing this value somewhere the agent cannot write is what would
        upgrade tamper evidence to something closer to tamper proofing.
        """
        if self._head is not None:
            return self._head
        head = GENESIS_HASH
        for record in self.read():
            head = record.get("record_hash", head)
        self._head = head
        return head

    # -- writing -----------------------------------------------------

    def append(self, record_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Append one record and return it, with chain fields attached."""
        _screen_payload(payload)
        prev = self.chain_head()
        body = {
            "record_type": record_type,
            "recorded_at": utcnow(),
            "policy_version": C.policy_version(),
            "code_version": C.CODE_VERSION,
            **payload,
        }
        digest = record_hash(prev, body)
        record = {**body, "prev_hash": prev, "record_hash": digest}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(record) + "\n")
        self._head = digest
        return record

    def append_decision(self, decision: Decision, run_id: str, *,
                        dry_run: bool,
                        explanation: str = "",
                        extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Record one decision in full, including what was refused.

        The `considered` list is the part that makes this evidence rather
        than a receipt. A record showing only the chosen action tells a
        reviewer nothing about whether the alternatives were even looked at;
        a record showing that a 10% discount was ranked, priced at a
        specific ENR, and then blocked by the sweep discount budget is
        something a reviewer can actually check.
        """
        did = decision_id(run_id, decision.event_id)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "decision_id": did,
            "dry_run": bool(dry_run),
            "event_id": decision.event_id,
            "event_type": decision.event_type,
            "amount_inr": round(float(decision.amount_inr), 2),
            "customer_id": decision.customer_id,
            "root_cause": decision.root_cause,
            "root_cause_confidence": round(float(decision.root_cause_confidence), 4),
            "root_cause_distribution": {
                k: round(float(v), 4) for k, v in decision.root_cause_distribution.items()
            },
            "action": decision.action,
            "chosen": decision.chosen.to_dict(),
            "arithmetic": explanation,
            "considered": [s.to_dict() for s in decision.considered],
            "rejected_reasons": dict(decision.rejected_reasons),
            "guardrails_applied": list(decision.guardrails_applied),
            "requires_human_approval": bool(decision.requires_human_approval),
            "approval_reason": decision.approval_reason,
            "expected_net_recovery_inr": round(float(decision.expected_net_recovery_inr), 2),
        }
        if extra:
            payload.update(dict(extra))
        return self.append(RECORD_DECISION, payload)

    def append_execution(self, *, run_id: str, decision_id_: str, event_id: str,
                         action: str, adapter: str, status: str,
                         idempotency_key_: str, dry_run: bool,
                         provider_ref: str = "", detail: str = "",
                         extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Record an attempted side effect.

        `status` is one of: simulated, refused, skipped_duplicate,
        awaiting_approval, error. Refusals are recorded with the same weight
        as successes — an adapter that declines to send a message because
        consent was missing is a safety event worth being able to count.
        """
        payload: dict[str, Any] = {
            "run_id": run_id,
            "decision_id": decision_id_,
            "event_id": event_id,
            "action": action,
            "adapter": adapter,
            "status": status,
            "idempotency_key": idempotency_key_,
            "dry_run": bool(dry_run),
            "provider_ref": provider_ref,
            "detail": detail,
        }
        if extra:
            payload.update(dict(extra))
        return self.append(RECORD_EXECUTION, payload)

    # -- verification -------------------------------------------------

    def verify_chain(self) -> dict[str, Any]:
        """Recompute every hash and report the first inconsistency.

        Returns a dict rather than raising, because the caller is usually a
        dashboard panel or a CLI check that wants to display the result. A
        clean file gives `{"ok": True, "records": n, "head": "..."}`.
        """
        prev = GENESIS_HASH
        count = 0
        for lineno, record in enumerate(self.read(), start=1):
            count += 1
            claimed_prev = record.get("prev_hash")
            claimed_hash = record.get("record_hash")
            if claimed_prev != prev:
                return {
                    "ok": False, "records": count, "broken_at_line": lineno,
                    "reason": (
                        f"record {lineno} claims to follow {str(claimed_prev)[:12]}… "
                        f"but the previous record hashes to {prev[:12]}… — a record "
                        f"was inserted or deleted here"
                    ),
                }
            body = {k: v for k, v in record.items()
                    if k not in ("prev_hash", "record_hash")}
            recomputed = record_hash(prev, body)
            if recomputed != claimed_hash:
                return {
                    "ok": False, "records": count, "broken_at_line": lineno,
                    "reason": (
                        f"record {lineno} ({record.get('record_type')}, "
                        f"event {record.get('event_id', '-')}) hashes to "
                        f"{recomputed[:12]}… but stores {str(claimed_hash)[:12]}… — "
                        f"its contents were modified after it was written"
                    ),
                }
            prev = claimed_hash
        return {"ok": True, "records": count, "head": prev,
                "reason": "every record hashes to its stored digest and links to its predecessor"}


# ---------------------------------------------------------------------
# The run index
# ---------------------------------------------------------------------

class RunIndex:
    """A separate append-only file listing every sweep the agent has run.

    Kept apart from the decision log so that "what runs have there been"
    is an O(runs) read rather than a scan of every decision ever made. It
    is what the dashboard's run selector is built on, and it is why a run
    id survives a process restart.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or C.RUN_INDEX_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        _screen_payload(payload)
        record = {"recorded_at": utcnow(), **payload}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(record) + "\n")
        return record

    def start(self, run_id: str, *, dry_run: bool, surfaces: Iterable[str],
              split: Optional[str], event_count: int,
              at_risk_inr: float) -> dict[str, Any]:
        return self.append({
            "record_type": RECORD_RUN,
            "phase": "started",
            "run_id": run_id,
            "dry_run": bool(dry_run),
            "surfaces": list(surfaces),
            "split": split or "all",
            "event_count": int(event_count),
            "at_risk_inr": round(float(at_risk_inr), 2),
            "policy_version": C.policy_version(),
            "code_version": C.CODE_VERSION,
        })

    def finish(self, run_id: str, *, summary: Mapping[str, Any],
               chain_head: str) -> dict[str, Any]:
        return self.append({
            "record_type": RECORD_RUN,
            "phase": "finished",
            "run_id": run_id,
            "chain_head": chain_head,
            **dict(summary),
        })

    def latest_run_id(self) -> Optional[str]:
        finished = [r for r in self.read() if r.get("phase") == "finished"]
        if finished:
            return str(finished[-1]["run_id"])
        started = [r for r in self.read() if r.get("phase") == "started"]
        return str(started[-1]["run_id"]) if started else None


# ---------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------

class ApprovalQueue:
    """Human sign-off, recorded as its own append-only chain.

    Approvals live in a separate file from decisions for a mundane but
    important reason: the person approving is not the process deciding.
    Interleaving them in one file makes it easy to write code that treats
    "the agent decided" and "a human agreed" as the same kind of fact. They
    are not, and improvement #6 asks specifically for a workflow where the
    second is required before anything happens.

    The queue is derived, not stored: pending items are decisions flagged
    `requires_human_approval` that have no matching resolution record. That
    means the queue cannot drift out of sync with the decision log, and it
    cannot be emptied by editing a status field.
    """

    def __init__(self, path: Optional[str] = None,
                 store: Optional[AuditStore] = None):
        self.path = path or C.APPROVALS_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.store = store if store is not None else AuditStore()

    def _read(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def resolutions(self) -> dict[str, dict[str, Any]]:
        """Latest resolution per decision id."""
        latest: dict[str, dict[str, Any]] = {}
        for record in self._read():
            latest[str(record["decision_id"])] = record
        return latest

    def resolve(self, decision_id_: str, *, approver: str, granted: bool,
                reason: str = "") -> dict[str, Any]:
        """Record a human's decision on a gated action.

        `approver` is a free-text identifier supplied by the caller. This
        build has no authentication, and pretending otherwise would be
        worse than saying so: the field records *who claimed* to approve,
        which is only meaningful behind a real identity layer. The
        dashboard endpoint that writes here is documented as requiring one
        before any non-demo use.
        """
        if not approver or not str(approver).strip():
            raise ValueError("approver is required — an unattributed approval is not one")
        record = {
            "record_type": RECORD_APPROVAL,
            "recorded_at": utcnow(),
            "decision_id": str(decision_id_),
            "approver": str(approver).strip(),
            "granted": bool(granted),
            "reason": str(reason),
            "policy_version": C.policy_version(),
        }
        _screen_payload(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(record) + "\n")
        # Mirror into the main chain so the decision log alone is a complete
        # account of what happened to every gated action.
        self.store.append(RECORD_APPROVAL, {
            "decision_id": str(decision_id_),
            "approver": record["approver"],
            "granted": record["granted"],
            "reason": record["reason"],
        })
        return record

    def pending(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Gated decisions with no resolution yet, newest first."""
        resolved = self.resolutions()
        out = []
        for record in self.store.read():
            if record.get("record_type") != RECORD_DECISION:
                continue
            if not record.get("requires_human_approval"):
                continue
            if run_id and record.get("run_id") != run_id:
                continue
            if str(record.get("decision_id")) in resolved:
                continue
            out.append(record)
        out.reverse()
        return out

    def is_granted(self, decision_id_: str) -> bool:
        record = self.resolutions().get(str(decision_id_))
        return bool(record and record.get("granted"))


# ---------------------------------------------------------------------
# Cross-run execution ledger
# ---------------------------------------------------------------------

@dataclass
class ExecutionLedger:
    """What has already been done, read back from the audit trail.

    This is the piece that makes frequency caps and idempotency real rather
    than notional. In-memory sweep counters bound one run; they say nothing
    about the run twenty minutes ago. A customer capped at two contacts per
    seven days is only actually capped if the second run can see what the
    first one sent, and the audit trail is the only durable record of that —
    so the ledger is built from it rather than from a side table that could
    disagree with it.

    Refused and duplicate-skipped executions are excluded: an action that
    never happened must not consume a customer's contact allowance.
    """
    executed_keys: set[str] = field(default_factory=set)
    attempts_by_event: dict[str, int] = field(default_factory=dict)
    contacts_by_customer: dict[str, list[str]] = field(default_factory=dict)

    COUNTED_STATUSES = frozenset({"simulated", "executed"})

    @classmethod
    def load(cls, store: Optional[AuditStore] = None) -> "ExecutionLedger":
        store = store if store is not None else AuditStore()
        ledger = cls()
        # Decision records carry the customer id; execution records do not,
        # so build the event -> customer map as we go.
        customer_of: dict[str, str] = {}
        for record in store.read():
            kind = record.get("record_type")
            if kind == RECORD_DECISION:
                customer_of[str(record.get("event_id"))] = str(record.get("customer_id"))
                continue
            if kind != RECORD_EXECUTION:
                continue
            if record.get("status") not in cls.COUNTED_STATUSES:
                continue
            key = str(record.get("idempotency_key", ""))
            if key:
                ledger.executed_keys.add(key)
            event_id = str(record.get("event_id"))
            action = str(record.get("action"))
            if action in RETRY_ACTIONS:
                ledger.attempts_by_event[event_id] = ledger.attempts_by_event.get(event_id, 0) + 1
            if action in OUTREACH_ACTIONS:
                cid = customer_of.get(event_id, "")
                if cid:
                    ledger.contacts_by_customer.setdefault(cid, []).append(
                        str(record.get("recorded_at", ""))
                    )
        return ledger

    def has_executed(self, key: str) -> bool:
        return key in self.executed_keys

    def prior_attempts(self, event_id: str) -> int:
        return self.attempts_by_event.get(event_id, 0)

    def contacts_within(self, customer_id: str, hours: float,
                        now: Optional[datetime] = None) -> int:
        """Contacts to this customer inside the trailing window."""
        stamps = self.contacts_by_customer.get(customer_id, [])
        if not stamps:
            return 0
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=float(hours))
        count = 0
        for raw in stamps:
            try:
                when = datetime.fromisoformat(raw)
            except ValueError:
                # An unparseable timestamp is counted, not ignored. Failing
                # closed on a frequency cap costs one message; failing open
                # costs a customer's goodwill.
                count += 1
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when >= cutoff:
                count += 1
        return count

    def hours_since_last_contact(self, customer_id: str,
                                 now: Optional[datetime] = None) -> Optional[float]:
        stamps = self.contacts_by_customer.get(customer_id, [])
        if not stamps:
            return None
        now = now or datetime.now(timezone.utc)
        newest: Optional[datetime] = None
        for raw in stamps:
            try:
                when = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if newest is None or when > newest:
                newest = when
        if newest is None:
            return None
        return max(0.0, (now - newest).total_seconds() / 3600.0)


# ---------------------------------------------------------------------
# Read helpers used by the service layer
# ---------------------------------------------------------------------

def decisions_for_run(run_id: str, store: Optional[AuditStore] = None
                      ) -> list[dict[str, Any]]:
    store = store if store is not None else AuditStore()
    return [r for r in store.read()
            if r.get("record_type") == RECORD_DECISION and r.get("run_id") == run_id]


def executions_for_run(run_id: str, store: Optional[AuditStore] = None
                       ) -> list[dict[str, Any]]:
    store = store if store is not None else AuditStore()
    return [r for r in store.read()
            if r.get("record_type") == RECORD_EXECUTION and r.get("run_id") == run_id]


def find_decision(decision_id_: str, store: Optional[AuditStore] = None
                  ) -> Optional[dict[str, Any]]:
    store = store if store is not None else AuditStore()
    for record in store.read():
        if (record.get("record_type") == RECORD_DECISION
                and str(record.get("decision_id")) == str(decision_id_)):
            return record
    return None


def latest_decision_for_event(event_id: str, store: Optional[AuditStore] = None
                              ) -> Optional[dict[str, Any]]:
    """The most recent recorded decision about one event, if there is one.

    Keyed on the event rather than the decision, because "has this ever been
    worked, and what did we do?" is a different question from "show me this
    record". An event can appear in several runs — a second sweep on the same
    day still records a decision, usually `no_action` with a
    `skipped_duplicate` reason — so the last one is the current answer.

    A linear scan, because the trail is an append-only file and not an index.
    Deliberately so: an index is a second copy of the truth that can disagree
    with it. If the scan ever costs too much, the fix is an index built *from*
    the log and rebuilt on demand, never one written alongside it.
    """
    latest: Optional[dict[str, Any]] = None
    for record in store.read() if store else AuditStore().read():
        if (record.get("record_type") != RECORD_DECISION
                or str(record.get("event_id")) != str(event_id)):
            continue
        if latest is None or str(record.get("recorded_at") or "") >= str(
                latest.get("recorded_at") or ""):
            latest = record
    return latest
