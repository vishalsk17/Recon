"""Tests for the revenue recovery agent.

Run them all from the project root:

    python -m unittest discover -s tests -t .

The `-t .` matters and is not decoration. Without it, discovery treats `tests/`
as the top-level directory, the package-relative `from .helpers import ...` has
no parent package to resolve against, and every module fails to import — which
looks like seven broken tests rather than a wrong command. The line above was
wrong in an earlier version of this file and produced exactly that.

Stdlib `unittest` only, for the same reason the rest of the project has almost
no dependencies: a test suite that cannot be run because pytest is missing is
not a test suite.

What these tests are for is worth stating, because it shapes what is in here and
what is not. They are not here to raise a coverage number. Every test asserts a
property that the README, the model card or the security note *claims* — so that
a change which quietly falsifies one of those claims fails here rather than
being discovered by a reader who trusted them. That is why several tests read
the shipped source or the generated audit trail rather than a fixture: the claim
is about the real artefact.

The files, and the claim each one is responsible for:

  test_fixtures.py           the scaffolding in helpers.py matches the real data
                             contract, and the two customer stand-ins are not
                             interchangeable
  test_defensive_posture.py  no cardholder data in any shape, no outcome label
                             reachable from the decision path, no live transport
  test_economics.py          the arithmetic: incremental against inaction, and
                             the retention term kept separate from the money in
                             play
  test_policy.py             every guardrail, each one proved to be the rule
                             doing the blocking
  test_audit.py              the hash chain detects tampering, forbidden fields
                             cannot be written, and running twice is safe
  test_narrator.py           the model sees only a fact sheet, gets no tools, and
                             its output is refused unless every figure traces
                             back to the decision
  test_server.py             loopback only, no route that starts a sweep, one
                             write path that can only release a decision that
                             already exists

Three things skip themselves cleanly rather than failing when the artefacts are
absent: tests needing generated data, trained weights, or the shipped audit
trail. A fresh checkout reports skips instead of a wall of errors that hide the
real failures. Everything else — the arithmetic, the static-source checks, the
whole of test_fixtures — runs in milliseconds against nothing on disk.

No test in this directory may append to `data/audit/`. That trail is the demo
evidence and its chain head is quoted in the documentation, so a test run that
lengthened it would make those two disagree. `helpers.AuditCase` checks the size
of all three shipped files after every test and names the culprit if one grew,
because this rule was broken once and stayed invisible for a full run.
"""
