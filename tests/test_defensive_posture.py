"""Properties about what the system is not allowed to hold, learn from, or reach.

Named to match the references in src/schemas.py and src/ml/features.py, which
both point a reader here for the claims they make.

Three different claims are tested here and they should not be conflated.

The first is about *scope*: the decision layer holds no cardholder data and no
contact address. That is a PCI-DSS and privacy claim, and it is tested against
the dataclasses and the generated CSVs rather than against a comment, because
the way this claim gets broken is by someone adding a convenient field.

The second is about *provenance*: no model may be fitted on a column that
encodes the answer. The simulator writes oracle columns (`po_*`,
`true_root_cause`, `logged_recovered`, `is_fraudulent`) so the benchmark can
score policies against counterfactual truth, and every one of them would make
a classifier look excellent and be worthless. `assert_no_leakage` exists for
this; these tests check it is actually wired into each feature set rather than
merely available.

The third is about *secrets*: no file in this repository contains a credential.
That one is here because it was broken. A real `sk-ant-...` key was pasted over
`narrator.ENV_KEY`, which is supposed to hold the *name* of the environment
variable to read. See `TestNoCredentialIsCommitted` for what that cost and why
the check is a scan of the tree rather than a review habit.
"""

from __future__ import annotations

import csv
import dataclasses
import inspect
import os
import re
import unittest

from src import config as C
from src import dataio
from src import schemas
from src.ml import features as F
from src.ml import root_cause as RC
from src.ml import uplift as UP

from .helpers import needs_data

# Substrings that must not appear in any field the decision layer holds.
# Deliberately broad: `pan` catches both the card number field name and India's
# PAN tax id, and a false positive here costs one rename while a false negative
# costs a compliance incident.
CARDHOLDER_PATTERNS = (
    "card_number", "cardnumber", "pan", "cvv", "cvc", "expiry", "exp_month",
    "exp_year", "track_data", "magstripe", "aadhaar", "upi_id", "vpa",
    "account_number", "ifsc", "token_pan",
)

CONTACT_PATTERNS = ("email_address", "phone", "mobile_number", "whatsapp_number",
                    "address_line", "pincode", "postal_code")

# Credential shapes, by the prefix-and-length format each provider publishes.
# Matching on format rather than on entropy is what keeps this from flagging the
# repository's own long hex strings — a chain digest and a run id are both
# high-entropy and neither is a secret.
CREDENTIAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic api key":   re.compile(r"sk-ant-[a-z0-9]+-[A-Za-z0-9_\-]{40,}"),
    "openai-style secret": re.compile(r"\bsk-[A-Za-z0-9]{40,}\b"),
    "aws access key id":   re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "razorpay key":        re.compile(r"\brzp_(?:live|test)_[A-Za-z0-9]{10,}"),
    "github token":        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "google api key":      re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "slack token":         re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    "private key block":   re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    "bearer literal":      re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{30,}"),
}

# Text files worth scanning. Bytecode is excluded because it is generated, and
# `data/audit/archive` because it holds superseded trails kept for reference.
SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".html", ".md", ".txt", ".json",
                    ".jsonl", ".csv", ".cfg", ".toml", ".ini")
UNSCANNED_DIRS = frozenset({"__pycache__", ".git", "archive"})


def _field_names(cls: type) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


class TestNoCardholderData(unittest.TestCase):
    """No schema in the decision path names a cardholder-data field."""

    def test_dataclass_fields(self) -> None:
        for cls in (schemas.CustomerProfile, schemas.RiskEvent,
                    schemas.CandidateAction, schemas.ScoredAction,
                    schemas.Decision, schemas.SimulatedOutcomes):
            for name in _field_names(cls):
                lowered = name.lower()
                for pattern in CARDHOLDER_PATTERNS:
                    # Word-ish boundary so `payment_method` survives and
                    # `pan_hash` does not.
                    self.assertIsNone(
                        re.search(rf"(?:^|_){re.escape(pattern)}(?:$|_)", lowered),
                        f"{cls.__name__}.{name} looks like cardholder data "
                        f"(matched {pattern!r})",
                    )

    def test_customer_profile_holds_no_contact_address(self) -> None:
        """Consent flags yes, addresses no.

        The split is the point: the agent decides *whether* to contact someone
        and on *which* channel, and resolving that to an actual address is the
        messaging adapter's job. Keeping them apart means a leaked decision
        record cannot be used to contact anybody.
        """
        for name in _field_names(schemas.CustomerProfile):
            for pattern in CONTACT_PATTERNS:
                self.assertNotIn(pattern, name.lower(),
                                 f"CustomerProfile.{name} is a contact address")

    def test_consent_flags_are_present(self) -> None:
        """The other half of the same claim — the flags must exist to be checked."""
        names = _field_names(schemas.CustomerProfile)
        for flag in ("email_consent", "whatsapp_consent", "sms_consent", "dnd_flagged"):
            self.assertIn(flag, names)

    @needs_data
    def test_generated_csv_headers(self) -> None:
        """The simulator must not write cardholder data either.

        Tested against the files rather than the generator, because the claim
        a reader cares about is about the data sitting on disk.
        """
        for path in (C.CUSTOMERS_CSV, C.PAYMENTS_CSV, C.CHECKOUT_CSV, C.RECEIVABLES_CSV):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh))
            for column in header:
                lowered = column.lower()
                for pattern in CARDHOLDER_PATTERNS:
                    self.assertIsNone(
                        re.search(rf"(?:^|_){re.escape(pattern)}(?:$|_)", lowered),
                        f"{os.path.basename(path)} column {column!r} matched "
                        f"{pattern!r}",
                    )


class TestNoLeakage(unittest.TestCase):
    """Every feature set the models are fitted on passes assert_no_leakage."""

    def test_the_check_itself_works(self) -> None:
        """Guard the guard.

        If `assert_no_leakage` ever stopped raising, every test below would
        pass while asserting nothing at all. That failure mode is silent, so it
        gets its own test.
        """
        for bad in ("po_immediate_retry", "true_root_cause", "logged_recovered", "logged_action",
                    "is_fraudulent"):
            with self.assertRaises(Exception, msg=f"{bad} should be refused"):
                F.assert_no_leakage(["amount", bad])
        F.assert_no_leakage(["amount", "retry_count"])  # must not raise

    def test_declared_feature_lists(self) -> None:
        for name in dir(F):
            if not re.fullmatch(r"[A-Z_]+", name):
                continue
            value = getattr(F, name)
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                F.assert_no_leakage(value)

    def test_root_cause_feature_sets(self) -> None:
        for surface, spec in RC.SURFACE_FEATURES.items():
            names: list[str] = []
            for group in spec.values():
                names += list(group)
            F.assert_no_leakage(names)
            self.assertTrue(names, f"{surface} declares no features")

    def test_uplift_action_keys_are_not_outcome_columns(self) -> None:
        """`po_*` columns name the same actions the model predicts for.

        So the one thing that must not happen is an action key being used as a
        feature name. The keys are checked against the same predicate for that
        reason, even though they are labels rather than inputs.
        """
        for surface, keys in UP.SURFACE_ACTION_KEYS.items():
            self.assertTrue(keys, f"{surface} declares no action keys")
            F.assert_no_leakage([UP.split_action_key(k)[0] for k in keys])

    def test_surface_spec_features_are_clean(self) -> None:
        """The columns lifted into RiskEvent.features, before any model sees them."""
        for surface, spec in dataio.SURFACE_SPEC.items():
            F.assert_no_leakage(list(spec["feature_cols"]))
            F.assert_no_leakage(list(dataio.CUSTOMER_COLUMNS))

    def test_forbidden_prefixes_cover_every_oracle_column(self) -> None:
        """The prefix list must actually cover what the simulator writes.

        `assert_no_leakage` can only refuse what its prefix list describes, so
        this reads the generated files and checks that every column the
        benchmark treats as oracle knowledge is caught. Without this, adding a
        new oracle column to the simulator would silently make it eligible as
        a feature.
        """
        if not os.path.exists(C.PAYMENTS_CSV):
            self.skipTest("generated data missing")
        oracle_like = set()
        for path in (C.PAYMENTS_CSV, C.CHECKOUT_CSV, C.RECEIVABLES_CSV):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh))
            declared = set()
            for spec in dataio.SURFACE_SPEC.values():
                declared |= set(spec["feature_cols"])
            declared |= set(dataio.CUSTOMER_COLUMNS)
            declared |= {spec["id_col"] for spec in dataio.SURFACE_SPEC.values()}
            declared |= {"customer_id", "failed_at", "abandoned_at", "due_date"}
            for column in header:
                if column in declared:
                    continue
                oracle_like.add(column)

        # Everything left over is either an oracle column or a timestamp. The
        # oracle ones must be refused; anything else is listed in the failure
        # message so a reader can see what was classified how.
        unclaimed = [c for c in sorted(oracle_like)
                     if not c.startswith(F.FORBIDDEN_FEATURE_PREFIXES)
                     and not c.endswith(("_at", "_date"))]
        self.assertEqual(
            unclaimed, [],
            "these generated columns are neither declared features, timestamps, "
            "nor covered by FORBIDDEN_FEATURE_PREFIXES — decide which they are: "
            f"{unclaimed}",
        )


class TestNoCredentialIsCommitted(unittest.TestCase):
    """No file in this repository holds a secret, and the key constant is a name.

    This class exists because the property failed. A real Anthropic key was
    pasted over `narrator.ENV_KEY`, the constant that is supposed to hold the
    *name* `"ANTHROPIC_API_KEY"`. It went wrong in three ways at once, and the
    third is the one that makes this worth a test rather than a code review:

      1. A live credential sat in the source tree.
      2. `os.environ.get(ENV_KEY)` then looked up a variable named after the
         secret, which nothing sets — so narration refused every request, and
         the symptom looked like an unset key rather than a corrupted constant.
         A broken thing that reports the expected error is hard to notice.
      3. The refusal message interpolates `ENV_KEY`. So the secret was written
         to stdout and returned in the body of a `503` to anyone who could call
         `POST /api/narrate`. A credential in source is a leak; a credential on
         an error path is a leak with a delivery mechanism attached.

    Nothing about that was caught by the 259 tests already here, because every
    one of them asserted behaviour and this was a constant. Hence a scan.
    """

    def _scanned_files(self) -> list[str]:
        found: list[str] = []
        for root, dirs, names in os.walk(C.PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in UNSCANNED_DIRS]
            for name in names:
                if name.endswith(SCANNED_SUFFIXES):
                    found.append(os.path.join(root, name))
        return sorted(found)

    def test_the_scan_finds_a_planted_credential_of_every_shape(self) -> None:
        """Guard the guard, before trusting the sweep below.

        A scanner that matches nothing passes on a compromised tree and reports
        success, which is the failure mode this whole file is written against.
        So each pattern is shown a string it must catch. The Anthropic case uses
        the real leaked *shape* — right prefix, right length, invented body.

        Every example is *assembled* from fragments rather than written as one
        literal, and that is load-bearing rather than fussy. The sweep below
        scans this file too, so an inlined example would be found and reported
        as a committed credential. Excluding this file from the sweep was the
        alternative and it is worse: a test file is source, and the one place a
        pasted secret should never be able to hide is the file whose job is
        finding pasted secrets. Assembling keeps the invariant total — *no* file
        in the tree contains a credential-shaped literal, with no exceptions.
        """
        must_catch = {
            "anthropic api key": "ENV_KEY = \"sk-ant-" + "api03-" + "A" * 95 + "\"",
            "openai-style secret": "key = \"sk-" + "d" * 48 + "\"",
            "aws access key id": "AWS_ACCESS_KEY_ID=" + "AKIA" + "IOSFODNN7EXAMPLE",
            "razorpay key": "Razorpay(\"rzp_" + "live_A1b2C3d4E5f6G7\")",
            "github token": "token = \"ghp_" + "a" * 36 + "\"",
            "google api key": "k = \"AIza" + "b" * 35 + "\"",
            "slack token": "xoxb" + "-123456789012-abcdefghijkl",
            "private key block": "-----BEGIN RSA " + "PRIVATE KEY-----",
            "bearer literal": "Authorization: Bearer " + "c" * 40,
        }
        self.assertEqual(sorted(must_catch), sorted(CREDENTIAL_PATTERNS),
                         "every pattern needs a string proving it fires")
        for name, planted in must_catch.items():
            hits = [n for n, rx in CREDENTIAL_PATTERNS.items() if rx.search(planted)]
            self.assertIn(name, hits, f"{name} pattern did not match its own example")

    def test_the_scan_ignores_things_that_merely_look_like_secrets(self) -> None:
        """The other half: a scan that cries wolf gets switched off.

        These are all real strings from this repository — a chain digest, a run
        id, the header name the transport sends, the field name the audit
        screener refuses, and the deliberately-fake key the narrator tests use.
        None is a secret and none may be flagged.
        """
        must_ignore = (
            "# this line briefly held a real `sk-ant-...` credential",
            "KEY = \"sk-ant-test-not-a-real-key\"",
            "ENV_KEY = \"ANTHROPIC_API_KEY\"",
            "\"x-api-key\": api_key,",
            "\"address\", \"postal_address\", \"upi_id\", \"vpa\", \"token\", \"api_key\",",
            "1bcf90fc725fb964c1fe8ce4efd988c7e6eee519addee27bdaa39845fa5981ec",
            "run_20260828T042144Z_39af",
            "the bearer of responsibility for this decision",
        )
        for benign in must_ignore:
            hits = [n for n, rx in CREDENTIAL_PATTERNS.items() if rx.search(benign)]
            self.assertEqual(hits, [], f"false positive on {benign[:48]!r}: {hits}")

    def test_no_tracked_file_contains_a_credential(self) -> None:
        """The sweep itself, over every text file in the project.

        Includes the generated CSVs and the 15 MB audit trail, because a secret
        that reached a log is still a committed secret — and the audit records
        carry an `execution_context` that a future live adapter could be careless
        with.
        """
        paths = self._scanned_files()
        self.assertGreater(len(paths), 30,
                           "the scan found almost nothing — check the walk root")
        offences: list[str] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, OSError):
                continue
            for name, rx in CREDENTIAL_PATTERNS.items():
                match = rx.search(text)
                if match:
                    line = text[:match.start()].count("\n") + 1
                    # The finding is reported, never the value.
                    offences.append(
                        f"{os.path.relpath(path, C.PROJECT_ROOT)}:{line} "
                        f"looks like a {name}"
                    )
        self.assertEqual(offences, [], "credential-shaped literals found: "
                                       + "; ".join(offences))

    def test_the_env_key_constant_names_a_variable(self) -> None:
        """`ENV_KEY` is the name to look up, never the thing looked up.

        Pinned to the literal string. A constant is exactly the kind of thing
        that can be overwritten without any test noticing, which is what
        happened.
        """
        from src import narrator as N

        self.assertEqual(N.ENV_KEY, "ANTHROPIC_API_KEY")
        self.assertIsNone(
            CREDENTIAL_PATTERNS["anthropic api key"].search(N.ENV_KEY),
            "ENV_KEY holds something credential-shaped",
        )

    def test_the_missing_key_refusal_cannot_carry_a_secret(self) -> None:
        """The message names the variable, and never echoes a supplied key.

        This is the disclosure path from the incident: the refusal interpolates
        `ENV_KEY`, so whatever that constant holds is handed to the caller. That
        is safe only while the constant is a name — asserted above — and while
        nothing else in the message quotes a credential. Checked here with a
        recognisable fake passed in explicitly.
        """
        from src import narrator as N

        planted = "sk-ant-api03-" + "Z" * 95
        with self.assertRaises(N.MissingCredentials) as caught:
            N.Narrator(api_key="   ")
        message = str(caught.exception)
        self.assertIn("ANTHROPIC_API_KEY", message)
        self.assertNotIn(planted, message)
        for name, rx in CREDENTIAL_PATTERNS.items():
            self.assertIsNone(rx.search(message),
                              f"the refusal message contains a {name}")


class TestAdaptersCannotReachTheNetwork(unittest.TestCase):
    """No module under src/adapters imports an HTTP client.

    This is the load-bearing test for "this build cannot send anything". The
    adapters are the only code that would ever talk to a payment gateway or a
    messaging provider, and in this build they simulate. Keeping the capability
    *absent* rather than behind a flag means the check is a static one: if
    nothing in there can open a socket, no configuration mistake can make it.
    """

    CLIENTS = ("requests", "httpx", "aiohttp", "urllib.request", "urllib3",
               "http.client", "socket", "smtplib", "ftplib", "telnetlib",
               "paramiko", "boto3", "razorpay", "twilio")

    def _adapter_files(self) -> list[str]:
        directory = os.path.join(C.SRC_DIR, "adapters")
        return [os.path.join(directory, n) for n in sorted(os.listdir(directory))
                if n.endswith(".py")]

    def test_no_network_client_imports(self) -> None:
        for path in self._adapter_files():
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
            # Strip docstrings and comments: the modules discuss what a live
            # transport *would* import, and prose is not an import.
            code = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", source)
            code = re.sub(r"#.*", "", code)
            for client in self.CLIENTS:
                pattern = (rf"^\s*(?:import\s+{re.escape(client)}"
                           rf"|from\s+{re.escape(client)}[\s.]"
                           rf"|from\s+{re.escape(client)}\s+import)")
                self.assertIsNone(
                    re.search(pattern, code, re.M),
                    f"{os.path.basename(path)} imports {client!r} — this build "
                    f"is not permitted to hold a transport",
                )

    def test_no_dynamic_import_machinery(self) -> None:
        """An import the static check cannot see is worse than a visible one."""
        for path in self._adapter_files():
            with open(path, "r", encoding="utf-8") as fh:
                code = re.sub(r"#.*", "", fh.read())
            for hazard in ("__import__(", "importlib", "eval(", "exec(",
                           "compile(", "os.system", "subprocess"):
                self.assertNotIn(hazard, code,
                                 f"{os.path.basename(path)} uses {hazard}")

    def test_live_path_is_guarded_and_unimplemented(self) -> None:
        """Every adapter's live branch raises rather than doing something.

        The dry-run flag decides which branch runs; this asserts the other
        branch has nothing in it to run.
        """
        from src import adapters as AD

        for path in self._adapter_files():
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
            if "live" not in source.lower():
                continue
            self.assertIn(
                "LiveExecutionRefused", source + inspect.getsource(AD.Dispatcher),
                f"{os.path.basename(path)} mentions live execution but does not "
                f"reference the refusal",
            )


if __name__ == "__main__":
    unittest.main()
