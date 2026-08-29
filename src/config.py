"""
Configuration loading and version stamping.

Everything tunable lives in config/policy.yaml rather than scattered as
module constants. Two reasons, both about auditability:

  1. A decision made six months ago can be replayed against the exact
     policy that produced it, because POLICY_VERSION is recorded on every
     audit line and the config is a versioned file in the repo.
  2. Nobody has to grep the codebase to answer "what is the discount
     cap" during a compliance review. It is one file.

Loading is cached, and the loaded config is treated as read-only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

import yaml

# ---------------------------------------------------------------------
# Paths. Resolved from this file's location so every entrypoint works
# regardless of the directory it was invoked from.
# ---------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
WEB_DIR = os.path.join(SRC_DIR, "web")

POLICY_PATH = os.path.join(CONFIG_DIR, "policy.yaml")

# Data files
CUSTOMERS_CSV = os.path.join(DATA_DIR, "customers.csv")
PAYMENTS_CSV = os.path.join(DATA_DIR, "failed_payments.csv")
CHECKOUT_CSV = os.path.join(DATA_DIR, "checkout_abandonment.csv")
RECEIVABLES_CSV = os.path.join(DATA_DIR, "overdue_receivables.csv")

# Model artifacts
ROOT_CAUSE_MODEL_PATH = os.path.join(ARTIFACT_DIR, "root_cause_model.json")
UPLIFT_MODEL_PATH = os.path.join(ARTIFACT_DIR, "uplift_model.json")

# Audit + run outputs
AUDIT_LOG_PATH = os.path.join(DATA_DIR, "audit", "decisions.jsonl")
RUN_INDEX_PATH = os.path.join(DATA_DIR, "audit", "runs.jsonl")
APPROVALS_PATH = os.path.join(DATA_DIR, "audit", "approvals.jsonl")
BENCHMARK_PATH = os.path.join(ARTIFACT_DIR, "benchmark.json")

# Code version, distinct from policy version. Policy can change without
# code changing, and vice versa; both are stamped on each decision.
CODE_VERSION = "3.1.0"


def _deep_freeze(obj: Any) -> Any:
    """Make a nested dict/list structure read-only.

    Guardrail config that any module could mutate at runtime is not a
    guardrail. Freezing means an accidental `cfg["limits"]["x"] = 999`
    raises instead of silently widening a money limit.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(v) for v in obj)
    return obj


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Mapping[str, Any]:
    """Load and freeze the policy config."""
    path = path or POLICY_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    _validate(raw, path)
    return _deep_freeze(raw)


def _validate(cfg: dict, path: str) -> None:
    """Fail loudly and early on a malformed policy file.

    A config typo that silently defaults a discount cap to zero (or to
    infinity) is exactly the class of bug that matters most here, so the
    required keys are checked explicitly rather than discovered by a
    KeyError three modules later.
    """
    required_top = ["policy_version", "execution", "limits", "retries",
                    "contact", "receivables", "economics", "llm"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required section(s): {missing}")

    limits = cfg["limits"]
    for key in ["max_auto_approve_amount_inr", "max_discount_pct",
                "min_confidence_to_act", "min_expected_net_recovery_inr"]:
        if key not in limits:
            raise ValueError(f"{path}: limits.{key} is required")

    if not 0 <= limits["max_discount_pct"] <= 100:
        raise ValueError(f"{path}: limits.max_discount_pct must be 0-100")
    if not 0 <= limits["min_confidence_to_act"] <= 1:
        raise ValueError(f"{path}: limits.min_confidence_to_act must be 0-1")
    if limits["max_auto_approve_amount_inr"] <= 0:
        raise ValueError(f"{path}: limits.max_auto_approve_amount_inr must be > 0")

    retries = cfg["retries"]
    if retries.get("max_attempts_per_payment", 0) < 1:
        raise ValueError(f"{path}: retries.max_attempts_per_payment must be >= 1")
    if retries.get("max_attempts_per_payment") > 5:
        # Not a style preference. More than a handful of automated attempts
        # against one instrument is the signature of card testing, and it
        # gets merchants flagged. Refuse to run in that configuration.
        raise ValueError(
            f"{path}: retries.max_attempts_per_payment > 5 is refused — "
            "high automated retry counts against a single instrument are "
            "indistinguishable from card testing and risk issuer blocks."
        )
    if retries.get("min_hours_between_attempts", 0) < 1:
        raise ValueError(
            f"{path}: retries.min_hours_between_attempts must be >= 1 — "
            "rapid-fire retries are an abuse pattern, not a recovery strategy."
        )


def policy_version(cfg: Mapping[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return str(cfg["policy_version"])


def ensure_dirs() -> None:
    """Create the output directories the pipeline writes into."""
    for d in [DATA_DIR, ARTIFACT_DIR, os.path.join(DATA_DIR, "audit")]:
        os.makedirs(d, exist_ok=True)


def kill_switch_engaged(cfg: Mapping[str, Any] | None = None) -> bool:
    """True when the ops halt file is present at the project root.

    Checked before every execution, so an operator can stop the agent
    mid-sweep by touching one file, without needing to kill a process or
    redeploy.
    """
    cfg = cfg or load_config()
    name = cfg["execution"].get("kill_switch_file", "HALT")
    return os.path.exists(os.path.join(PROJECT_ROOT, name))
