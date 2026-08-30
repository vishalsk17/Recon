#!/usr/bin/env python3
"""One command to take this project from a fresh checkout to a running dashboard.

    python run.py                 # do whatever still needs doing, then serve
    python run.py --no-serve      # same, but stop before starting the server
    python run.py --force         # redo every step, including a fresh sweep
    python run.py --check         # verify the trail and run the test suite

`src/__init__.py` advertises this file as doing the whole pipeline in order, so
that is what it does: generate the dataset, fit the models, work a sweep, score
the benchmark, serve the dashboard. Each step is skipped when its output is
already on disk, which makes the script safe to run twice and makes the common
case — a checkout that already has artefacts — start the dashboard immediately.

**Why skipping is the default rather than a flag.** Two of these steps are not
idempotent in the way a build step is. `data/generate_all.py` rewrites the CSVs,
and the shipped audit trail refers to event ids inside them; `src.agent run`
*appends* to that trail, because it is append-only by design. So a script that
eagerly re-ran everything would quietly invalidate the evidence the README
quotes figures from, and it would do it on the first command a new reader types.
Re-running is a decision, so it needs `--force`, and `--force` says out loud
what it is about to change before it does it.

Nothing here is required. Every step is one command, they are all listed in the
README, and running them by hand is the better way to understand the project.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Each step: label, the command, and the artefact whose presence means "done".
# The path is what makes skipping possible and honest — presence of the output,
# not a stamp file this script wrote about itself.
DATA_FILES = [
    os.path.join(HERE, "data", "customers.csv"),
    os.path.join(HERE, "data", "failed_payments.csv"),
    os.path.join(HERE, "data", "checkout_abandonment.csv"),
    os.path.join(HERE, "data", "overdue_receivables.csv"),
]
MODEL_FILES = [
    os.path.join(HERE, "artifacts", "root_cause_model.json"),
    os.path.join(HERE, "artifacts", "uplift_model.json"),
]
AUDIT_FILE = os.path.join(HERE, "data", "audit", "decisions.jsonl")
BENCHMARK_FILE = os.path.join(HERE, "artifacts", "benchmark.json")


def _present(paths: list[str]) -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths)


def _run(label: str, command: list[str]) -> None:
    """Run one step, streaming its output, and stop the script if it fails.

    No output capture: these steps print training tables and sweep summaries
    that are the most informative thing on the screen, and swallowing them to
    re-print a tidy summary would be worse.
    """
    print(f"\n{'=' * 74}\n{label}\n  $ {' '.join(command)}\n{'=' * 74}", flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=HERE)
    if result.returncode != 0:
        print(f"\n{label} failed (exit {result.returncode}). Stopping here rather "
              f"than running later steps against a half-built state.",
              file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"  ({label.lower()} took {time.time() - started:.1f}s)", flush=True)


def _skip(label: str, why: str) -> None:
    print(f"\n-- skipping {label}: {why}")


def check_dependencies() -> None:
    """Fail with the pip command rather than a traceback three imports deep."""
    missing = []
    for module, package in (("numpy", "numpy"), ("pandas", "pandas"), ("yaml", "PyYAML")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"error: missing dependencies: {', '.join(missing)}\n"
              f"  pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2)


def do_check() -> int:
    """Verify the audit chain, then run the test suite."""
    _run("Verifying the audit hash chain", [sys.executable, "-m", "src.agent", "verify"])
    _run("Running the test suite",
         [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Build and serve the Revenue Recovery Agent.",
        epilog="Steps whose output already exists are skipped; use --force to redo them.",
    )
    parser.add_argument("--force", action="store_true",
                        help="redo every step, rewriting the dataset and appending "
                             "a new sweep to the audit trail")
    parser.add_argument("--no-serve", action="store_true",
                        help="build everything but do not start the dashboard")
    parser.add_argument("--check", action="store_true",
                        help="verify the audit chain and run the tests, then exit")
    parser.add_argument("--port", type=int, default=None,
                        help="port for the dashboard (default: the server's own)")
    args = parser.parse_args(argv)

    check_dependencies()

    if args.check:
        return do_check()

    if args.force:
        # Said before doing, because one of these is not reversible in the
        # ordinary sense: the audit trail is append-only, so a forced sweep adds
        # records that cannot be removed, and the record counts quoted in the
        # README and artifacts/verified_metrics.md will no longer match.
        print("--force: the dataset will be rewritten and a new sweep appended to\n"
              "         the audit trail. The trail is append-only, so the record\n"
              "         count quoted in the documentation will no longer match.")

    if args.force or not _present(DATA_FILES):
        _run("Generating the synthetic dataset",
             [sys.executable, os.path.join("data", "generate_all.py")])
    else:
        _skip("dataset generation", "the four CSVs are already present")

    if args.force or not _present(MODEL_FILES):
        _run("Training the models", [sys.executable, "-m", "src.train"])
    else:
        _skip("training", "artifacts/root_cause_model.json and uplift_model.json exist")

    if args.force or not _present([AUDIT_FILE]):
        _run("Working a sweep over the held-out split",
             [sys.executable, "-m", "src.agent", "run", "--split", "test"])
    else:
        _skip("the sweep", "an audit trail already exists, and it is append-only — "
                           "use --force to add another run")

    if args.force or not _present([BENCHMARK_FILE]):
        _run("Scoring the agent against the baselines",
             [sys.executable, "-m", "src.benchmark"])
    else:
        _skip("the benchmark", "artifacts/benchmark.json exists")

    _run("Verifying the audit hash chain", [sys.executable, "-m", "src.agent", "verify"])

    if args.no_serve:
        print("\nEverything is built. Start the dashboard with:\n"
              "  python -m src.server --open")
        return 0

    command = [sys.executable, "-m", "src.server", "--open"]
    if args.port is not None:
        command += ["--port", str(args.port)]
    print(f"\n{'=' * 74}\nStarting the Recon\n{'=' * 74}", flush=True)
    # Not via _run: this one blocks until Ctrl-C, and its own handler prints the
    # goodbye. Exec-style hand-off keeps the exit code honest.
    return subprocess.run(command, cwd=HERE).returncode


if __name__ == "__main__":
    raise SystemExit(main())
