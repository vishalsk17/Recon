"""
Trains both model families and reports honestly on them.

    python -m src.train

What gets reported, and why each number is here
-----------------------------------------------
* **Accuracy against a majority baseline and a noise ceiling.** Bare accuracy
  is close to meaningless on an imbalanced label set. The label-noise ceiling
  is printed alongside it: because 7% of recorded causes are deliberately
  mis-tagged by the generator, even a model that perfectly recovers the true
  cause can only score about 0.944 against the *recorded* labels. Quoting
  85% without that context invites the obvious question of why it isn't 99%,
  and the honest answer is that 99% would mean the data was too clean.

* **Confidence calibration.** The human-review guardrail triggers below a
  confidence threshold, so the threshold is only meaningful if the
  confidence is. This measures whether "0.7 confident" is right about 70%
  of the time.

* **Per-action calibration for the recovery models.** These probabilities get
  multiplied by rupee amounts in src/economics.py. A model that ranks well
  but is systematically 15 points high produces precise, confident, wrong
  money figures — so calibration error matters more here than AUC does.

Everything is fitted on the `train` split only, and every number reported is
computed on the held-out `test` split. Splits come from a stable hash of the
event id (see src/dataio.py), so they are identical for every model and
across runs.
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import config as C
from . import dataio
from .ml.root_cause import RootCauseBundle, RootCauseClassifier
from .ml.uplift import RecoveryBundle, RecoveryModel
from .schemas import CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE, PAYMENT_FAILURE

SURFACES = [PAYMENT_FAILURE, CHECKOUT_ABANDONMENT, OVERDUE_RECEIVABLE]

# Must match LABEL_NOISE_RATE / the number of causes in data/generate_payments.py.
PAYMENT_LABEL_NOISE_RATE = 0.07
PAYMENT_N_CAUSES = 5


def noise_ceiling(noise_rate: float, n_classes: int) -> float:
    """Best achievable accuracy against deliberately noisy labels.

    With probability (1 - noise) the recorded label is the true cause and a
    perfect model gets it right. With probability `noise` the label was
    replaced by a uniform draw, which coincidentally matches the true cause
    1/n_classes of the time.
    """
    return (1 - noise_rate) + noise_rate / n_classes


def _fmt_pct(x: float | None) -> str:
    return "   n/a" if x is None else f"{x:6.1%}"


def train_root_cause() -> tuple[RootCauseBundle, dict[str, Any]]:
    print("\n" + "=" * 74)
    print("ROOT-CAUSE CLASSIFIERS  —  why is this revenue at risk?")
    print("=" * 74)

    bundle = RootCauseBundle()
    report: dict[str, Any] = {}

    for surface in SURFACES:
        train_df = dataio.load_surface_df(surface, split="train")
        test_df = dataio.load_surface_df(surface, split="test")

        clf = RootCauseClassifier(surface).fit(train_df)
        metrics = clf.evaluate(test_df)
        clf.metrics = metrics
        bundle.models[surface] = clf
        report[surface] = metrics

        print(f"\n--- {surface} ---")
        print(f"  train / test rows      {len(train_df):,} / {len(test_df):,}")
        print(f"  accuracy (held out)    {metrics['accuracy']:.3f}")
        print(f"  majority baseline      {metrics['majority_baseline']:.3f}"
              f"   <- accuracy must beat this to mean anything")
        if surface == PAYMENT_FAILURE:
            ceiling = noise_ceiling(PAYMENT_LABEL_NOISE_RATE, PAYMENT_N_CAUSES)
            gap = ceiling - metrics["accuracy"]
            print(f"  label-noise ceiling    {ceiling:.3f}"
                  f"   <- unreachable by construction; {gap:.3f} short of it")
        print(f"  log loss               {metrics['log_loss']:.3f}")
        print(f"  confidence calib. err  {metrics['confidence_calibration_error']:.3f}"
              f"   <- gates the human-review threshold")
        print("  per class:")
        print(f"    {'cause':<34}{'n':>6}{'prec':>8}{'rec':>8}{'f1':>8}")
        for row in metrics["per_class"]:
            print(f"    {row['class']:<34}{row['support']:>6}"
                  f"{row['precision']:>8.2f}{row['recall']:>8.2f}{row['f1']:>8.2f}")

    return bundle, report


def train_recovery() -> tuple[RecoveryBundle, dict[str, Any]]:
    print("\n" + "=" * 74)
    print("RECOVERY MODELS  —  P(recovered | event, action), one model per action")
    print("=" * 74)
    print("Fitted on the randomised exploration log only. The oracle counterfactual")
    print("columns are never read here; they are used solely to score policies in")
    print("src/benchmark.py. That separation is what makes the uplift numbers honest.")

    bundle = RecoveryBundle()
    report: dict[str, Any] = {}

    for surface in SURFACES:
        train_df = dataio.load_surface_df(surface, split="train")
        test_df = dataio.load_surface_df(surface, split="test")

        model = RecoveryModel(surface).fit(train_df)
        metrics = model.evaluate(test_df)
        model.metrics = metrics
        bundle.models[surface] = model
        report[surface] = metrics

        print(f"\n--- {surface} ---")
        print(f"  overall AUC            {metrics['overall_auc']}")
        print(f"  overall Brier          {metrics['overall_brier']}")
        print(f"  overall calib. error   {metrics['overall_calibration_error']}"
              f"   <- these probabilities are multiplied by rupees")
        print(f"    {'action':<46}{'train n':>9}{'pred':>8}{'obs':>8}{'AUC':>8}")
        for row in metrics["per_action"]:
            n_train = model.train_counts.get(row["action"], 0)
            print(f"    {row['action']:<46}{n_train:>9}"
                  f"{row['mean_predicted']:>8.3f}{row['observed_rate']:>8.3f}"
                  f"{row['auc']:>8.3f}")

        # The point of per-action models: does the recommended action
        # actually vary between events? If one action dominated everywhere,
        # expected-value ranking would be pointless and a lookup table
        # would do the job.
        preds = model.predict_all(test_df)
        keys = list(preds)
        import numpy as np
        stacked = np.column_stack([preds[k] for k in keys])
        best = [keys[i] for i in stacked.argmax(axis=1)]
        counts: dict[str, int] = {}
        for b in best:
            counts[b] = counts.get(b, 0) + 1
        print("  highest-probability action varies across events:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<46}{v:>6}  ({v / len(best):.0%})")

    return bundle, report


def main() -> None:
    C.ensure_dirs()
    cfg = C.load_config()

    print("=" * 74)
    print(f"TRAINING  —  code v{C.CODE_VERSION}, policy v{cfg['policy_version']}")
    print("=" * 74)

    rc_bundle, rc_report = train_root_cause()
    rec_bundle, rec_report = train_recovery()

    rc_bundle.save(C.ROOT_CAUSE_MODEL_PATH)
    rec_bundle.save(C.UPLIFT_MODEL_PATH)

    report_path = os.path.join(C.ARTIFACT_DIR, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({
            "code_version": C.CODE_VERSION,
            "policy_version": cfg["policy_version"],
            "root_cause": rc_report,
            "recovery": rec_report,
        }, fh, indent=1)

    print("\n" + "=" * 74)
    print("Saved:")
    print(f"  {os.path.relpath(C.ROOT_CAUSE_MODEL_PATH, C.PROJECT_ROOT)}")
    print(f"  {os.path.relpath(C.UPLIFT_MODEL_PATH, C.PROJECT_ROOT)}")
    print(f"  {os.path.relpath(report_path, C.PROJECT_ROOT)}")
    print("\nModels are plain JSON — open one and read the coefficients. No pickle,")
    print("so loading a model file is not code execution.")
    print("\nNext: python -m src.agent run")
    print("=" * 74)


if __name__ == "__main__":
    main()
