"""
Shared simulation machinery for the synthetic data generators.

Why this file exists
--------------------
The v2 generators emitted a single `was_recovered` column: whether the
event recovered under *one* implied intervention. That is enough to report
"the agent recovered X" but not enough to answer the question judges
actually ask, which is "compared to what?" You cannot compute the uplift
of a policy from a log that only contains one action per event.

So this simulator emits **potential outcomes**: for every event, whether it
would have recovered under *each* action variant available on its surface.
That gives two things the previous design could not:

  * Honest offline policy evaluation. Any policy — no intervention, retry
    everything, the agent — can be scored on the same events by looking up
    the outcome of whatever action it chose. See src/benchmark.py.
  * A learnable uplift signal. The models train on a *randomised
    exploration log* (one action sampled per event, plus only that
    action's outcome), never on the full counterfactual table. The oracle
    columns exist solely for scoring.

Correlated outcomes via a Gaussian copula
-----------------------------------------
Naively drawing each action's outcome independently would be wrong: it
would imply an event that resists a reminder is no less likely to resist a
discount, when in reality some customers are simply more recoverable than
others. Drawing them from a single shared latent would be equally wrong in
the other direction — it would make the best action identical for every
event, and the ranking problem trivial.

So each outcome is drawn from a Gaussian copula: a shared per-event latent
(weight `RHO`) plus per-action noise. This preserves each action's marginal
probability exactly while producing realistic correlation *and* genuine
heterogeneity in which action is best for whom. That heterogeneity is what
an expected-value ranker can exploit and a fixed cause-to-action lookup
cannot.
"""

from __future__ import annotations

import random
from statistics import NormalDist

# Correlation between action outcomes for the same event. 0 would mean
# recoverability is entirely action-specific; 1 would mean the best action
# is the same for everyone. 0.65 leaves real per-customer heterogeneity.
RHO = 0.65

_NORM = NormalDist()


def clamp(p: float, lo: float = 0.005, hi: float = 0.98) -> float:
    """Keep probabilities away from 0 and 1.

    Exact 0/1 would make the copula's inverse CDF infinite, and a real
    recovery channel is never literally impossible or certain.
    """
    return max(lo, min(hi, p))


class OutcomeSampler:
    """Draws correlated potential outcomes for one event."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def draw(self, probs: dict[str, float]) -> dict[str, bool]:
        """probs: action_key -> marginal recovery probability.

        Returns action_key -> recovered?, correlated across actions but
        with each marginal preserved.
        """
        z_shared = self.rng.gauss(0.0, 1.0)
        tail = (1.0 - RHO ** 2) ** 0.5

        out: dict[str, bool] = {}
        for key, p in probs.items():
            p = clamp(p)
            z_action = self.rng.gauss(0.0, 1.0)
            combined = RHO * z_shared + tail * z_action
            # combined ~ N(0,1); P(combined < probit(p)) == p
            out[key] = combined < _NORM.inv_cdf(p)
        return out


def sample_logged_action(rng: random.Random, action_keys: list[str]) -> str:
    """Pick the action recorded in the exploration log for this event.

    Uniform random over the learnable action set. This stands in for a
    real merchant running a randomised exploration period before handing
    the decision to a model — which is the only honest way to get
    unbiased per-action recovery estimates from production data.
    """
    return rng.choice(action_keys)


def parse_action_key(key: str) -> tuple[str, float]:
    """Split a variant key like 'offer_bounded_discount@10' into
    ('offer_bounded_discount', 10.0). Bare keys return a 0.0 parameter."""
    if "@" in key:
        base, param = key.split("@", 1)
        return base, float(param)
    return key, 0.0


def write_csv(path: str, rows: list[dict]) -> None:
    import csv
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):,} rows -> {os.path.relpath(path)}")
