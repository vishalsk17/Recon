"""Revenue Recovery Agent (v3).

A bounded, auditable agent that decides how to recover revenue at risk
across three surfaces: failed payments, abandoned checkouts and overdue
receivables.

Entry points (run from the project root):

    python data/generate_all.py     # build the synthetic dataset
    python -m src.train             # fit and report on the models
    python -m src.agent run         # run a sweep, write the audit trail
    python -m src.benchmark         # baselines vs the agent, with CIs
    python -m src.server            # Recovery Command Centre dashboard

Or `python run.py` to do all of the above in order.
"""

from .config import CODE_VERSION  # noqa: F401

__version__ = CODE_VERSION
