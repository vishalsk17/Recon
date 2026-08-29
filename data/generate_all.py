"""
Runs all four generators in dependency order.

Customers must exist first — every event joins to a customer, and the
event generators read that table to correlate ticket sizes, engagement and
consent with the account.

    python data/generate_all.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import generate_customers      # noqa: E402
import generate_payments       # noqa: E402
import generate_checkout       # noqa: E402
import generate_receivables    # noqa: E402


def main() -> None:
    print("=" * 68)
    print("Generating synthetic revenue-at-risk dataset")
    print("=" * 68)
    generate_customers.main()
    generate_payments.main()
    generate_checkout.main()
    generate_receivables.main()
    print("=" * 68)
    print("Done. Next: python -m src.train")
    print("=" * 68)


if __name__ == "__main__":
    main()
