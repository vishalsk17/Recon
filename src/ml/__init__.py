"""Machine-learning layer.

Two model families, both plain regularised logistic regression serialised
to JSON (see logistic.py for why that is a deliberate choice rather than a
limitation):

* `root_cause` — why is this revenue at risk? One multinomial model per
  surface, returning a calibrated posterior over causes.
* `uplift` — P(recovered | event, action), one binary model per action
  variant per surface. This is what makes expected-value ranking possible.
"""
