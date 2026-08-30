# Model card

Four model families, all fitted in this repository, all serialisable to JSON,
none of them the interesting part of the system.

That last clause is the honest headline. The money in this project comes from
pricing every available action honestly and refusing the ones policy forbids —
not from sharp prediction. The models are deliberately simple, their
discrimination is modest, and the design assumes it. A section at the end
explains why that is a defensible choice rather than an excuse.

Every figure below comes from `artifacts/training_report.json`, produced by
`python -m src.train`, measured on a held-out `test` split the models never saw.
Code version 3.1.0, policy version 3.1.0. The same figures are recorded in
`artifacts/verified_metrics.md`, which is the file any documented number in this
project has to appear in first.

---

## What the models are

**Root-cause classifier** — one regularised multinomial logistic regression per
surface, answering "why is this revenue at risk". Four to five classes each.

**Recovery-probability models ("uplift")** — one binary logistic regression per
`(surface, action)` pair, answering "if I take this action on this event, what is
the chance it recovers". Fourteen models across three surfaces. These feed
directly into the expected-net-recovery arithmetic in `src/economics.py`.

Two actions are priced without a fitted model, and the count above excludes them
on purpose. `stop_and_flag_fraud` has `P(recover) = 0.0` — a blocked payment does
not settle, which is a definition rather than an estimate — and
`request_human_review` is priced at 0.85 of the best automated option's ENR,
which is a stated assumption about how well a person does with a case a machine
declined to take. Both carry `probability_is_assumed=True` through to the audit
record and onto the dashboard, so a reviewer reading either decision is told the
number is an assumption instead of having to know it. Fourteen fitted models plus
two stated assumptions is the honest description of what prices a candidate set.

Both families are implemented in `src/ml/logistic.py`: gradient descent, L2
regularisation, numpy only, about 235 lines. No scikit-learn.

Two reasons, and neither is aesthetic. First, the weights serialise to JSON, so a
reviewer can read a model as text instead of trusting a pickle — and pickle is a
code-execution format, which is not a thing to put in a pipeline that moves
money. Second, the arithmetic behind a decision that a compliance reviewer may
ask about six months from now should be inspectable without installing anything.

Feature encoding is shared between training and inference in
`src/ml/features.py`, which is the only defence against the train/serve skew
that silently ruins models like these.

---

## How well they work

### Root cause

Accuracy is reported against the majority-class baseline, because accuracy on an
imbalanced multi-class problem means nothing without it.

| surface | n | accuracy | majority baseline | log loss | calibration error |
|---|---:|---:|---:|---:|---:|
| payment_failure | 974 | 81.3% | 32.2% | 0.608 | 0.074 |
| checkout_abandonment | 568 | 83.5% | 31.0% | 0.471 | 0.062 |
| overdue_receivable | 302 | 86.8% | 38.7% | 0.364 | 0.020 |

Comfortably above baseline on all three. The calibration error matters more than
the accuracy here, because the number the system actually consumes is the
*confidence*, not the label: `min_confidence_to_act: 0.55` routes low-confidence
cases to a human, and a confidence that does not mean what it says would make
that gate either useless or arbitrary.

### Recovery probability

| surface | n | Brier | calibration error | AUC |
|---|---:|---:|---:|---:|
| payment_failure | 974 | 0.1466 | 0.0295 | 0.7421 |
| checkout_abandonment | 568 | 0.2059 | 0.0657 | 0.7148 |
| overdue_receivable | 302 | 0.2068 | 0.0693 | 0.7171 |

**An AUC near 0.72 is modest and is not to be dressed up.**

### The pooled calibration figures flatter the models

This is the most important line in this document, and it is not visible in the
table above.

Broken out per action, calibration is substantially worse than the pooled number
suggests, because errors in opposite directions cancel when pooled:

| surface / action | n | observed | predicted | calib. error | AUC |
|---|---:|---:|---:|---:|---:|
| payment_failure / do_nothing | 196 | 0.112 | 0.108 | 0.026 | 0.640 |
| payment_failure / immediate_retry | 202 | 0.198 | 0.181 | 0.025 | 0.769 |
| payment_failure / delayed_retry@12 | 193 | 0.321 | 0.250 | **0.114** | 0.714 |
| payment_failure / delayed_retry@48 | 183 | 0.240 | 0.229 | 0.083 | 0.737 |
| payment_failure / prompt_new_payment_method | 200 | 0.165 | 0.214 | 0.052 | 0.615 |
| checkout / do_nothing | 109 | 0.128 | 0.112 | 0.023 | 0.623 |
| checkout / send_reminder_email | 123 | 0.309 | 0.340 | 0.092 | 0.717 |
| checkout / send_reminder_whatsapp | 99 | 0.485 | 0.405 | **0.142** | 0.671 |
| checkout / offer_bounded_discount@5 | 119 | 0.462 | 0.442 | 0.119 | **0.577** |
| checkout / offer_bounded_discount@10 | 118 | 0.500 | 0.516 | 0.081 | 0.642 |
| receivable / do_nothing | 78 | 0.180 | 0.228 | 0.099 | 0.618 |
| receivable / automated_reminder | 71 | 0.437 | 0.328 | **0.164** | 0.688 |
| receivable / …with_payment_plan_offer | 63 | 0.540 | 0.401 | **0.138** | 0.752 |
| receivable / escalate_to_collections | 90 | 0.389 | 0.395 | 0.127 | 0.644 |

Three things a reader should take from this:

**The per-action samples are small.** Between 63 and 202 held-out events per
model. Every figure in that table carries a wide interval that is not printed,
and differences of a few points between rows should not be read as real.

**`offer_bounded_discount@5` has an AUC of 0.577.** That is barely above
chance. For that action the model is close to predicting a constant, and its
value to the system is almost entirely the *level* it predicts (which is
reasonably calibrated at 0.442 against 0.462 observed) rather than any ability to
tell one cart from another.

**The two receivables reminder models under-predict by 8 to 11 points.**
`automated_reminder` recovers 43.7% of the time and is predicted at 32.8%;
the payment-plan variant recovers 54.0% and is predicted at 40.1%. This is a
systematic bias in a consistent direction, and its consequence is specific: the
agent *under-values* the cheapest, least intrusive receivables actions, and will
therefore choose them less often than it should. It biases toward inaction,
which is the safe direction to be wrong in — but it is a bias, not conservatism
by design, and it should be fixed rather than described.

### Training volume

Between 216 and 659 training events per action model. The simulator randomises
which action was logged per event, so each model sees a genuine random subset
rather than a policy-selected one — which is what makes these estimates
unbiased for the population, and also what keeps them small.

---

## What the models cannot learn, on purpose

**No model has ever seen a retry against suspected fraud.**

`data/generate_payments.py` deliberately excludes them from the logged data, and
the comment there explains why: no real merchant randomises retries against
transactions their fraud system flagged, so that data would not exist in a real
log either. The consequence is stated plainly in the generator — *the uplift
model cannot learn "retrying fraud fails", because it never observes it.*

This is exactly why fraud handling is a hard guardrail rather than something
left to the arithmetic. Two independent mechanisms reach the same conclusion:

1. **The economics.** The chargeback term (`chargeback_cost_multiplier: 1.5`)
   makes every retry negative-value once fraud probability is high, and lifts
   `stop_and_flag_fraud` to the top of the ranking, before any rule is consulted.
2. **The guardrail.** `fraud_suspected` is on `never_retry_root_causes`, and
   retries are additionally blocked whenever the fraud *posterior* exceeds
   `max_fraud_probability_for_retry: 0.15` — reading the whole distribution, not
   its argmax, because `{insufficient_funds: 0.41, fraud_suspected: 0.39}`
   predicts "insufficient funds" and would otherwise sail through.

The redundancy is deliberate: a miscalibrated probability must not be able to
make retrying suspected fraud look profitable. Both mechanisms are tested
independently, in `test_fixtures.py` and `test_policy.py` respectively.

**No model may be fitted on an outcome column.** The simulator writes `po_*`
counterfactuals, `true_root_cause`, `logged_recovered` and `is_fraudulent` so the
benchmark can score against truth. `assert_no_leakage` refuses all of them by
prefix, wired into every feature list and asserted by tests that also guard the
guard. See `SECURITY.md`.

---

## What the models are used for, and what decides

The models produce probabilities. They do not choose anything.

```
root cause + confidence  ─┐
                          ├─→  ENR per candidate action  ─→  ranked  ─→  guardrails  ─→  chosen
P(recover | event, action)─┘         (economics.py)                    (guardrails.py)
```

Expected net recovery, per candidate, incremental against doing nothing:

```
ENR(a) = incremental_margin(a)
       − action_cost(a)
       − expected_failure_cost(a)
       − incremental_chargeback_cost(a)
       − contact_fatigue_penalty(a)
       + incremental_retained_ltv(a)
```

The discount is not a term of its own. It lives inside `incremental_margin`,
because a discount is given to everyone who converts — including the customers
who would have converted anyway — so the correct comparison is
`p_a × amount × (margin − discount)` against `p_0 × amount × margin`. Writing
the discount as a separate subtraction would charge it only against the
incremental converters and would therefore make discounting look cheaper than it
is. Every term is differenced against doing nothing, which is why `do_nothing`
scores exactly 0.0 and why a negative ENR means an action is worse than
inaction rather than merely worse than the alternatives. The authoritative form
is the docstring of `src/economics.py`.

Then guardrails run, and they can only ever say **no**. Economics proposes;
policy disposes. On the shipped sweep, policy refused 4,925 options and routed
396 decisions (21%) to a person.

The models therefore cannot cause an action that policy forbids, no matter how
wrong they are. A miscalibrated probability can cause a *worse choice among
permitted actions*, or an unnecessary human review, and that is the actual blast
radius of model error in this design. It is a deliberately small one.

Nothing in the diagram above is a language model, and that is the point of the
diagram. `src/narrator.py` sits strictly downstream of `chosen`: it renders a
decision that has already been made into prose, it has no tools, and no module
on the path from an event to an executed action imports it. `python -m src.agent
run` makes and executes every decision with no language model in the process.

---

## Why simple models are the right call here

The benchmark is the argument, and it does not flatter the agent.

On 1,844 held-out events, scored in realised rupees against counterfactual
outcomes with a 2,000-replicate paired bootstrap, the agent nets 3,802,294 INR.
It significantly beats doing nothing, compliant retry-everything, and its own
unattended variant. Against the better lawful baselines it is **not
statistically separable** — three of them post a higher point estimate, and
every one of those intervals crosses zero.

If sharper models were the lever, that is where it would show. They are not the
lever, because the binding constraints on this problem are not predictive:

- **Consent and quiet hours** remove candidate actions regardless of their value.
  A perfect model cannot WhatsApp someone who has not opted in.
- **The value basis** matters roughly 3× more than the probability. Pricing
  receivables at face value and carts at gross margin — because an overdue
  invoice covers goods already delivered and a cart does not — reprices two of
  three surfaces by about that factor. Getting that wrong dwarfs an AUC point.
- **The ceiling is low.** `oracle_per_event`, which sees each event's
  counterfactual outcome, nets 13,163,161 INR. That is the most any predictor
  could possibly extract, and the gap between it and the agent is mostly
  guardrails and consent, not model error.

So the models are built to be *honest and calibrated enough to price with*, and
the effort went into the pricing, the refusals, and the record. What the agent
adds over a statistical wash is 396 decisions routed to a person, 4,925 refusals
each with a stated reason, and a hash-linked record of every one — none of which
any baseline provides, and none of which a better classifier would.

---

## Limitations

**The data is synthetic.** Every probability, cost and outcome comes from
`data/generate_*.py`. The models are fitted on data produced by a generative
process, and held-out splitting does not fix the fact that the process is
knowable. Calibration measured this way is optimistic about the real world in a
way no split can correct.

**Chargeback exposure is barely exercised.** The simulator sets
`P(recover | fraud) ≈ 0.01`, so truly fraudulent payments are near-unrecoverable
and almost nothing "recovers" fraud to charge back. Realised chargeback cost
across the whole sweep is 2,972.64 INR for `do_nothing` and **0.00 for the
agent** — the one metric where the agent ties the oracle. The direction is
right and the mechanism is tested, but the magnitudes are trivial against
millions in recovery, and they would not be in production, where chargebacks
carry dispute-ratio consequences this simulator does not model.

**No temporal validation.** The split is random, not chronological. A real
deployment would need to know how fast these weights go stale, and this tells it
nothing.

**No fairness analysis.** Features include customer segment, tenure, prior
payment behaviour and estimated value. Nothing checks whether recovery effort is
distributed defensibly across segments, and on a real customer base that
question would need answering before deployment — a model that quietly tries
harder for enterprise accounts is a business decision, and it should be an
explicit one.

**Fourteen action models is a lot of small models.** Per-action models are the
honest way to estimate per-action effects, but they fragment the data. A single
model with action interactions would pool strength across actions and probably
calibrate better at these sample sizes. That is the first thing to try next.

**The single-run figures have no error bars.** The sweep numbers (recovery
totals, refusal counts) come from one run over one split. Only the benchmark
comparison is bootstrapped. Treat the sweep figures as a description of that run
rather than as estimates.

**Narration is a third-party model call, and this card does not cover it.** When
narration is enabled, a fact sheet derived from an already-made decision — the
surface, the customer segment, relationship length, prior payment count, contacts
in the last seven days, rupee amounts, the chosen action and the options policy
refused — is POSTed to whichever vendor `llm.provider` selects. It carries no
name, address, contact detail or account identifier, which is what makes sending
it to a third party defensible at all, but it is still relationship context
leaving the process. Nine providers are supported plus any OpenAI-compatible
endpoint, output is validated by the same gate regardless of which one is used,
and the vendor is recorded on each `Draft` rather than inferred. None of the
models this card describes are involved, and no decision depends on the call:
which vendor sees the fact sheet is an operator choice, not a property of these
models.

---

## Reproducing

```
python data/generate_all.py     # synthetic dataset, seeded
python -m src.train             # weights + artifacts/training_report.json
python -m src.benchmark         # baselines, bootstrap CIs
```

Both training and generation are seeded, so the figures above are reproducible.
`python -m src.train` prints the majority baseline next to every accuracy and
the reliability table next to every calibration figure, because an accuracy
without its baseline is not a result.
