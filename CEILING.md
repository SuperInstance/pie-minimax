# The ceiling, and what the composition penalty actually is

**This file exists because a number without a ceiling is not a result, it is a number.**
`README.md` reported a linear 9×9 model at top-1 0.1807. That figure is uninterpretable on
its own: 0.1807 could be 10% of what is achievable or 90% of it. This measures the ceiling,
and the answer changes what the original number means.

## A correction first, because the dataset was miscounted

The earlier documents said **180,361 reachable our-turn states**. That number cannot exist.
A 3×3 board has 3⁹ = 19,683 distinct states, so even labelling every one with whose turn
gives at most 39,366. **180,361 exceeds the entire state space by a factor of 4.6.**

The real figures, from `minmax.enumerate_reachable()`:

| | |
|---|---|
| reachable board states | **5,478** |
| with US to move — the training set | **2,423** |
| with more than one optimal move | **1,177 (48.6%)** |
| optimal-set sizes | 1 to 9; 456 positions have 3 optimal moves |

**48.6%, not 14.7%.** That is the number that matters: a label set that picks one optimal
move relabels nearly half the dataset as errors. Every accuracy here is reported on the
**single-optimal subset** (n=1246) as well, where there is no tie-break to argue about.

## The ceiling, 5-fold cross-validated

Variance is across **data folds**, not across seeds. An earlier version reported `std == 0`
for every model because different random inits converge to the same solution — and
`std == 0` is INCONCLUSIVE, never a pass.

| model | top-1, all 2,423 | top-1, single-optimal only |
|---|---|---|
| random empty cell (**floor**) | 0.5753 ± 0.0212 | 0.4206 ± 0.0284 |
| **linear 9×9** (matched budget) | 0.7148 ± 0.0170 | 0.5708 ± 0.0175 |
| decision tree, depth 8 | 0.7590 ± 0.0253 | 0.6503 ± 0.0326 |
| **decision tree, depth 16** | **0.7879 ± 0.0224** | **0.6793 ± 0.0289** |
| exact solver (ceiling) | 1.0000 | 1.0000 |

**The linear model reaches 0.840 of the best tree's accuracy.** The tree is 1.62× the
floor; the linear model is 1.36×. So 0.1807 was never "a neural net is bad at minimax" —
it was "a linear map is bad at minimax", which is true and much less interesting, and it
was reported without the number that would have said so.

## The pre-registered prediction, and the half of it that survives

Positions were split by **COMPOSED-ness**: SIMPLE has at most one of your own lines one
stone from completing, which a single local term can express; COMPOSED has two or more, so
the right move depends on a *count over separate lines*.

**Raw accuracy is confounded, and the confound is in the chance line.** A position with two
of your own threats is structurally a *late* position — fewer empty cells — so chance is
already high and a fixed amount of skill moves the number less:

```
subset                     chance     tree   linear     tree fills  linear fills
SIMPLE   (0-1 own threat)    0.2547   0.6685   0.5509          55.5%         39.7%
COMPOSED (2+ own threats)    0.5395   0.8008   0.6435          56.7%         22.6%
```

`fills` is the fraction of the gap between chance and 1.0 that the model closes. Raw
accuracy says COMPOSED is *harder*; normalised, the tree's numbers are **indistinguishable**
(55.5% vs 56.7%).

**So the composition penalty is real for the linear model and absent for the tree** —
a real drop of roughly 43% of closed headroom, against no measurable drop at all for the
non-linear model.

**That is a sharper claim than "composition is hard".** It says the penalty is a property of
**additivity**, not of the task. A sum of local votes cannot represent a count over separate
lines; a model that can form intermediate conjunctions can, and does. Minimax composition is
not beyond these models — it is beyond *this representation*.

## What the GPU agent should take from this

1. **Report accuracy as a fraction of the ceiling**, not as a bare number. Every experiment
   in the queue needs this and most of them do not have it yet.
2. **Report top-1 and single-optimal accuracy.** With 48.6% of positions having several
   right answers, a single-move label set is measuring agreement with a tie-break.
3. **The SIMPLE/COMPOSED contrast must be depth-matched.** A position with two own threats
   is a late position, and the naive contrast therefore has the wrong sign. This is the
   obvious next experiment, and the reason it is worth running is that the obvious version
   of it produces the opposite conclusion.
4. **Variance must come from data resampling.** Seed variance was exactly zero across five
   seeds for every model here, and a zero-variance result is inconclusive, not a pass.

## Reproduce

```
python3 run3.py        # ~2 min, no dependencies beyond numpy
```
