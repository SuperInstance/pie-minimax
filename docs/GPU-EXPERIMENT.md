# GPU experiment brief — `pie-minimax`

**Does a network absorb minimax with zero search?**

This is the slice of the master queue that concerns this repository. The full document,
with all ten experiments and their decision trees, is at
[`SuperInstance/fleet-triage` → `docs/GPU-EXPERIMENTS.md`](https://github.com/SuperInstance/fleet-triage/blob/main/docs/GPU-EXPERIMENTS.md).

1. **State whether CUDA or CPU actually ran.** A CPU fallback reported as a GPU result is a
   fabricated measurement. Record the device string with the number.
2. **Report the variance, not just the mean.** If the measured quantity has `std == 0` over
   the sample, the result is **INCONCLUSIVE, never PASSED**. This is a hard rule: an earlier
   sharding experiment scored `PASSED` on the literal comparison `0 < 0`.
3. **The ground truth must be computed, not asserted.** Every game number below is exact.
   If a label is approximate, say by how much.
4. **Non-degeneracy is a precondition, not a result.** Assert that the data has variance
   *before* evaluating any relational claim about it.
5. **A control must vary the thing it audits, by a different path than the audited thing.**
   The most expensive mistake in this project's history: a control built from the same call
   path as the probes, which therefore confirmed a fault instead of auditing it.
6. **A ratio whose denominator can be zero is a construction, not a measurement.** It always
   produces a finding, and the finding is always false.
7. **Seed everything and record the seed.** Two runs of the same experiment that differ are
   a finding about the seed, not about the method.
8. **Cross-port arithmetic must preserve the reference loop/summation order.** Float addition
   is not associative; a port that vectorises changes the answer.

---

### Experiment 1 — Can a network absorb minimax composition, with zero search?

**Repo:** `pie-minimax` · **Ground truth:** exact, 5,478 reachable states (2,423 with us to move)
**Result so far:** a 9→9 linear model, 81 parameters, top-1 **0.1807**. Random floor
**0.1431**. Set-recall 0.1748. **48.6% of the 2,423 us-to-move states have multiple optimal moves.**

The linear model beats chance by 0.037. That is a small, real, *uninteresting* margin.

| Result | What it means |
|---|---|
| Linear stays ≈ 0.18 after any fix | **The representation cannot express minimax.** Minimax is a composition of local threats; a sum of local terms has no way to represent "two simultaneous wins" because that is a *count*, not a sum. This is a strong, publishable negative. |
| Nonlinear closes a *little* (0.25–0.40) | Capacity helps a bit; representation is still the binding constraint. Report the gap to the decision-tree ceiling — **the number that matters is the ratio to that ceiling, not the raw accuracy.** |
| Nonlinear approaches the tree ceiling | Minimax composition is learnable by a small net. Surprising, and a real result. |
| **Any model beats the decision-tree ceiling** | **Stop. Something is wrong with the labels, not the model.** The tree is an exact computation. Beating it means the evaluation leaked. This branch is the most important one. |
| Accuracy collapses on COMPOSED states (≥2 simultaneous wins) | **The prediction carried forward from 3×3 holds.** You have isolated the mechanism: local voting handles single threats, fails on threat *counts*. This is the cleanest result available in the whole queue. |

**Pre-registered test, carried forward:** partition states into **SIMPLE** (0 or 1 immediate
win available — expressible by one linear term) and **COMPOSED** (≥2 simultaneous wins —
requires counting). If accuracy collapses on COMPOSED, you have found the mechanism, not
just a number.

### Experiment 2 — The decision-tree ceiling

**Not started, and it gates Experiment 1.** No conclusion about neural capacity is valid
without it. Fit a decision tree to the same 5,478 reachable states (2,423 with us to move) and measure its top-1. A shallow
tree, if it beats the linear model substantially, says the task is *nonlinearly separable but
shallowly structured* — a different and more actionable claim than "neural nets are bad at
this." A deep tree matching the linear model says minimax is not a simple function of local
structure at all.

**Without this, Experiment 1's numbers are uninterpretable.** Do this first. It is CPU work.

---

## 4. Reporting format

Every experiment reports, in this order:

1. **Device.** Did CUDA actually run? Paste the device string.
2. **Data provenance.** Which commit, which digest, how many states/positions, and the
   FNV-1a 64 of the input file.
3. **The ceiling.** What is perfect, and what did you get as a fraction of it.
4. **Variance.** Mean ± std over N seeds, with the seeds listed. **std == 0 means INCONCLUSIVE.**
5. **The branch.** Which row of which decision tree above you landed on, quoted.
6. **Controls.** What ran that could have failed. If nothing could have failed, say that —
   it is the finding.


---

## CORRECTION — the state count was impossible

This document previously said **180,361 reachable our-turn states**. That number cannot exist: a 3x3 board has 3^9 = 19,683 distinct states, so even labelling every one with whose turn gives at most 39,366. **180,361 is larger than the entire state space by a factor of 4.6.**

The real numbers, from the repo's own `enumerate_reachable()`:

- **5,478** reachable board states in total
- **2,423** of them with US to move — this is the training set size
- **1,177 (48.6%)** have more than one optimal move, not 14.7%
- optimal-set sizes run 1 to 9; 456 positions have 3 optimal moves, 116 have 5

The multi-optimal fraction matters more than the raw count: **a label set that picks one optimal move relabels 48.6% of positions as errors.** Any accuracy measured with single-move labels on this dataset is measuring agreement with an arbitrary tie-break, not correctness.

The number was wrong in the same way as the fleet's `historybloat` signal — a figure that nobody checked against the size of the space it claims to count. **3^9 = 19,683 is a fact you can check in your head; the number should have been checked against it before it was written down.**
