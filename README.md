# pie-minimax

Can a **local** rule reproduce a **global** optimum?

Tic-tac-toe is the smallest box in which that question has an **exact** answer. The optimal
policy is computed, not labelled — no annotation noise, no disagreement between experts, no
"what did they mean". Every number here is a fraction of a policy that is known to be
correct, so the measurement is sharp in a way nothing in this project has been so far.

This is rung one of the ladder: tic-tac-toe → connect four → Reversi/gomoku → blackjack.
Each rung is chosen so the ground truth stays exact or becomes stochastic-but-known.

## The result

**The prediction, stated before the run:** optimal play is a MINIMAX over a game tree. A
linear map from board to per-cell score is a *sum of independent per-cell votes* — there is
no composition between cells, which is exactly what minimax requires. So a linear model
should get local positions and fail the rest.

| model | params | top-1 optimal | set recall |
|---|---|---|---|
| random scorer (the floor) | — | 0.1431 | 0.1431 |
| **LINEAR 9→9** | **81** | **0.1807** | 0.1748 |

**81 parameters, and it barely beats guessing.** The prediction holds.

## The ground truth has a wrinkle worth knowing about

5,478 reachable states (2,423 with us to move), and **26,505 of them (14.7%) have more than one
equally-correct move.** Optimal sets run from 1 to 9 moves wide.

Training on *a* move rather than the *set* therefore relabels 14.7% of the data as error
and teaches the model to avoid correct play. The loss here is
`-log Σ_{m ∈ Opt} softmax(z)_m` for exactly that reason. This is the kind of detail that
silently costs a training run and presents as "the model is bad".

## Three bugs this file shipped before it worked

All three produced a plausible number rather than a crash, and all three are the same
family as everything else in this project:

1. **The minimax let the opponent replay our own cell** (`respond(play(b,m), m)`), so
   `value(empty_board)` returned −1 for what is a certain draw. Every number measured
   against it would have been measuring my typo.
2. **The tree walk only ever played as us**, filling the board with +1s, so it was never
   our turn past the opening and it reported **exactly one reachable state** — with no
   error. A walk that explores one side of the tree is not a smaller result, it is a wrong
   one.
3. **The gradient was identically constant**: `dL/dP` was written as
   `(P·1[opt]).sum(1) / q`, which is `q/q == 1` everywhere. The model never moved and the
   "result" was pure initialisation. Reported as `top1 = 322.0` because the metric was a
   sum labelled as a mean.

## What is NOT settled, and why

**"How much nonlinearity does optimal play need?" — unresolved, and I am not going to
publish a number I cannot stand behind.**

| config | params | top-1 |
|---|---|---|
| linear, lr 0.5 / 0.1 / 0.02 | 81 | 0.1807 / 0.1549 / 0.1449 |
| hidden h=8, lr 0.5 / 0.1 / 0.02 | 161 | 0.1207 / 0.1389 / 0.1020 |
| hidden h=32, lr 0.1 | 617 | 0.1969 |

h=8 is worse than linear at **every** setting and worse than random at two of them. h=32
beats linear at one. **A result that flips with learning rate and is non-monotone in width
is a measurement of my optimiser, not of model capacity.** Reporting "nonlinearity does not
help" here would be the same mistake as reporting "the judge is blind" in an earlier file
this session: mistaking a broken instrument for a property of the world.

**The next step is not a bigger model, it is a better-conditioned fit** — or better, a
reference that *can* represent the answer (a decision tree on the nine cells should reach
~1.0), which both validates the harness and gives the nonlinear models a ceiling to be
measured against rather than a random floor.

## Run it

```bash
python3 minmax.py          # the exact policy; importable, cached, 1.2s for all 180k states
python3 sweep.py           # linear vs hidden
python3 tune.py            # the learning-rate sweep that shows the confound
```


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
