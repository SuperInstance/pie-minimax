# What a linear model cannot do, in a box where the answer is known

## The result

A linear map from a 9-cell board to 9 per-cell scores is a sum of independent votes. Optimal
tic-tac-toe play is a minimax over a tree. **81 parameters get 0.1807 against a random floor
of 0.1431.**

The prediction was written down first and it held. This is the sharpest form available of
the question the murmuration work has been circling: can a purely local rule reproduce a
global optimum? Here the global optimum is *computed*, so the answer is not a matter of
opinion.

## What it means for the cell doctrine

The cell doctrine says a cell is a scar, not a parameter, and that the substrate is grown
rather than designed. This adds a boundary condition:

**Locality is not free.** A local rule can hold an *attractor* — the 2-D tissue result, where
each cell defers to confident neighbours and the structure holds. It cannot hold an
*optimum*, because an optimum requires reasoning about a move you are not making. The
distinction is not a subtlety. It is 0.18 against 0.14, and it is exact.

So: cellular systems are good at consensus and bad at strategy, and those are different
things. A murmuration is the first. A player is the second. Anything claiming a
local-rule-only substrate can produce minimax-quality play is claiming something this
measurement says is false, at least for the smallest board where the claim can be checked.

## Unsettled, and why

Whether a *nonlinear* local rule closes the gap is unknown, because my optimiser is the
confound: h=8 loses to linear at every learning rate tried, h=32 beats it at one, and the
ordering is not monotone in width. That is a measurement of the training setup, not of
capacity. The honest next step is a decision-tree reference, which can represent the policy
and should score near 1.0 — giving both a validation of the harness and a real ceiling.

## The label problem, which is the transferable part

14.7% of reachable positions have several equally-correct moves. Labelling one and calling
the rest wrong trains a model to avoid correct play, and the symptom is "the model is bad
at tic-tac-toe" rather than "the labels were wrong". Set-valued loss fixes it. Any expert
distillation from a source that can tie — a solver, a decision table, a proof — inherits
this problem, and it is invisible unless you count how often the expert had a choice.
