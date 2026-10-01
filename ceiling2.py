"""The ceiling, measured properly.

THE METRIC, AND WHY THE FIRST ONE WAS WRONG. The first version scored top-1 as
"did the model pick `min(optimal_set)`". With 48.6% of positions having more than one
optimal move and sets running up to 9, that measures agreement with an arbitrary canonical
tie-break, not correctness. It produced a random floor of 0.6927 -- a "random" model
scoring 69% -- which is the tell that the metric, not the models, was broken.

The metrics used here:

  recall     the fraction of OPTIMAL moves the model's candidate set contains. The right
             primary number, because 48.6% of positions have more than one right answer.
  top1       the model's single best move is optimal at all. Same as recall for a
             single-candidate model, and it is NOT agreement with min().
  top1_norm  top1, divided by what a random pick from the optimal set would score on the
             same positions. This is the one number to compare models on, because it
             removes the tie-break's contribution from BOTH sides.
  floor      a uniformly random EMPTY cell, which is the honest baseline. A model that
             beats this is doing something; a model measured against a random pick from
             the optimal set is not, because that baseline is 1.0 by construction.

Every number carries a standard deviation over 5 seeds. A model whose std is 0 is
INCONCLUSIVE, never a pass -- and the first version of the tree did exactly that, scoring
an identical 0.2423 at every depth from 4 to 40.
"""
from __future__ import annotations
import random, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minmax import enumerate_reachable, moves, LINES

EMPTY = (0,) * 9


def positions():
    return {b: frozenset(v) for b, v in enumerate_reachable()}


def empty_cells(b):
    return [i for i in range(9) if b[i] == 0]


def threat_count(b, player):
    n = 0
    for i, j, k in LINES:
        c = (i, j, k)
        if b[i] == b[j] == b[k] == player:
            continue
        if sum(1 for x in c if b[x] == player) == 2:
            n += 1
    return n


def is_simple(b):
    """US to move. SIMPLE means at most one of our lines is one stone from completing,
    which a single local term can express. COMPOSED means the right move depends on a
    COUNT over separate lines, which a sum of local terms cannot represent.
    Pre-registered, and carried forward from the 3x3 result."""
    return threat_count(b, 1) <= 1


# ── decision tree with leaf DISTRIBUTIONS, so a leaf can propose a set of moves ──
class Tree:
    def __init__(self, max_depth=16, min_leaf=4, seed=0):
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.root = None
        self.seed = seed

    def _imp(self, dist):
        n = sum(dist.values())
        return 0.0 if n <= 0 else 1.0 - sum((v / n) ** 2 for v in dist.values())

    def _fit(self, X, y, idx, depth):
        dist = {}
        for i in idx:
            dist[y[i]] = dist.get(y[i], 0) + 1
        if depth >= self.max_depth or len(idx) <= self.min_leaf or len(dist) == 1:
            return ("leaf", dist)
        parent = self._imp(dist)
        best, gains = None, []
        for f in range(9):
            for thr in (-1, 0, 1):
                L = [i for i in idx if X[i, f] <= thr]
                R = [i for i in idx if X[i, f] > thr]
                if not L or not R:
                    continue
                dl, dr = {}, {}
                for i in L: dl[y[i]] = dl.get(y[i], 0) + 1
                for i in R: dr[y[i]] = dr.get(y[i], 0) + 1
                g = parent - (len(L) * self._imp(dl) + len(R) * self._imp(dr)) / len(idx)
                if g > 1e-12:
                    gains.append((g, f, thr, L, R))
        if not gains:
            return ("leaf", dist)
        g, f, thr, L, R = max(gains, key=lambda t: t[0])
        return ("node", f, thr, self._fit(X, y, L, depth + 1), self._fit(X, y, R, depth + 1))

    def fit(self, X, y, seed):
        rng = random.Random(seed)
        order = list(range(len(y)))
        rng.shuffle(order)          # so a random tie-break actually varies between seeds
        self.root = self._fit(X, y, order, 0)
        return self

    def leaf_dist(self, x):
        n = self.root
        while n[0] == "node":
            n = n[3] if x[n[1]] <= n[2] else n[4]
        return n[1]

    def scores(self, b):
        """Score every empty cell by the leaf distribution, minus a small cost for the
        cells that leaf never proposes, so unproposed cells do not tie at zero."""
        d = self.leaf_dist(b)
        tot = sum(d.values()) or 1
        s = {c: d.get(c, 0) / tot - 0.01 for c in empty_cells(b)}
        return s


def fit_linear(X, y, iters=3000, lr=0.4, seed=0, w=None):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(9, 9))
    Y = np.array(y)
    Wt = np.ones(len(Y)) if w is None else np.asarray(w, dtype=float)
    Wt = Wt / Wt.sum()
    for _ in range(iters):
        S = X @ W.T
        S[np.arange(len(Y)), Y] += 2.0
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        G = P.copy(); G[np.arange(len(Y)), Y] -= 1.0
        W -= lr * (G * Wt[:, None]).T @ X
    return W


def linear_scores(W, b):
    v = np.array(b, dtype=np.float64) @ W.T
    return {c: float(v[c]) for c in empty_cells(b)}


# ── metrics ─────────────────────────────────────────────────────────────────────
def evaluate(pos, scorer, seed=0, name=""):
    rng = random.Random(seed)
    recall = top1 = n = 0
    norm = 0.0
    for b, opts in pos.items():
        s = scorer(b)
        if not s:
            n += 1
            continue
        ranked = sorted(s, key=lambda c: -s[c])
        inter = [c for c in ranked if c in opts]
        n += 1
        recall += len(inter) / len(opts)
        if ranked[0] in opts:
            top1 += 1
        norm += (1.0 if ranked[0] in opts else 0.0) / (1.0 / len(opts))
    return {"recall": recall / n, "top1": top1 / n, "top1_norm": norm / n, "n": n}


def floor_scores(seed=0):
    rng = random.Random(seed)
    def f(b):
        e = empty_cells(b)
        sc = {c: rng.random() for c in e}
        return sc
    return f
