"""Can a LINEAR model reproduce the optimal policy? The smallest version of the question.

THE HYPOTHESIS UNDER TEST, stated before running it.

Tic-tac-toe optimal play is a MINIMAX over a game tree. A linear map from board to
per-cell score is a SUM OF INDEPENDENT PER-CELL VOTES:

    score[i] = sum_j W[i][j] * board[j]

There is no composition between cells. Minimax needs exactly that composition -- "if I play
here, they play there, and then I am losing on a line I did not see". A linear map cannot
represent it, because the score for cell i cannot depend on which cell j is jointly played
with it.

So the prediction is: a linear model gets the positions where a threat is LOCAL and fails
the positions where the threat is somewhere else. That is a sharp, falsifiable, and
extremely cheap prediction, and it is the cleanest model of "can a local rule be a global
expert" I can build -- which is the question the whole murmuration thread has been circling
without a ground truth to test against.

TRAINING LABELS ARE SETS, NOT MOVES. 14.7% of positions have several equally-correct
answers. Training on one of them teaches the model to avoid correct play on the others.
That is measured, not assumed, and it is the kind of detail that silently costs a training
run.
"""
from __future__ import annotations
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minmax import enumerate_reachable, value, optimal, winner


def build(rng, n_states, states):
    idx = [rng.randrange(len(states)) for _ in range(n_states)]
    B = np.array([[1.0 if states[i][0][j] == 1 else (-1.0 if states[i][0][j] == -1 else 0.0)
                   for j in range(9)] for i in idx], dtype=np.float64)
    M = np.zeros((len(idx), 9))
    for r, i in enumerate(idx):
        for m in states[i][1]:
            M[r, m] = 1.0
    return B, M, [states[i][0] for i in idx]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(B, M, steps=400, lr=0.5, seed=0, hidden=0, l2=1e-4, init=None):
    """hidden=0 is the linear model. hidden>0 is the same thing with one hidden layer,
    which is the minimal way to buy interaction between cells."""
    rng = np.random.default_rng(seed)
    n, d = B.shape
    if hidden:
        W1 = rng.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
        W2 = rng.normal(0, 0.5, (hidden, 9));  b2 = np.zeros(9)
        params = [W1, b1, W2, b2]
    else:
        W = init if init is not None else rng.normal(0, 0.1, (d, 9))
        params = [W]

    def forward(x):
        if hidden:
            h = np.maximum(0.0, x @ params[0] + params[1])
            return h @ params[2] + params[3]
        return x @ params[0]

    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        P = softmax(forward(B))
        # loss = -sum_r log( sum_{m in Opt(r)} P[r,m] )
        q = np.maximum((P * M).sum(axis=1, keepdims=True), 1e-12)
        # dL/dP[r,m] = -M[r,m] / q_r     (q_r = sum over the OPTIMAL SET)
        # The first version wrote (P * 1[M]).sum(1) / q, which is q/q == 1 everywhere --
        # an identically-constant gradient, so the model never moved and the "result"
        # was pure initialisation.
        G = M / q
        dZ = (P - G) / n
        if hidden:
            x = B; h = np.maximum(0.0, x @ params[0] + params[1])
            gW2 = h.T @ dZ; gb2 = dZ.sum(axis=0)
            dh = dZ @ params[2].T * (h > 0)
            gW1 = x.T @ dh; gb1 = dh.sum(axis=0)
            grads = [gW1 + l2 * params[0], gb1, gW2 + l2 * params[2], gb2]
        else:
            grads = [B.T @ dZ + l2 * params[0]]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** t); vh = v[i] / (1 - 0.999 ** t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
    return forward


def evaluate(score_fn, B, M, boards):
    S = score_fn(B)
    top1 = 0.0
    setrec = []
    nopt = []
    for r in range(len(B)):
        k = max(1, int(M[r].sum()))
        top = np.argsort(-S[r])[:k]
        opt = set(np.nonzero(M[r])[0])
        top1 += 1.0 if top[0] in opt else 0.0
        setrec.append(len(opt & set(top.tolist())) / len(opt))
        nopt.append(len(opt))
    return {"top1_in_optimal": float(top1) / len(B),
            "set_recall": float(np.mean(setrec)),
            "mean_optimal_set": float(np.mean(nopt))}


def is_threat_local(board, mv):
    """Heuristic used ONLY to split the evaluation set, never to train on it.

    A position is 'local' if the chosen optimal move is on the same row/column/diagonal as
    the opponent's biggest mark. If the linear model does much better on local positions
    than on non-local ones, that is the predicted failure mode showing up in the data."""
    me = [i for i in range(9) if board[i] == 1]
    them = [i for i in range(9) if board[i] == -1]
    if not them:
        return None
    threats = {i for i in range(9) for t in them for j in (i,) if (i + t) % 3 == 0}
    best = max(i for i in range(9) if board[i] == 1) if me else 0
    return bool(best in threats)
