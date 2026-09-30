"""Exact optimal play for tic-tac-toe, as data.

Not a model. The policy is computed, not learned, and that is the whole point: it is a
ground truth with no annotation noise, no disagreement between labelers, and no "what did
the expert mean". Every number measured against it is exact.

Representation. A board is 3x3 over {-1 (opponent), 0 (empty), +1 (us)}. Flattened to a
length-9 vector. A linear model is therefore a 9x9 map from board to a score per cell --
eighty-one parameters, and the ONLY way it can express a move preference is additively:
score[i] = sum_j W[i][j] * board[j].

That is worth stating plainly, because it is the hypothesis under test. Tic-tac-toe
optimal play is a MINIMAX over a game tree. A linear map over the board is a sum of
independent per-cell votes. There is no composition between cells, which is exactly what
minimax needs. So the prediction is: **a linear model gets the easy positions and fails the
hard ones**, and the failure will be concentrated in positions that require seeing a threat
somewhere other than where you are about to play.
"""
from __future__ import annotations
from functools import lru_cache
import random

# board: length-9 tuple of -1 (them), 0 (empty), +1 (us)
LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),      # cols
    (0, 4, 8), (2, 4, 6),                  # diagonals
)


def winner(b):
    for a, c, d in LINES:
        if b[a] == b[c] == b[d] != 0:
            return b[a]
    return 0


def moves(b):
    return [i for i in range(9) if b[i] == 0]


def play(b, mv):
    nb = list(b)
    nb[mv] = 1
    return tuple(nb)


def respond(b, mv):
    nb = list(b)
    nb[mv] = -1
    return tuple(nb)


def _opponent_best_after(b, m):
    """After WE play m, the opponent takes their best reply. Their best is our worst."""
    if winner(b) == 1:
        return 1
    replies = moves(b)
    if not replies:
        return 0
    return min(value(respond(b, r)) for r in replies)


@lru_cache(maxsize=None)
def value(b):
    """+1 we win with perfect play, 0 draw, -1 we lose. Exact, and it is OUR turn.

    The first version of this had `respond(play(b, m), m)` -- which let the opponent
    replay OUR OWN cell and overwrite our move. It returned -1 for the empty board, which
    is the classic tic-tac-toe draw, and would have poisoned every number measured against
    it. The ground truth has to be right or the whole exercise is measuring my typo."""
    w = winner(b)
    if w:
        return w
    legal = moves(b)
    if not legal:
        return 0
    return max(_opponent_best_after(play(b, m), m) for m in legal)


@lru_cache(maxsize=None)
def optimal(b):
    """The set of optimal moves -- a SET, not a choice.

    Several moves are frequently equally correct. A training set that picks one of them
    relabels the others as mistakes, and then you spend your budget teaching a model to
    avoid correct play."""
    w = winner(b)
    if w or not moves(b):
        return ()
    vals = {m: _opponent_best_after(play(b, m), m) for m in moves(b)}
    best = max(vals.values())
    return tuple(m for m, v in vals.items() if v == best)


def board_our_turn(b):
    """Only positions where it is our move -- the states a model would ever be asked about."""
    return moves(b) and (b.count(1) == b.count(-1))


def enumerate_reachable(max_plies=9):
    """Every position reachable with US to move, together with its exact optimal set.

    The walk alternates players. The first version only ever played as us, which filled
    the board with +1s, meant it was never our turn past the opening, and returned exactly
    one state -- the empty board -- while reporting no error. A walk that only explores
    one side of the tree is not a smaller result, it is a wrong one."""
    out = []
    def walk(b, plies, our_turn):
        if plies > max_plies:
            return
        if winner(b) or not moves(b):
            return
        if our_turn:
            opt = optimal(b)
            if opt:
                out.append((b, opt))
        step = play if our_turn else respond
        for m in moves(b):
            walk(step(b, m), plies + 1, not our_turn)
    walk((0,) * 9, 0, True)
    return out


def random_optimal_position(rng):
    """Sample a position we can actually face, with its optimal set."""
    pos = enumerate_reachable()
    return pos[rng.randrange(len(pos))]
