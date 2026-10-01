# CEILING-VERIFY — wave-63 multi-beam verification of the decision-tree ceiling

Verifier: independent SuperInstance agent (cross-agent study lane), 2026-10-01.
Scope: Experiment 2 (`fleet-triage/docs/GPU-EXPERIMENTS.md` §Experiment 2, "Do this first. It is CPU work")
plus the pre-registered SIMPLE/COMPOSED partition it feeds. Multi-beam per fleet doctrine:
"a beam that disagrees with the chart is not a malfunction — it is the fish."

Device: CPU-only container, 2 logical cores, numpy 2.1.3, sklearn 1.5.2, Python 3.
Receipts: `receipts/exp2-multibeam-verify.json` (v1), `-v2.json`, `-v3.json`,
pinned dataset dump `receipts/exp2-distinct-boards.txt`.

---

## 1. The ceiling: VERIFIED (multi-beam)

5-fold CV on the 2,423 distinct our-turn boards (Random(0) shuffle, folds `keys[i::5]`),
top-1 = the model's single pick (among EMPTY cells) is in the optimal set.

| model | all 2,423 | single-optimal (n=1,246) | source |
|---|---|---|---|
| custom Gini tree, depth 16 | **0.7879 ± 0.0224** | 0.6793 ± 0.0289 | committed `run3.py` protocol, re-run this wave |
| sklearn DecisionTree gini-16, empty-aware scoring | **0.7907 ± 0.0313** | 0.6915 ± 0.0400 | independent implementation, registered band ±0.03 → PASS (Δ 0.0028) |
| custom linear 9×9 (matched budget) | **0.7148 ± 0.0170** | 0.5708 ± 0.0175 | committed protocol, re-run — reproduces EXACTLY |
| sklearn LogisticRegression, empty-aware scoring | **0.6888 ± 0.0119** | 0.5825 ± 0.0204 | independent implementation, registered band ±0.05 → PASS (Δ 0.026) |
| floor (uniform random EMPTY cell) | 0.5753 ± 0.0212 | 0.4206 ± 0.0284 | unchanged from run3 |
| exact solver | 1.0000 | 1.0000 | by construction |

Registered decision-tree branch landed (Exp 2 table): **tree ≫ linear on distinct boards,
tree saturates by depth 12** → "nonlinearly separable but shallowly structured." The
0.1807 headline is retired with its ceiling: linear fills 0.7148/0.7879 = **0.908 of the
best tree** under the corrected protocol (run3's own 0.840 ratio used tree-single 0.6793
vs linear-single 0.5708 = 0.840; both bookings agree the old "neural nets are bad at
minimax" reading is dead).

## 2. Methodological receipts (each one caught a live instrument defect)

**M1 — empty-cell-aware scoring is load-bearing.** A beam that argmaxes sklearn's
`predict` over all 9 classes scores **0.6416** (tree) / **0.2637** (logreg — below the
floor) because occupied-cell predictions are auto-misses the game forbids. Scoring
max P(cell) over EMPTY cells only restores agreement (0.7907 / 0.6888). Any future
beam must respect the move-legality constraint at scoring time; v2's "disagreement"
was this artifact, root-caused, receipted in `-v2.json`.

**M2 — the blocked-line partition bug is real and large.** `is_simple()` counts a line
with 2 own marks as a threat even when the third cell is OPPONENT-occupied (a dead
line). Fixed definition (2 own + 1 EMPTY, no opponent): full-set COMPOSED drops
1,230 → 342; single-optimal COMPOSED drops 718 → **22, and all 22 are trivial**
(exactly one empty cell ⇒ chance = 1.0; filled headroom undefined).

**M3 — the composition-collapse story does not survive the fix at 3×3.** Under the
corrected definition there are essentially NO non-trivial COMPOSED single-optimal
boards, so the pre-registered Experiment-1 COMPOSED-collapse test is NOT executable
as specified on this dataset (LOW_POWER, n=22 trivial). On the full board set the
fixed-definition contrast REVERSES: both models fill MORE headroom on COMPOSED
(tree 0.463→0.823, linear 0.298→0.493) — COMPOSED boards are late boards where the
open threat structure itself is informative. What the old definition was detecting
was **pressure** (blocked lines), not composition. The run3 proposal of a
"depth-matched COMPOSED-vs-SIMPLE contrast" must be re-derived under the fixed
definition before any GPU experiment consumes it.

**M4 — partition numbers depend on training protocol.** run3's partition section
trains class-internally (COMPOSED models see only COMPOSED boards); my beams train
full and evaluate per class. Beam A (class-internal, old def): linear 39.7% → 22.6%
filled. Beam B (full-train, old def): linear 28.1% → 24.2%. Beam C (full-train, fixed
def): no penalty, reversed. Tree shows no composition penalty in ALL THREE beams —
that part is robust. The linear penalty is protocol-sensitive and deflates from
"a real 43% loss" to small-or-absent once the chance basis and blocked lines are
handled. Booked as an open question with all three receipts attached.

**M5 — hygiene receipt on run3.py.** The printed headroom table is hardcoded
constants (`rows2`), not computed in-script; it can silently inherit a stale run.
Also `sweep.py`/`tune.py` hard-code `sys.path.insert('/workspace/projects/...')`.

**M6 — dataset digests.** The 63-d study's digests (distinct-board FNV-1a 64
`0x110ce15d74ea1781`, sha256 `a4fc7dfb…`) are **UNREPRODUCED**: 8+ canonicalization
variants tried, none match. Our canon is now pinned in
`receipts/exp2-distinct-boards.txt` (sorted lines; board chars x/o/.; `|`;
sorted optimal moves; trailing newline): sha256
`ae19d8ad4a2c070535e682cbbb69d3e73e276f3fdcde0ca294e3d827e82c13f9`,
FNV-1a 64 `0x65a75b94d9804cfd`.
A code-generated dataset has no digest until someone writes one; now it does.
The path-weighted list (180,361 entries) remains the wrong population for
measurement (63-d's own finding, confirmed here: 2,423 distinct boards is the
honest denominator; multi-optimal = 48.6% per board, not 14.7% per path).

## 3. What this changes upstream

- `GPU-EXPERIMENTS.md` Exp 2 row: **not started → DONE (CPU), VERIFIED-MULTIBEAM.**
- Exp 1's "pre-registered test carried forward" (SIMPLE/COMPOSED partition) needs the
  M2/M3 correction before the 4×4 rung replicates it; otherwise 4×4 inherits a
  partition that measures pressure, not composition.
- Exp 4's curve should plot against tree-16 ≈ 0.79 (all boards) / 0.68 (single) ceilings.

## 4. Controls

- Digest canary: FNV-1a 64 implementation checked against the fleet constant
  `fnv1a64("café Δ 日本語") = 0x24a555471370b18d` before use.
- Reproduction beam ran the committed code path unmodified (means AND stds exact).
- Negative control for M1: the v2 receipt retains the below-floor 0.2637 logreg number
  with its diagnosis, so the artifact class is searchable later.
