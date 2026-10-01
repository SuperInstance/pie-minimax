"""pie-minimax A1 closure — MLP 9->64->9 on exact minimax labels, CUDA, seeds [1,2,3].

Pre-registered spec: SuperInstance/fleet-triage docs/RTX4050-BUILDSPECS.md
"pie-minimax A1 build spec" (PR #1, merged 2026-10-01).

P1: global top-1 in [0.25, 0.40]       (PASS iff mean mlp_top1_global in range)
P2: composed < 70% of global            (PASS iff mlp_top1_composed < 0.70 * global)
Frozen mapping. No goalpost migration. Report either way.

Protocol parity with the reproduced baseline (sweep.py):
  - labels: enumerate_reachable() via linear_expert.build(), set-valued
  - loss:   -log sum_{m in Opt} softmax(z)_m   (same as linear_expert.train)
  - split:  20,000 train / 8,000 test sampled with replacement from the
            180,361-row path artifact, rng = random.Random(seed)  (sweep used 0;
            here the seed also drives sampling, torch/numpy/cuda, per spec
            "seeds [1,2,3] pinned everywhere")
  - eval:   top1 = argmax in optimal set (identical to linear_expert.evaluate)

Early stop on train-loss plateau; restore best-loss checkpoint. Never trained
past plateau. INSTRUMENT-01: >=0.6s sustained synced CUDA ramp before timing.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, time, json, random
sys.path.insert(0, "/home/eileen/projects/pie-minimax")
import numpy as np
import torch
import torch.nn as nn

from minmax import enumerate_reachable, winner, play
import linear_expert as L

assert torch.cuda.is_available(), "CUDA required — never report CPU as GPU"
DEVICE = "cuda:0"
DEVSTRING = "cuda:0 (RTX 4050, 6GB)"
torch.use_deterministic_algorithms(True)
SEEDS = [1, 2, 3]
N_TRAIN, N_TEST = 20000, 8000          # sweep.py split sizes
LR, MAX_STEPS, PATIENCE, MIN_DELTA = 1e-2, 4000, 200, 1e-6


def immediate_wins(b):
    return [m for m in range(9) if b[m] == 0 and winner(play(b, m)) == 1]


def composed_mask(boards):
    return np.array([len(immediate_wins(b)) >= 2 for b in boards])


def gpu_ramp(seconds=0.6):
    """INSTRUMENT-01: sustained synced CUDA load before any timing."""
    t0 = time.time()
    a = torch.randn(4096, 4096, device=DEVICE)
    while time.time() - t0 < seconds:
        a = a @ a * 1e-3
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    return time.time() - t0


def evaluate_torch(model, B, M):
    model.eval()
    with torch.no_grad():
        S = model(B)
        top = S.argmax(dim=1)
        opt_sets = [np.nonzero(M[r])[0] for r in range(len(B))]
        top1 = float(np.mean([top[r].item() in set(opt_sets[r].tolist()) for r in range(len(B))]))
    return top1


def run_seed(seed, states):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    Btr, Mtr, _ = L.build(rng, N_TRAIN, states)
    Bte, Mte, bte = L.build(rng, N_TEST, states)
    comp_te = composed_mask(bte)

    Btr_t = torch.tensor(Btr, dtype=torch.float32, device=DEVICE)
    Mtr_t = torch.tensor(Mtr, dtype=torch.float32, device=DEVICE)
    Bte_t = torch.tensor(Bte, dtype=torch.float32, device=DEVICE)

    model = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 9)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    torch.cuda.synchronize()
    t0 = time.time()
    best_loss, best_state, since = float("inf"), None, 0
    stopped_at = MAX_STEPS
    for step in range(MAX_STEPS):
        model.train()
        P = torch.softmax(model(Btr_t), dim=1)
        q = (P * Mtr_t).sum(dim=1).clamp_min(1e-12)
        loss = -torch.log(q).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        l = loss.item()
        if l < best_loss - MIN_DELTA:
            best_loss, since = l, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                stopped_at = step + 1
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.cuda.synchronize()
    train_secs = time.time() - t0

    top1_global = evaluate_torch(model, Bte_t, Mte)
    if comp_te.sum() > 0:
        top1_comp = evaluate_torch(model, Bte_t[comp_te], Mte[comp_te])
    else:
        top1_comp = float("nan")
    return {
        "seed": seed, "params": n_params, "steps_run": stopped_at,
        "final_train_loss": best_loss, "train_secs": train_secs,
        "top1_global": top1_global, "top1_composed": top1_comp,
        "composed_test_n": int(comp_te.sum()),
        "composed_test_share": float(comp_te.mean()),
    }


def main():
    states = enumerate_reachable()
    assert len(states) == 180361, f"path artifact size changed: {len(states)}"
    uniq = {}
    for b, opt in states:
        uniq[b] = opt
    comp_universe = sum(1 for b in uniq if len(immediate_wins(b)) >= 2)
    composed_share_universe = comp_universe / len(uniq)

    ramp = gpu_ramp(0.6)
    wall0 = time.time()
    runs = [run_seed(s, states) for s in SEEDS]
    wall = time.time() - wall0

    glob = [r["top1_global"] for r in runs]
    comp = [r["top1_composed"] for r in runs]
    mean_g, mean_c = float(np.mean(glob)), float(np.mean(comp))

    p1 = "PASS" if 0.25 <= mean_g <= 0.40 else ("FAIL-HIGH" if mean_g > 0.40 else "FAIL-LOW")
    p2 = "PASS" if mean_c < 0.70 * mean_g else "FAIL"
    if mean_g > 0.6:
        note = "global top-1 > 0.6: composition thesis weakens (report either way)"
    else:
        note = ""

    receipt = {
        "experiment": "pie-minimax-closure-mlp",
        "pre_registered": "2026-10-01",
        "predictions": {"P1": "global top-1 in [0.25,0.40]", "P2": "composed < 70% of global"},
        "device": DEVSTRING,
        "seeds": SEEDS,
        "results": {
            "linear_top1": 0.1807,
            "mlp_top1_global": round(mean_g, 4),
            "mlp_top1_composed": round(mean_c, 4),
            "composed_share": round(composed_share_universe, 4),
        },
        "verdict": {"P1": p1, "P2": p2},
        "details": {
            "linear_reproduction": {"top1": 0.1807, "floor": 0.1431, "set_recall": 0.1748,
                                    "protocol": "sweep.py, seed 0, exact match"},
            "state_space": {"path_rows": 180361, "distinct_our_turn_boards": len(uniq),
                            "composed_boards": comp_universe,
                            "composed_path_weighted_rows": 11520},
            "mlp": {"arch": "9-64-9 relu", "params": runs[0]["params"], "lr": LR,
                    "loss": "set-valued -log sum_{m in Opt} softmax(z)_m",
                    "early_stop": f"patience {PATIENCE} on train loss, min_delta {MIN_DELTA}",
                    "optimizer": "Adam full-batch"},
            "per_seed": [{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r.items()} for r in runs],
            "std_global": float(np.std(glob)), "std_composed": float(np.std(comp)),
            "verdict_note": note,
            "gpu_ramp_seconds": round(ramp, 3),
            "wall_clock_seconds": round(wall, 1),
        },
    }
    out = "/home/eileen/projects/quilt-gpu-lab/results/pie_minimax_closure.json"
    with open(out, "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
