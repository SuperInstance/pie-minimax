import sys, time, random
sys.path.insert(0, '/workspace/projects/pie-minimax')
from minmax import enumerate_reachable
import linear_expert as L
import numpy as np

states = enumerate_reachable()
rng = random.Random(0)
Btr, Mtr, _ = L.build(rng, 20000, states)
Bte, Mte, bte = L.build(rng, 8000, states)

print("  how much NONLINEARITY does optimal play need?\n")
print(f"  {'model':36} {'params':>7} {'secs':>6} {'top-1':>8} {'recall':>8}")
b = L.evaluate(lambda B: np.random.default_rng(0).normal(0, 1, B.shape), Bte, Mte, bte)
print(f"  {'random scorer (the floor)':36} {'-':>7} {'-':>6} {b['top1_in_optimal']:>8.4f} {b['set_recall']:>8.4f}")
t = time.time(); f = L.train(Btr, Mtr, hidden=0, steps=300, lr=0.5); r = L.evaluate(f, Bte, Mte, bte)
print(f"  {'LINEAR 9->9':36} {81:>7} {time.time()-t:>6.1f} {r['top1_in_optimal']:>8.4f} {r['set_recall']:>8.4f}")
for h in (8, 32):
    t = time.time(); f = L.train(Btr, Mtr, hidden=h, steps=300, lr=0.5)
    n = 9*h + h + 9*h + 9; r = L.evaluate(f, Bte, Mte, bte)
    print(f"  {'+ hidden h='+str(h):36} {n:>7} {time.time()-t:>6.1f} {r['top1_in_optimal']:>8.4f} {r['set_recall']:>8.4f}")
