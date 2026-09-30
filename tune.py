import sys, time, random
sys.path.insert(0, '/workspace/projects/pie-minimax')
from minmax import enumerate_reachable
import linear_expert as L
import numpy as np

states = enumerate_reachable(); rng = random.Random(0)
Btr, Mtr, _ = L.build(rng, 20000, states)
Bte, Mte, bte = L.build(rng, 8000, states)
base = L.evaluate(lambda B: np.random.default_rng(0).normal(0, 1, B.shape), Bte, Mte, bte)["top1_in_optimal"]
print(f"  random floor {base:.4f}\n")
print(f"  {'config':32} {'top-1':>8} {'recall':>8}   note")
for hidden, lr, steps in [(0,0.5,300),(0,0.1,800),(0,0.02,2000),
                          (8,0.5,300),(8,0.1,800),(8,0.02,2000),
                          (32,0.1,800),(32,0.02,2000)]:
    t=time.time()
    f = L.train(Btr, Mtr, hidden=hidden, steps=steps, lr=lr)
    r = L.evaluate(f, Bte, Mte, bte)
    n = 81 if hidden==0 else 9*hidden+hidden+9*hidden+9
    tag = f"h={hidden or 'linear'} lr={lr} steps={steps}"
    print(f"  {tag:32} {r['top1_in_optimal']:>8.4f} {r['set_recall']:>8.4f}   {n} params, {time.time()-t:.0f}s")
