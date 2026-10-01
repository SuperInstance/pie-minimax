import os, sys, random, json
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, "/home/eileen/projects/pie-minimax")
import numpy as np, torch, torch.nn as nn
from minmax import enumerate_reachable, winner, play
import linear_expert as L
torch.use_deterministic_algorithms(True)
DEV="cuda:0"
states = enumerate_reachable()
uniq={}
for b,o in states: uniq[b]=o
boards=sorted(uniq.keys())
def wins(b): return [m for m in range(9) if b[m]==0 and winner(play(b,m))==1]
rng0=random.Random(7); rng0.shuffle(boards)
folds=[boards[i::5] for i in range(5)]
accs=[]
for k in range(5):
    test=folds[k]; train=[b for i in range(5) if i!=k for b in folds[i]]
    seed=100+k; random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng=random.Random(seed)
    train_list=[(b,uniq[b]) for b in train for _ in range(max(1,20000//len(train)))]
    Btr=np.array([[float(b[j]) for j in range(9)] for b,_ in train_list])
    Mtr=np.zeros((len(train_list),9))
    for r,(b,o) in enumerate(train_list):
        for m in o: Mtr[r,m]=1.0
    Bte=np.array([[float(b[j]) for j in range(9)] for b in test])
    comp=[len(wins(b))>=2 for b in test]
    Btr_t=torch.tensor(Btr,dtype=torch.float32,device=DEV); Mtr_t=torch.tensor(Mtr,dtype=torch.float32,device=DEV)
    Bte_t=torch.tensor(Bte,dtype=torch.float32,device=DEV)
    model=nn.Sequential(nn.Linear(9,64),nn.ReLU(),nn.Linear(64,9)).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=1e-2)
    best, best_state, since=float("inf"), None, 0
    for step in range(4000):
        P=torch.softmax(model(Btr_t),dim=1)
        loss=-torch.log((P*Mtr_t).sum(1).clamp_min(1e-12)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        l=loss.item()
        if l<best-1e-6: best, since=l,0; best_state={kk:v.detach().clone() for kk,v in model.state_dict().items()}
        else:
            since+=1
            if since>=200: break
    model.load_state_dict(best_state)
    with torch.no_grad():
        top=model(Bte_t).argmax(1).cpu().numpy()
    hit=[top[r] in set(uniq[test[r]]) for r in range(len(test))]
    accs.append(float(np.mean(hit)))
    chit=[hit[r] for r in range(len(test)) if comp[r]]
    print(f"fold {k}: n={len(test)} top1={np.mean(hit):.4f} composed_n={len(chit)} composed={np.mean(chit) if chit else float('nan'):.4f}")
print("5-fold mean:", np.mean(accs), "std:", np.std(accs))
