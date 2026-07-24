"""Experiment 6 — graph-encoded photophysics for cross-platform transfer.

Three estimators: (i) unconditioned, trained on NV + hBN; (ii) graph-
conditioned (level-structure template embedded by a message-passing
encoder), trained on NV + hBN; (iii) oracle, trained on all four
platforms.  Zero-shot evaluation on GaN and SiV — platforms never seen
by (i) and (ii) — quantifies how much physics-graph conditioning recovers
of the transfer gap.

Writes results/exp6_graph.json.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/sparq")
from sparq.datasets import make_batch, make_eval_set, CFG
from sparq.estimators import HistCNN, balanced_accuracy
from sparq.gnn import GraphEncoder, template_graph
from sparq.physics import sample_site

t0 = time.time()
STEPS = 1300
BATCH = 256
TRAIN_PLATFORMS = ["NV", "hBN"]
ALL_PLATFORMS = ["NV", "hBN", "GaN", "SiV"]
T_EVAL = [0.3, 3.0]
N_EVAL = 1200
N_SEEDS = 3
COND = 8


def gen_mixed(rng, batch, platforms):
    per = batch // len(platforms)
    parts = [make_batch(rng, per, platform=p) for p in platforms]
    out = {}
    for k in ("hist", "aux", "y_cls", "y_g2", "y_valid"):
        out[k] = np.concatenate([p[k] for p in parts])
    out["platforms"] = sum([[pl] * per for pl in platforms], [])
    return out


def train(conditioned, platforms, seed):
    torch.manual_seed(seed)
    net = HistCNN(CFG.n_bins, cond_dim=COND if conditioned else 0)
    enc = GraphEncoder(out_dim=COND) if conditioned else None
    params = list(net.parameters()) + (list(enc.parameters()) if enc else [])
    opt = torch.optim.Adam(params, 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(seed)
    graphs = {p: tuple(torch.from_numpy(x) for x in template_graph(p))
              for p in ALL_PLATFORMS}
    for step in range(STEPS):
        b = gen_mixed(rng, BATCH, platforms)
        hist = torch.from_numpy(b["hist"])
        aux = torch.from_numpy(b["aux"])
        y = torch.from_numpy(b["y_cls"])
        yg = torch.from_numpy(b["y_g2"])
        cond = None
        if conditioned:
            embs = {p: enc(*graphs[p]) for p in set(b["platforms"])}
            cond = torch.stack([embs[p] for p in b["platforms"]])
        vv = torch.from_numpy(b["y_valid"])
        logits, reg = net(hist, aux, cond)
        loss = F.cross_entropy(logits[vv], y[vv]) \
            + 0.5 * F.mse_loss(reg, yg)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 300 == 0:
            print(f"  step {step} loss {float(loss):.4f}")
    net.eval()
    return net, enc


@torch.no_grad()
def eval_platform(net, enc, platform, T, seed):
    site_rng = np.random.default_rng(500 + seed)
    sites = [sample_site(site_rng, platform) for _ in range(N_EVAL)]
    r = np.random.default_rng(900 + seed)
    ev = make_eval_set(r, N_EVAL, T, platform=platform, sites=sites)
    hist = torch.from_numpy(ev["hist"])
    aux = torch.from_numpy(ev["aux"])
    cond = None
    if enc is not None:
        z = enc(*[torch.from_numpy(x) for x in template_graph(platform)])
        cond = z[None].repeat(N_EVAL, 1)
    logits, reg = net(hist, aux, cond)
    vv = ev["y_valid"]
    acc = balanced_accuracy(ev["y_cls"][vv], logits.argmax(1).numpy()[vv])
    mae = float(np.mean(np.abs(reg.numpy() - ev["y_g2"])))
    return acc, mae


models = {}
print("== unconditioned, NV+hBN")
models["uncond_2p"] = train(False, TRAIN_PLATFORMS, 21)
print("== graph-conditioned, NV+hBN")
models["graph_2p"] = train(True, TRAIN_PLATFORMS, 21)
print("== oracle, all platforms")
models["oracle_4p"] = train(False, ALL_PLATFORMS, 21)

results = {}
for name, (net, enc) in models.items():
    results[name] = {}
    for p in ALL_PLATFORMS:
        for T in T_EVAL:
            accs, maes = [], []
            for s in range(N_SEEDS):
                a, m = eval_platform(net, enc, p, T, s)
                accs.append(a); maes.append(m)
            results[name][f"{p}@{T}"] = dict(
                acc=[float(np.mean(accs)), float(np.std(accs))],
                mae=[float(np.mean(maes)), float(np.std(maes))])
    print(name, {k: round(v['acc'][0], 3) for k, v in results[name].items()})

# transfer-gap recovery on unseen platforms (averaged over T and platform)
def mean_unseen(name):
    vals = [results[name][f"{p}@{T}"]["acc"][0]
            for p in ("GaN", "SiV") for T in T_EVAL]
    return float(np.mean(vals))

gap_recovery = ((mean_unseen("graph_2p") - mean_unseen("uncond_2p")) /
                max(mean_unseen("oracle_4p") - mean_unseen("uncond_2p"), 1e-9))
out = dict(results=results, steps=STEPS,
           unseen=dict(uncond=mean_unseen("uncond_2p"),
                       graph=mean_unseen("graph_2p"),
                       oracle=mean_unseen("oracle_4p"),
                       gap_recovery=float(gap_recovery)))
print("unseen-platform summary:", out["unseen"])
with open("/home/claude/sparq/results/exp6_graph.json", "w") as f:
    json.dump(out, f)

# save the graph-conditioned model for the figure/paper
torch.save(models["graph_2p"][0].state_dict(),
           "/home/claude/sparq/results/models/cnn_graph.pt")
print(f"saved ({time.time()-t0:.0f}s)")
