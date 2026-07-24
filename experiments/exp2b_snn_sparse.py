"""Experiment 2b — activity-regularized (sparse) spiking estimator.

Identical to the exp2 SNN but trained with a spike-rate penalty (standard
neuromorphic practice) so hidden activity is event-driven rather than
dense. Re-evaluates the accuracy sweep, anytime latency, and measured
energy; results supersede the exp2 SNN rows via make_numbers.

Writes results/exp2b_snn.json and models/snn_sparse.pt.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/sparq")
from sparq.datasets import make_batch, make_eval_set, CFG, N_SLICES
from sparq.estimators import (SpikingG2Net, evaluate, balanced_accuracy,
                              HistCNN)
from sparq.physics import sample_site

t0 = time.time()
STEPS = 1200
BATCH = 224
LAMBDA_SPK = 0.6          # penalty on mean activity fraction
T_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
N_EVAL = 1200
N_SEEDS = 5
TARGET_ACC = 0.90

torch.manual_seed(3)
snn = SpikingG2Net(CFG.n_bins)
opt = torch.optim.Adam(snn.parameters(), 1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
rng = np.random.default_rng(3)

for step in range(STEPS):
    b = make_batch(rng, BATCH, T_dist=("logu", 0.03, 30.0), platform="NV")
    stream = torch.from_numpy(b["stream"])
    aux = torch.from_numpy(b["aux"])
    y = torch.from_numpy(b["y_cls"])
    yg = torch.from_numpy(b["y_g2"])
    v = torch.from_numpy(b["y_valid"])
    logits, reg, n_spk = snn(stream, aux, count_spikes=True)
    S = logits.shape[1]
    logits_v, y_v = logits[v], y[v]
    sl = list(range(3, S, 4)) + [S - 1]          # supervision checkpoints
    w = torch.linspace(0.2, 1.0, len(sl))
    w = w / w.sum()
    ce = sum(wi * F.cross_entropy(logits_v[:, s], y_v)
             for wi, s in zip(w, sl))
    activity = n_spk / (BATCH * S * (snn.h1 + snn.h2))
    lam = LAMBDA_SPK * min(1.0, step / (0.3 * STEPS))   # ramp in
    loss = (ce + 0.5 * F.mse_loss(reg[:, -1], yg)
            + 0.3 * F.mse_loss(reg[:, S // 2], yg) + lam * activity)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 200 == 0:
        print(f"  step {step}: loss {float(loss):.3f} "
              f"activity {float(activity)*100:.2f}%", flush=True)
snn.eval()
torch.save(snn.state_dict(),
           "/home/claude/sparq/results/models/snn_sparse.pt")

# ---------------------------------------------------------------- sweep
site_rng = np.random.default_rng(999)
eval_sites = [sample_site(site_rng, "NV") for _ in range(N_EVAL)]
res = {"acc": [], "mae": []}
for T in T_GRID:
    accs, maes = [], []
    for seed in range(N_SEEDS):
        r = np.random.default_rng(1000 + seed)
        ev = make_eval_set(r, N_EVAL, T, sites=eval_sites)
        out = evaluate(snn, ev, is_snn=True)
        accs.append(out["bal_acc"]); maes.append(out["mae_g2"])
    res["acc"].append([float(np.mean(accs)), float(np.std(accs))])
    res["mae"].append([float(np.mean(maes)), float(np.std(maes))])
    print(f"T={T}: acc={res['acc'][-1][0]:.3f}", flush=True)

def time_to_target(accs, target=TARGET_ACC):
    a = np.array([x[0] for x in accs]); logT = np.log10(T_GRID)
    for i in range(len(a) - 1):
        if a[i] < target <= a[i + 1]:
            f = (target - a[i]) / (a[i + 1] - a[i])
            return float(10 ** (logT[i] + f * (logT[i + 1] - logT[i])))
    return float(T_GRID[0]) if a[0] >= target else float("nan")
ttt = time_to_target(res["acc"])
print("time-to-target:", ttt)

# ---------------------------------------------------------------- anytime
r = np.random.default_rng(2024)
ev1 = make_eval_set(r, N_EVAL, 1.0, sites=eval_sites)
with torch.no_grad():
    logits, _ = snn(torch.from_numpy(ev1["stream"]),
                    torch.from_numpy(ev1["aux"]))
    probs = torch.softmax(logits, -1).numpy()
margins = np.abs(probs[..., 1] - probs[..., 0])
vv = ev1["y_valid"]
anytime = {}
for theta in [0.5, 0.6, 0.8, 0.9, 0.95]:
    commit = np.argmax(margins > theta, axis=1).astype(float)
    never = ~(margins > theta).any(1)
    commit[never] = N_SLICES - 1
    slice_ms = 1000.0 / N_SLICES
    pred_at = probs[np.arange(len(commit)), commit.astype(int), 1] > 0.5
    anytime[str(theta)] = dict(
        median_ms=float(np.median((commit[vv] + 1) * slice_ms)),
        mean_ms=float(np.mean((commit[vv] + 1) * slice_ms)),
        acc=balanced_accuracy(ev1["y_cls"][vv], pred_at[vv].astype(int)),
        frac_full=float(never[vv].mean()))
print("anytime:", {k: (v["median_ms"], round(v["acc"], 3))
                   for k, v in anytime.items()})

# ---------------------------------------------------------------- energy
E_SYNOP = 23.6e-12
E_MAC_FP32 = 4.6e-12
E_MAC_INT8 = 1.0e-12
cnn_macs = HistCNN(CFG.n_bins).macs_per_inference(CFG.n_bins)
energy = {}
for T in [0.1, 1.0, 10.0]:
    r = np.random.default_rng(31)
    ev = make_eval_set(r, 400, T, sites=eval_sites[:400])
    syn = snn.synops_per_inference(torch.from_numpy(ev["stream"]),
                                   torch.from_numpy(ev["aux"]))
    energy[str(T)] = dict(
        synops_mean=float(syn.mean()),
        e_snn_nJ=float(syn.mean() * E_SYNOP * 1e9),
        macs_cnn=int(cnn_macs),
        e_cnn_fp32_nJ=float(cnn_macs * E_MAC_FP32 * 1e9),
        e_cnn_int8_nJ=float(cnn_macs * E_MAC_INT8 * 1e9),
        adv_fp32=float(cnn_macs * E_MAC_FP32 / (syn.mean() * E_SYNOP)),
        adv_int8=float(cnn_macs * E_MAC_INT8 / (syn.mean() * E_SYNOP)))
    print(T, energy[str(T)]["e_snn_nJ"], "nJ", flush=True)

out = dict(T_grid=T_GRID, snn_sparse=res, time_to_target=ttt,
           anytime=anytime, energy=energy, lambda_spk=LAMBDA_SPK)
with open("/home/claude/sparq/results/exp2b_snn.json", "w") as f:
    json.dump(out, f)
print(f"saved ({time.time()-t0:.0f}s)")
