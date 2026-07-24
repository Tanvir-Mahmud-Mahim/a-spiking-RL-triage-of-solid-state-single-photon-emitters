"""Experiment 2 — estimator envelope: balanced accuracy and g2(0) error vs
acquisition time for (i) the conventional LM-fit pipeline, (ii) a
Kudyshev-style CNN on integrated histograms, (iii) the event-driven SNN,
each trained physics-in-the-loop; plus a 'clean-trained' CNN ablation
(trained only on high-statistics acquisitions) and the SNN anytime readout.
Also: measured event-driven energy accounting.

Writes results/exp2_estimators.json and models to results/models/.
"""
import json, sys, time, os
import numpy as np
import torch
from scipy.special import gammaln

sys.path.insert(0, "/home/claude/sparq")
from sparq.datasets import make_batch, make_eval_set, CFG, N_SLICES
from sparq.estimators import (HistCNN, SpikingG2Net, train_model, evaluate,
                              fit_g2_histogram, balanced_accuracy)
from sparq.physics import sample_site, expected_histogram


def bayes_reference(eval_batches, ref_n=6000, seed=77):
    """Monte-Carlo posterior reference (approximate Bayes envelope):
    exact per-site likelihoods against a large prior sample."""
    rr = np.random.default_rng(seed)
    ref_sites = [sample_site(rr, "NV") for _ in range(ref_n)]
    ref_good = np.array([s.g2_0 < 0.5 for s in ref_sites])
    rates = np.array([
        s.params["rate_kcps"] * 1e3 *
        (s.params["t_on_ms"] / (s.params["t_on_ms"] + s.params["t_off_ms"])
         if s.params["blinking"] else 1.0) for s in ref_sites])
    out = {}
    for T, ev in eval_batches.items():
        mu = np.stack([expected_histogram(s, T, CFG) for s in ref_sites])
        logmu = np.log(np.maximum(mu, 1e-12))
        H = ev["hist"]
        n_obs = (10 ** ev["aux"][:, 1]) * T
        ll = H @ logmu.T - mu.sum(1)[None, :]
        lam = rates * T
        ll += (n_obs[:, None] * np.log(lam)[None, :] - lam[None, :]
               - gammaln(n_obs + 1)[:, None])
        ll -= ll.max(1, keepdims=True)
        w = np.exp(ll)
        p_good = (w * ref_good[None, :]).sum(1) / w.sum(1)
        v = ev["y_valid"]
        out[T] = balanced_accuracy(ev["y_cls"][v],
                                   (p_good[v] > 0.5).astype(int))
    return out

os.makedirs("/home/claude/sparq/results/models", exist_ok=True)
rng = np.random.default_rng(11)
t0 = time.time()

STEPS_CNN = 1500
STEPS_SNN = 1500
T_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
N_EVAL = 1200
N_EVAL_FIT = 400          # LM fits are costly; evaluate on a subset
N_SEEDS = 5
N_SEEDS_FIT = 2
TARGET_ACC = 0.90         # matched-accuracy target = conventional asymptote

# ------------------------------------------------------------- training
def gen_pitl(rng_, batch):
    return make_batch(rng_, batch, T_dist=("logu", 0.03, 30.0), platform="NV")

def gen_clean(rng_, batch):
    return make_batch(rng_, batch, platform="NV", T_fixed=30.0)

print("== training CNN (physics-in-the-loop)")
cnn = HistCNN(CFG.n_bins)
train_model(cnn, gen_pitl, STEPS_CNN, seed=1)
print("== training CNN (clean/asymptotic-statistics only)")
cnn_clean = HistCNN(CFG.n_bins)
train_model(cnn_clean, gen_clean, STEPS_CNN, seed=2)
print("== training SNN (physics-in-the-loop, event-driven)")
snn = SpikingG2Net(CFG.n_bins)
train_model(snn, gen_pitl, STEPS_SNN, is_snn=True, seed=3)

torch.save(cnn.state_dict(), "/home/claude/sparq/results/models/cnn_pitl.pt")
torch.save(cnn_clean.state_dict(),
           "/home/claude/sparq/results/models/cnn_clean.pt")
torch.save(snn.state_dict(), "/home/claude/sparq/results/models/snn_pitl.pt")
print(f"training done ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------- evaluation
# fixed site population, fresh noise per seed
site_rng = np.random.default_rng(999)
eval_sites = [sample_site(site_rng, "NV") for _ in range(N_EVAL)]
print("eval P(good) =", np.mean([s.g2_0 < 0.5 for s in eval_sites]))

results = {m: {"acc": [], "mae": []} for m in
           ["fit", "cnn_clean", "cnn_pitl", "snn_pitl"]}
# Monte-Carlo Bayes reference (one noise seed per T)
bayes_evs = {}
for T in T_GRID:
    r = np.random.default_rng(1000)
    bayes_evs[T] = make_eval_set(r, N_EVAL, T, sites=eval_sites)
bayes_acc = bayes_reference(bayes_evs)
print("Bayes reference:", {t: round(a, 3) for t, a in bayes_acc.items()})
del bayes_evs
for T in T_GRID:
    accs = {m: [] for m in results}
    maes = {m: [] for m in results}
    for seed in range(N_SEEDS):
        r = np.random.default_rng(1000 + seed)
        ev = make_eval_set(r, N_EVAL, T, sites=eval_sites)
        y = ev["y_cls"]
        # LM fit baseline (subset + fewer seeds; it is by far the slowest)
        if seed < N_SEEDS_FIT:
            g2h = np.array([fit_g2_histogram(ev["hist"][i], T,
                                             10 ** ev["aux"][i, 1], CFG)[0]
                            for i in range(N_EVAL_FIT)])
            pred = (g2h < 0.5).astype(int)
            vf = ev["y_valid"][:N_EVAL_FIT]
            accs["fit"].append(
                balanced_accuracy(y[:N_EVAL_FIT][vf], pred[vf]))
            maes["fit"].append(
                float(np.mean(np.abs(g2h - ev["y_g2"][:N_EVAL_FIT]))))
        for name, model, is_snn in [("cnn_clean", cnn_clean, False),
                                    ("cnn_pitl", cnn, False),
                                    ("snn_pitl", snn, True)]:
            out = evaluate(model, ev, is_snn=is_snn)
            accs[name].append(out["bal_acc"])
            maes[name].append(out["mae_g2"])
    for m in results:
        results[m]["acc"].append([float(np.mean(accs[m])),
                                  float(np.std(accs[m]))])
        results[m]["mae"].append([float(np.mean(maes[m])),
                                  float(np.std(maes[m]))])
    print(f"T={T:6.2f}s  " + "  ".join(
        f"{m}:{results[m]['acc'][-1][0]:.3f}" for m in results))

# ------------------------------------------------------------- time-to-target
def time_to_target(accs, target=TARGET_ACC):
    a = np.array([x[0] for x in accs])
    logT = np.log10(T_GRID)
    for i in range(len(a) - 1):
        if a[i] < target <= a[i + 1]:
            f = (target - a[i]) / (a[i + 1] - a[i])
            return float(10 ** (logT[i] + f * (logT[i + 1] - logT[i])))
    if a[0] >= target:
        return float(T_GRID[0])
    return float("nan")

ttt = {m: time_to_target(results[m]["acc"]) for m in results}
print("time-to-95%:", ttt)

# ------------------------------------------------------------- anytime SNN
# distribution of commitment times at T = 1 s with a confidence gate
r = np.random.default_rng(2024)
ev1 = make_eval_set(r, N_EVAL, 1.0, sites=eval_sites)
with torch.no_grad():
    logits, _ = snn(torch.from_numpy(ev1["stream"]),
                    torch.from_numpy(ev1["aux"]))
    probs = torch.softmax(logits, -1).numpy()      # [B, S, 2]
margins = np.abs(probs[..., 1] - probs[..., 0])
vv = ev1["y_valid"]
anytime = {}
for theta in [0.6, 0.8, 0.9, 0.95]:
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

# ------------------------------------------------------------- energy
E_SYNOP = 23.6e-12       # J per synaptic op, Intel Loihi (Davies 2018)
E_MAC_FP32 = 4.6e-12     # J per 32-bit MAC, 45 nm CMOS (Horowitz 2014)
E_MAC_INT8 = 1.0e-12     # J per 8-bit MAC, edge accelerator class
energy = {}
for T in [0.1, 1.0, 10.0]:
    r = np.random.default_rng(31)
    ev = make_eval_set(r, 400, T, sites=eval_sites[:400])
    syn = snn.synops_per_inference(torch.from_numpy(ev["stream"]),
                                   torch.from_numpy(ev["aux"]))
    macs = cnn.macs_per_inference(CFG.n_bins)
    energy[str(T)] = dict(
        synops_mean=float(syn.mean()),
        e_snn_nJ=float(syn.mean() * E_SYNOP * 1e9),
        macs_cnn=int(macs),
        e_cnn_fp32_nJ=float(macs * E_MAC_FP32 * 1e9),
        e_cnn_int8_nJ=float(macs * E_MAC_INT8 * 1e9),
        adv_fp32=float(macs * E_MAC_FP32 / (syn.mean() * E_SYNOP)),
        adv_int8=float(macs * E_MAC_INT8 / (syn.mean() * E_SYNOP)),
    )
print("energy:", energy)

out = dict(T_grid=T_GRID, results=results, time_to_target=ttt,
           bayes_acc={str(k): v for k, v in bayes_acc.items()},
           target_acc=TARGET_ACC, anytime=anytime, energy=energy,
           n_eval=N_EVAL, n_seeds=N_SEEDS,
           p_good=float(np.mean([s.g2_0 < 0.5 for s in eval_sites])),
           speedup_fit_over_snn=float(ttt["fit"] / ttt["snn_pitl"])
           if np.isfinite(ttt["fit"]) and np.isfinite(ttt["snn_pitl"]) else None)
with open("/home/claude/sparq/results/exp2_estimators.json", "w") as f:
    json.dump(out, f)
print(f"saved. total {time.time()-t0:.0f}s")
