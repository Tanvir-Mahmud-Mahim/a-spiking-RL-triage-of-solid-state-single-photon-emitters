"""Experiment 8 -- analyses requested in the second review round.

Checkpointed runner (each part writes results/exp8_parts/<part>.json or
.npy; `merge` assembles results/exp8_r2.json).

  a2    --cond {B_noimp,B_imp,C_blink} --reps 50
        repeated stream-simulator acquisitions for the bin-covariance
        analysis at additional operating conditions (condition A, the
        R1 analysis, is results/exp7_parts/A_{noimp,imp}.npy)
  a2stats
        off-diagonal correlation statistics for all conditions
  c2    --seed S
        Monte-Carlo Bayes reference versus reference-sample size M for
        one reference-sample seed (M up to 48000)
  d2    --prior LABEL
        balanced accuracy versus acquisition time for the trained SNN
        and CNN (5 noise seeds) and the LM fit (2 noise seeds, 400
        sites) on a deliberately shifted NV prior, and the resulting
        time to the 90% target
  merge
"""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, dataclasses, json, os, sys, time
import numpy as np

sys.path.insert(0, _ROOT)
from sparq.physics import (HBTConfig, EmitterSite, PLATFORMS,
                           DetectorImpairments, correlate,
                           simulate_photon_stream, expected_histogram,
                           sample_site)
from sparq.datasets import make_eval_set, CFG
from sparq.estimators import balanced_accuracy, fit_g2_histogram

R = _ROOT + "/results"
PD = f"{R}/exp8_parts"
os.makedirs(PD, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


# ================================================================ A2
COV_T = 0.5
COV_TARGET = 1000
CONDS = {
    # isolated bright single emitter with little background
    "B_noimp": dict(n=1, rho=0.95, rate=200, blink=False, imp=False),
    "B_imp": dict(n=1, rho=0.95, rate=200, blink=False, imp=True),
    # blinking single emitter (non-stationary source), ideal detectors
    "C_blink": dict(n=1, rho=0.85, rate=400, blink=True, imp=False),
}


def part_a2(cond, reps):
    c = CONDS[cond]
    path = f"{PD}/A2_{cond}.npy"
    H = np.load(path) if os.path.exists(path) else np.zeros((0, 121))
    start = len(H)
    if start >= COV_TARGET:
        log(f"(A2) {cond}: complete ({start})")
        return
    reps = min(reps, COV_TARGET - start)
    p = dict(tau1=14.0, tau2=250.0, a=0.8, rate_kcps=c["rate"],
             rho=c["rho"], blinking=c["blink"], t_on_ms=20.0,
             t_off_ms=8.0, platform="NV")
    site = EmitterSite(p, c["n"])
    imp = DetectorImpairments() if c["imp"] else None
    cfg = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.35)
    seed = 52000 + 11 * start + sorted(CONDS).index(cond)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        ta, tb = simulate_photon_stream(site, COV_T, rng, imp=imp)
        if imp is None:
            ta = np.sort(ta + rng.normal(0, cfg.sigma_irf, len(ta)))
            tb = np.sort(tb + rng.normal(0, cfg.sigma_irf, len(tb)))
        rows.append(correlate(ta, tb, cfg))
    H = np.vstack([H, np.array(rows, float)])
    np.save(path, H)
    log(f"(A2) {cond}: {start} -> {len(H)}")


def cov_stats(H):
    n_rep = len(H)
    C = np.corrcoef(H.T)
    off = C[~np.eye(C.shape[0], dtype=bool)]
    floor = 1.0 / np.sqrt(n_rep)
    return dict(
        n_rep=n_rep, acq_T_s=COV_T,
        mean_bin_count=float(H.mean()),
        mean_offdiag=float(np.mean(off)),
        mean_abs_offdiag=float(np.mean(np.abs(off))),
        p95_abs_offdiag=float(np.percentile(np.abs(off), 95)),
        max_abs_offdiag=float(np.max(np.abs(off))),
        adjacent_mean=float(np.mean(np.diag(C, 1))),
        noise_floor=floor,
        expected_mean_abs=float(np.sqrt(2 / np.pi) * floor),
        frac_above_2sigma=float(np.mean(np.abs(off) > 2 * floor)),
        fano=float(np.mean(H.var(0) / np.maximum(H.mean(0), 1e-9))),
        # acquisition-to-acquisition spread of the total coincidence count
        # versus the Poisson expectation, and the correlations that remain
        # after each histogram is rescaled to the mean total (removes a
        # common multiplicative factor; the sum constraint alone gives a
        # mean of -1/(n_bins - 1))
        cv_total=float(H.sum(1).std() / H.sum(1).mean()),
        cv_total_poisson=float(1 / np.sqrt(H.sum(1).mean())),
        **_norm_stats(H, floor))


def _norm_stats(H, floor):
    s = H.sum(1)
    Hn = H / s[:, None] * s.mean()
    C = np.corrcoef(Hn.T)
    off = C[~np.eye(C.shape[0], dtype=bool)]
    return dict(norm_mean_offdiag=float(np.mean(off)),
                norm_mean_abs_offdiag=float(np.mean(np.abs(off))),
                norm_frac_above_2sigma=float(np.mean(np.abs(off) > 2 * floor)))


def part_a2stats():
    res = {}
    src = {"A_noimp": f"{R}/exp7_parts/A_noimp.npy",
           "A_imp": f"{R}/exp7_parts/A_imp.npy"}
    for k in CONDS:
        src[k] = f"{PD}/A2_{k}.npy"
    for k, path in src.items():
        H = np.load(path)
        res[k] = cov_stats(H)
        s = res[k]
        log(f"(A2) {k}: n={s['n_rep']} mean r={s['mean_offdiag']:+.4f} "
            f"mean|r|={s['mean_abs_offdiag']:.4f} "
            f"(exp {s['expected_mean_abs']:.4f}) "
            f">2sig={100*s['frac_above_2sigma']:.1f}% "
            f"Fano={s['fano']:.3f} counts/bin={s['mean_bin_count']:.1f}")
    json.dump(res, open(f"{PD}/A2_stats.json", "w"), indent=1)


# ================================================================ C2
MS = (1500, 3000, 6000, 12000, 24000, 48000)
TS_C = (0.1, 1.0, 30.0)


def part_c2(seed, ts=TS_C, tag=""):
    from scipy.special import gammaln
    site_rng = np.random.default_rng(999)       # main-text population
    eval_sites = [sample_site(site_rng, "NV") for _ in range(1200)]
    evs = {}
    for T in ts:
        r = np.random.default_rng(1000)          # main-text noise seed
        evs[T] = make_eval_set(r, 1200, T, sites=eval_sites)
    rr = np.random.default_rng(seed)
    ref_all = [sample_site(rr, "NV") for _ in range(max(MS))]
    out = {}
    for T in ts:
        ev = evs[T]
        H = ev["hist"]
        n_obs = (10 ** ev["aux"][:, 1]) * T
        v = ev["y_valid"]
        out[str(T)] = {}
        for M in MS:
            ref = ref_all[:M]     # nested reference samples
            good = np.array([s.g2_0 < 0.5 for s in ref])
            rates = np.array([
                s.params["rate_kcps"] * 1e3 *
                (s.params["t_on_ms"] / (s.params["t_on_ms"]
                                        + s.params["t_off_ms"])
                 if s.params["blinking"] else 1.0) for s in ref])
            mu = np.stack([expected_histogram(s, T, CFG) for s in ref])
            logmu = np.log(np.maximum(mu, 1e-12))
            musum = mu.sum(1)
            lam = rates * T
            p_good = np.empty(len(H))
            for i0 in range(0, len(H), 200):
                sl = slice(i0, i0 + 200)
                ll = H[sl] @ logmu.T - musum[None, :]
                ll += (n_obs[sl, None] * np.log(lam)[None, :]
                       - lam[None, :] - gammaln(n_obs[sl] + 1)[:, None])
                ll -= ll.max(1, keepdims=True)
                w = np.exp(ll)
                p_good[sl] = (w * good[None, :]).sum(1) / w.sum(1)
            acc = balanced_accuracy(ev["y_cls"][v],
                                    (p_good[v] > 0.5).astype(int))
            out[str(T)][str(M)] = acc
            log(f"(C2) seed={seed} T={T} M={M}: {acc:.4f}")
            del mu, logmu
    json.dump(out, open(f"{PD}/C2{tag}_seed{seed}.json", "w"), indent=1)


# ================================================================ D2
T_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TARGET = 0.90
NV = PLATFORMS["NV"]
SHIFTS = {
    "in-prior": NV,
    "tau1 +30%": dataclasses.replace(NV, tau1_rng=(8 * 1.3, 25 * 1.3)),
    "tau1 -30%": dataclasses.replace(NV, tau1_rng=(8 * 0.7, 25 * 0.7)),
    "rate x0.6": dataclasses.replace(NV, rate_rng=(30 * 0.6, 350 * 0.6)),
    "tau2 +50%": dataclasses.replace(NV, tau2_rng=(150, 750)),
    "blink 2x": dataclasses.replace(NV, blink_p=0.30),
}
SLUG = {k: k.replace(" ", "_").replace("%", "pct").replace("+", "p")
        .replace("-", "m") for k in SHIFTS}


def sites_from(plat, n_sites, seed):
    r = np.random.default_rng(seed)
    return [EmitterSite(plat.sample(r),
                        int(r.choice([1, 2, 3, 4],
                                     p=(0.42, 0.30, 0.18, 0.10))))
            for _ in range(n_sites)]


def time_to_target(acc_means, target=TARGET):
    a = np.array(acc_means)
    logT = np.log10(T_GRID)
    if a[0] >= target:
        return float(T_GRID[0])
    for i in range(len(a) - 1):
        if a[i] < target <= a[i + 1]:
            f = (target - a[i]) / (a[i + 1] - a[i])
            return float(10 ** (logT[i] + f * (logT[i + 1] - logT[i])))
    return None


def part_d2(label):
    import torch
    from sparq.estimators import SpikingG2Net, HistCNN, evaluate
    torch.set_num_threads(1)
    snn = SpikingG2Net(CFG.n_bins)
    snn.load_state_dict(torch.load(f"{R}/models/snn_sparse.pt",
                                   map_location="cpu"))
    snn.eval()
    cnn = HistCNN(CFG.n_bins)
    cnn.load_state_dict(torch.load(f"{R}/models/cnn_pitl.pt",
                                   map_location="cpu"))
    cnn.eval()
    sites = sites_from(SHIFTS[label], 1200, 4321)
    res = {m: {"acc": [], "mae": []} for m in ("snn", "cnn", "fit")}
    for T in T_GRID:
        acc = {m: [] for m in res}
        mae = {m: [] for m in res}
        for seed in range(5):
            r = np.random.default_rng(1000 + seed)
            ev = make_eval_set(r, 1200, T, sites=sites)
            with torch.no_grad():
                o_s = evaluate(snn, ev, is_snn=True)
                o_c = evaluate(cnn, ev)
            acc["snn"].append(o_s["bal_acc"]); mae["snn"].append(o_s["mae_g2"])
            acc["cnn"].append(o_c["bal_acc"]); mae["cnn"].append(o_c["mae_g2"])
            if seed < 2:
                nf = 400
                g2h = np.array([fit_g2_histogram(
                    ev["hist"][i], T, 10 ** ev["aux"][i, 1], CFG)[0]
                    for i in range(nf)])
                vf = ev["y_valid"][:nf]
                acc["fit"].append(balanced_accuracy(
                    ev["y_cls"][:nf][vf], (g2h[vf] < 0.5).astype(int)))
                mae["fit"].append(float(np.mean(np.abs(
                    g2h - ev["y_g2"][:nf]))))
        for m in res:
            res[m]["acc"].append([float(np.mean(acc[m])),
                                  float(np.std(acc[m]))])
            res[m]["mae"].append([float(np.mean(mae[m])),
                                  float(np.std(mae[m]))])
        log(f"(D2) {label} T={T}: " + "  ".join(
            f"{m} {res[m]['acc'][-1][0]:.3f}" for m in res))
    ttt = {m: time_to_target([x[0] for x in res[m]["acc"]]) for m in res}
    res["ttt"] = ttt
    res["speedup_snn"] = (ttt["fit"] / ttt["snn"]
                          if ttt["fit"] and ttt["snn"] else None)
    res["speedup_cnn"] = (ttt["fit"] / ttt["cnn"]
                          if ttt["fit"] and ttt["cnn"] else None)
    log(f"(D2) {label}: ttt {ttt}, speedup SNN {res['speedup_snn']}")
    json.dump(res, open(f"{PD}/D2_{SLUG[label]}.json", "w"), indent=1)


# ================================================================ merge
def merge():
    out = {"covariance": json.load(open(f"{PD}/A2_stats.json"))}
    conv = {}
    for fn in sorted(os.listdir(PD)):
        if fn.startswith("C2_seed"):
            seed = fn[7:-5]
            conv.setdefault(seed, {}).update(json.load(open(f"{PD}/{fn}")))
        if fn.startswith("C2x_seed"):
            seed = fn[8:-5]
            conv.setdefault(seed, {}).update(json.load(open(f"{PD}/{fn}")))
    out["bayes_convergence_seeds"] = conv
    out["prior_shift_speedup"] = {
        k: json.load(open(f"{PD}/D2_{SLUG[k]}.json")) for k in SHIFTS
        if os.path.exists(f"{PD}/D2_{SLUG[k]}.json")}
    out["T_grid"] = T_GRID
    out["target"] = TARGET
    json.dump(out, open(f"{R}/exp8_r2.json", "w"), indent=1)
    log("wrote results/exp8_r2.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--cond", default="B_noimp")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--prior", default="in-prior")
    a = ap.parse_args()
    {"a2": lambda: part_a2(a.cond, a.reps),
     "a2stats": part_a2stats,
     "c2": lambda: part_c2(a.seed),
     "c2x": lambda: part_c2(a.seed, ts=(0.03, 0.3), tag="x"),
     "d2": lambda: part_d2(a.prior),
     "merge": merge}[a.part]()
