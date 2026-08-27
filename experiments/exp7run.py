"""Checkpointed runner for the revision analyses (exp7).

Usage:
  python3 experiments/exp7run.py a --cond noimp --reps 50   # append rows
  python3 experiments/exp7run.py astats                     # covariance stats
  python3 experiments/exp7run.py b1 --group 0|1
  python3 experiments/exp7run.py b2 --group 0|1|2
  python3 experiments/exp7run.py c
  python3 experiments/exp7run.py d
  python3 experiments/exp7run.py merge

Each part writes results/exp7_parts/<part>.json; `merge` assembles
results/exp7_reviewer.json in the schema make_numbers7.py expects.
Physics module is used untouched.
"""
import argparse, json, os, sys, time, dataclasses
import numpy as np

sys.path.insert(0, "/home/claude/sparq")
from sparq.physics import (HBTConfig, EmitterSite, PLATFORMS,
                           DetectorImpairments, correlate,
                           simulate_photon_stream, expected_histogram,
                           sample_site)
from sparq.exact import g2_exact, rates_from_site
from sparq.datasets import make_eval_set, CFG
from sparq.estimators import balanced_accuracy

R = "/home/claude/sparq/results"
PD = f"{R}/exp7_parts"
os.makedirs(PD, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


COV_T = 0.5          # seconds per repeated acquisition (part A)
COV_TARGET = 1000    # target repetitions per condition


def cov_params():
    return dict(tau1=14.0, tau2=250.0, a=0.8, rate_kcps=500, rho=0.8,
                blinking=False, t_on_ms=1, t_off_ms=1, platform="NV")


def part_a(cond, reps):
    """Append `reps` histogram rows for condition cond in {noimp, imp}."""
    path = f"{PD}/A_{cond}.npy"
    H = np.load(path) if os.path.exists(path) else np.zeros((0, 121))
    start = len(H)
    if start >= COV_TARGET:
        log(f"(A) {cond}: already {start} rows, done")
        return
    reps = min(reps, COV_TARGET - start)
    imp = DetectorImpairments() if cond == "imp" else None
    site = EmitterSite(cov_params(), 2)
    cfg = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.35)
    rng = np.random.default_rng(31000 + 7 * start + (1 if cond == "imp" else 0))
    rows = []
    for i in range(reps):
        ta, tb = simulate_photon_stream(site, COV_T, rng, imp=imp)
        if imp is None:  # apply IRF jitter by hand, as in exp1(b)
            ta = np.sort(ta + rng.normal(0, cfg.sigma_irf, len(ta)))
            tb = np.sort(tb + rng.normal(0, cfg.sigma_irf, len(tb)))
        rows.append(correlate(ta, tb, cfg))
    H = np.vstack([H, np.array(rows, float)])
    np.save(path, H)
    log(f"(A) {cond}: {start} -> {len(H)} rows")


def part_astats():
    res = {}
    for cond, key in (("noimp", "no_impairments"), ("imp", "with_impairments")):
        H = np.load(f"{PD}/A_{cond}.npy")
        n_rep = len(H)
        C = np.corrcoef(H.T)
        off = C[~np.eye(C.shape[0], dtype=bool)]
        floor = 1.0 / np.sqrt(n_rep)
        res[key] = dict(
            n_rep=n_rep,
            acq_T_s=COV_T,
            mean_abs_offdiag=float(np.mean(np.abs(off))),
            p95_abs_offdiag=float(np.percentile(np.abs(off), 95)),
            max_abs_offdiag=float(np.max(np.abs(off))),
            adjacent_mean=float(np.mean(np.diag(C, 1))),
            adjacent_max=float(np.max(np.abs(np.diag(C, 1)))),
            noise_floor=floor,
            frac_above_2sigma=float(np.mean(np.abs(off) > 2 * floor)),
            fano=float(np.mean(H.var(0) / np.maximum(H.mean(0), 1e-9))),
            mean_bin_count=float(H.mean()))
        log(f"(A) {key}: n={n_rep} mean|r|={res[key]['mean_abs_offdiag']:.4f} "
            f"floor={floor:.4f} frac>2sig={res[key]['frac_above_2sigma']:.4f} "
            f"Fano={res[key]['fano']:.3f}")
    json.dump(res, open(f"{PD}/covariance.json", "w"), indent=1)


# ---------------------------------------------------------------- B1
cfgv = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.0)


def g2_exact_binavg(cfg, k, oversample=9):
    off = (np.arange(oversample) + 0.5) / oversample - 0.5
    fine = (cfg.bin_centers[:, None] + off[None, :] * cfg.bin_width).ravel()
    g = g2_exact(fine, *k).reshape(cfg.n_bins, oversample)
    return g.mean(1)


def validate_config(tau1, tau2, a, label, rng):
    p = dict(tau1=tau1, tau2=tau2, a=a, rate_kcps=1500, rho=1.0,
             blinking=False, t_on_ms=1, t_off_ms=1, platform="NV")
    site = EmitterSite(p, 1)
    # Intrinsic-rate simulation memory scales as T/tau1; cap the number of
    # emission cycles (~1.2e8) so short-tau1 edge configs fit in RAM. The
    # detected rate is high, so even the shortest T keeps per-bin Poisson
    # noise at the percent level.
    T = min(1.2, 0.3 * tau1)
    ta, tb = simulate_photon_stream(site, T, rng, imp=None)
    hist = correlate(ta, tb, cfgv)
    flat = (len(ta) / T) * (len(tb) / T) * (cfgv.bin_width * 1e-9) * T
    g2_mc = hist / flat
    k = rates_from_site(tau1, tau2, a)
    g2_ref = g2_exact_binavg(cfgv, k)
    sd = np.sqrt(np.maximum(hist, 1)) / flat
    chi2 = float(np.mean(((g2_mc - g2_ref) / sd) ** 2))
    nrmse = float(np.sqrt(np.mean((g2_mc - g2_ref) ** 2)))
    log(f"(B1) {label}: tau1={tau1:.2f} tau2={tau2:.0f} a={a:.2f} "
        f"chi2={chi2:.2f} nrmse={nrmse:.4f}")
    return dict(label=label, tau1=tau1, tau2=tau2, a=a, chi2_red=chi2,
                nrmse=nrmse, n_coinc=int(hist.sum()))


EDGES = [
    (0.6, 10.0, 0.1, "edge-shortest-tau1-tau2"),
    (0.6, 2500.0, 3.0, "edge-shortest-tau1-longest-tau2-max-a"),
    (25.0, 100.0, 1.5, "edge-longest-tau1-NV-max-a"),
    (25.0, 500.0, 0.1, "edge-longest-tau1-tau2-min-a"),
    (1.5, 2500.0, 3.0, "edge-hBN-extreme-bunching"),
    (2.0, 10.0, 1.2, "edge-GaN-fast-shelving"),
]


def part_b1(group):
    rng = np.random.default_rng(2026 + group)
    plats = ("NV", "hBN") if group == 0 else ("GaN", "SiV")
    res = []
    for plat in plats:
        P = PLATFORMS[plat]
        for i in range(5):
            pr = P.sample(rng)
            res.append(validate_config(pr["tau1"], pr["tau2"], pr["a"],
                                       f"random-{plat}-{i}", rng))
    if group == 1:
        for tau1, tau2, a, lab in EDGES:
            res.append(validate_config(tau1, tau2, a, lab, rng))
    json.dump(res, open(f"{PD}/B1_g{group}.json", "w"), indent=1)


# ---------------------------------------------------------------- B2
CONDS = [
    dict(lbl="N2-rho0.8-IRF", n=2, rho=0.80, rate=500, blink=False),
    dict(lbl="N1-rho0.95-IRF", n=1, rho=0.95, rate=200, blink=False),
    dict(lbl="N3-rho0.6-IRF", n=3, rho=0.60, rate=800, blink=False),
    dict(lbl="N1-rho0.7-lowrate", n=1, rho=0.70, rate=80, blink=False),
    dict(lbl="N2-rho0.9-highrate", n=2, rho=0.90, rate=1000, blink=False),
    dict(lbl="N1-rho0.85-blinking", n=1, rho=0.85, rate=400, blink=True),
]


def part_b2(group):
    rng = np.random.default_rng(4000 + group)
    res = []
    for c in CONDS[2 * group:2 * group + 2]:
        p = dict(tau1=14.0, tau2=250.0, a=0.8, rate_kcps=c["rate"],
                 rho=c["rho"], blinking=c["blink"], t_on_ms=20.0,
                 t_off_ms=8.0, platform="NV")
        site = EmitterSite(p, c["n"])
        cfg_irf = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.35)
        H = []
        for i in range(60):
            ta, tb = simulate_photon_stream(site, 1.0, rng, imp=None)
            ta = np.sort(ta + rng.normal(0, cfg_irf.sigma_irf, len(ta)))
            tb = np.sort(tb + rng.normal(0, cfg_irf.sigma_irf, len(tb)))
            H.append(correlate(ta, tb, cfg_irf))
        H = np.array(H, float)
        mu_twin = expected_histogram(site, 1.0, cfg_irf)
        mu_mc, var_mc = H.mean(0), H.var(0)
        rel = float(np.mean(np.abs(mu_mc - mu_twin) / np.maximum(mu_twin, 1)))
        fano = float(np.mean(var_mc / np.maximum(mu_mc, 1e-9)))
        res.append(dict(**c, rel_mean_err_pct=100 * rel, fano=fano))
        log(f"(B2) {c['lbl']}: rel err {100*rel:.2f}%, Fano {fano:.3f}")
    json.dump(res, open(f"{PD}/B2_g{group}.json", "w"), indent=1)


# ---------------------------------------------------------------- C
def part_c():
    from scipy.special import gammaln
    site_rng = np.random.default_rng(999)
    eval_sites = [sample_site(site_rng, "NV") for _ in range(1200)]

    def bayes_acc(T, ref_n, seed=77):
        rr = np.random.default_rng(seed)
        ref_sites = [sample_site(rr, "NV") for _ in range(ref_n)]
        ref_good = np.array([s.g2_0 < 0.5 for s in ref_sites])
        rates = np.array([
            s.params["rate_kcps"] * 1e3 *
            (s.params["t_on_ms"] / (s.params["t_on_ms"] + s.params["t_off_ms"])
             if s.params["blinking"] else 1.0) for s in ref_sites])
        r = np.random.default_rng(1000)
        ev = make_eval_set(r, 1200, T, sites=eval_sites)
        mu = np.stack([expected_histogram(s, T, CFG) for s in ref_sites])
        logmu = np.log(np.maximum(mu, 1e-12))
        lam = rates * T
        H = ev["hist"]
        n_obs = (10 ** ev["aux"][:, 1]) * T
        p_good = np.empty(len(H))
        for i0 in range(0, len(H), 300):
            sl = slice(i0, i0 + 300)
            ll = H[sl] @ logmu.T - mu.sum(1)[None, :]
            ll += (n_obs[sl, None] * np.log(lam)[None, :] - lam[None, :]
                   - gammaln(n_obs[sl] + 1)[:, None])
            ll -= ll.max(1, keepdims=True)
            w = np.exp(ll)
            p_good[sl] = (w * ref_good[None, :]).sum(1) / w.sum(1)
        v = ev["y_valid"]
        return balanced_accuracy(ev["y_cls"][v],
                                 (p_good[v] > 0.5).astype(int))

    conv = {}
    for T in (0.1, 1.0, 30.0):
        conv[str(T)] = {}
        for M in (1500, 3000, 6000, 12000, 24000):
            a = bayes_acc(T, M)
            conv[str(T)][str(M)] = a
            log(f"(C) T={T}s M={M}: bal acc {a:.4f}")
    json.dump(conv, open(f"{PD}/C.json", "w"), indent=1)


# ---------------------------------------------------------------- D
def part_d():
    import torch
    from sparq.estimators import SpikingG2Net, HistCNN, evaluate
    snn = SpikingG2Net(CFG.n_bins)
    snn.load_state_dict(torch.load(f"{R}/models/snn_sparse.pt",
                                   map_location="cpu"))
    snn.eval()
    cnn = HistCNN(CFG.n_bins)
    cnn.load_state_dict(torch.load(f"{R}/models/cnn_pitl.pt",
                                   map_location="cpu"))
    cnn.eval()
    NV = PLATFORMS["NV"]
    shifts = {
        "in-prior": NV,
        "tau1 +30%": dataclasses.replace(NV, tau1_rng=(8 * 1.3, 25 * 1.3)),
        "tau1 -30%": dataclasses.replace(NV, tau1_rng=(8 * 0.7, 25 * 0.7)),
        "rate x0.6": dataclasses.replace(NV, rate_rng=(30 * 0.6, 350 * 0.6)),
        "tau2 +50%": dataclasses.replace(NV, tau2_rng=(150, 750)),
        "blink 2x": dataclasses.replace(NV, blink_p=0.30),
    }

    def sites_from(plat, n_sites, seed):
        r = np.random.default_rng(seed)
        return [EmitterSite(plat.sample(r),
                            int(r.choice([1, 2, 3, 4],
                                         p=(0.42, 0.30, 0.18, 0.10))))
                for _ in range(n_sites)]

    rob = {}
    with torch.no_grad():
        for lbl, plat in shifts.items():
            rob[lbl] = {}
            sites = sites_from(plat, 1200, 4321)
            for T in (0.1, 1.0):
                r = np.random.default_rng(777)
                ev = make_eval_set(r, 1200, T, sites=sites)
                rs = evaluate(snn, ev, is_snn=True)
                rc = evaluate(cnn, ev)
                rob[lbl][str(T)] = dict(snn_acc=rs["bal_acc"],
                                        snn_mae=rs["mae_g2"],
                                        cnn_acc=rc["bal_acc"],
                                        cnn_mae=rc["mae_g2"])
                log(f"(D) {lbl} T={T}: SNN {rs['bal_acc']:.4f} "
                    f"CNN {rc['bal_acc']:.4f}")
    json.dump(rob, open(f"{PD}/D.json", "w"), indent=1)


# ---------------------------------------------------------------- merge
def merge():
    out = {}
    out["covariance"] = json.load(open(f"{PD}/covariance.json"))
    b1 = (json.load(open(f"{PD}/B1_g0.json"))
          + json.load(open(f"{PD}/B1_g1.json")))
    chi2s = [v["chi2_red"] for v in b1]
    nrmses = [v["nrmse"] for v in b1]
    out["stream_vs_exact_sweep"] = dict(
        configs=b1, n=len(b1),
        chi2_mean=float(np.mean(chi2s)), chi2_max=float(np.max(chi2s)),
        chi2_min=float(np.min(chi2s)),
        nrmse_mean=float(np.mean(nrmses)), nrmse_max=float(np.max(nrmses)))
    out["twin_vs_stream_sweep"] = sum(
        (json.load(open(f"{PD}/B2_g{g}.json")) for g in range(3)), [])
    out["bayes_convergence"] = json.load(open(f"{PD}/C.json"))
    out["prior_shift"] = json.load(open(f"{PD}/D.json"))
    json.dump(out, open(f"{R}/exp7_reviewer.json", "w"), indent=1)
    log("wrote results/exp7_reviewer.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--cond", default="noimp")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--group", type=int, default=0)
    args = ap.parse_args()
    {"a": lambda: part_a(args.cond, args.reps),
     "astats": part_astats,
     "b1": lambda: part_b1(args.group),
     "b2": lambda: part_b2(args.group),
     "c": part_c,
     "d": part_d,
     "merge": merge}[args.part]()
