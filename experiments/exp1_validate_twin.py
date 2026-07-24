"""Experiment 1 — validate the twin end to end.

(a) Full Monte-Carlo photon streams vs the numerically exact master-equation
    g2(tau) for three emitter regimes (NV-like, hBN-like, GaN-like).
(b) Histogram-twin Poisson statistics vs the full MC pipeline (mean and
    variance per bin) including background and multi-emitter superposition.

Writes results/exp1_validation.json and arrays for the validation figure.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/sparq")
from sparq.physics import (HBTConfig, EmitterSite, correlate,
                           simulate_photon_stream, expected_histogram,
                           g2_measured)
from sparq.exact import g2_exact, effective_params, rates_from_site

rng = np.random.default_rng(7)
out = {}

# ----------------------------------------------------------------- (a)
regimes = {
    # count rates deliberately boosted (statistics validation; detector
    # impairments are off here) so per-bin Poisson noise is at the % level
    "NV-like":  dict(tau1=14.0, tau2=250.0, a=0.8,  rate_kcps=1000, rho=1.0,
                     blinking=False, t_on_ms=1, t_off_ms=1, platform="NV"),
    "hBN-like": dict(tau1=3.0,  tau2=120.0, a=1.6,  rate_kcps=1600, rho=1.0,
                     blinking=False, t_on_ms=1, t_off_ms=1, platform="hBN"),
    "GaN-like": dict(tau1=1.1,  tau2=60.0,  a=0.5,  rate_kcps=3000, rho=1.0,
                     blinking=False, t_on_ms=1, t_off_ms=1, platform="GaN"),
}
cfg = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.0)


def g2_exact_binavg(cfg, k, oversample=9):
    """Exact g2 averaged over each histogram bin (removes binning bias)."""
    off = (np.arange(oversample) + 0.5) / oversample - 0.5
    fine = (cfg.bin_centers[:, None] + off[None, :] * cfg.bin_width).ravel()
    g = g2_exact(fine, *k).reshape(cfg.n_bins, oversample)
    return g.mean(1)


val_curves = {}
t0 = time.time()
for name, p in regimes.items():
    site = EmitterSite(p, 1)
    T = {"NV-like": 3.0, "hBN-like": 1.0, "GaN-like": 0.4}[name]
    t_a, t_b = simulate_photon_stream(site, T, rng, imp=None)
    hist = correlate(t_a, t_b, cfg)
    r_a, r_b = len(t_a) / T, len(t_b) / T
    flat = r_a * r_b * (cfg.bin_width * 1e-9) * T
    g2_mc = hist / flat
    # exact reference from the *actual* CTMC rates used by the simulator
    k = rates_from_site(p["tau1"], p["tau2"], p["a"])
    g2_ref = g2_exact_binavg(cfg, k)
    t1e, t2e, ae = effective_params(*k)
    # deviation metric: mean absolute deviation over bins, relative to the
    # dynamic range, plus Poisson-consistency chi2
    sd = np.sqrt(np.maximum(hist, 1)) / flat
    resid = (g2_mc - g2_ref)
    chi2 = float(np.mean((resid / sd) ** 2))
    mad = float(np.mean(np.abs(resid)))
    nrmse = float(np.sqrt(np.mean(resid ** 2)))
    val_curves[name] = dict(tau=cfg.bin_centers.tolist(),
                            g2_mc=g2_mc.tolist(), g2_exact=g2_ref.tolist(),
                            counts=hist.tolist(), flat=flat,
                            chi2_red=chi2, mad=mad, nrmse=nrmse,
                            eff=dict(tau1=t1e, tau2=t2e, a=ae),
                            n_coinc=int(hist.sum()), T_s=T)
    print(f"[a] {name}: {int(hist.sum())} coincidences, chi2_red={chi2:.3f}, "
          f"MAD={mad:.4f}, NRMSE={nrmse:.4f}  ({time.time()-t0:.1f}s)")

out["mc_vs_exact"] = val_curves

# ----------------------------------------------------------------- (b)
# Twin vs MC with background + N=2 emitters + IRF: compare per-bin mean and
# variance across repeated short acquisitions.
p = dict(tau1=14.0, tau2=250.0, a=0.8, rate_kcps=500, rho=0.8,
         blinking=False, t_on_ms=1, t_off_ms=1, platform="NV")
site2 = EmitterSite(p, 2)
cfg_irf = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.35)
T_short = 1.0
n_rep = 80
mc_h = []
for i in range(n_rep):
    ta, tb = simulate_photon_stream(site2, T_short, rng, imp=None)
    ta = np.sort(ta + rng.normal(0, cfg_irf.sigma_irf, len(ta)))
    tb = np.sort(tb + rng.normal(0, cfg_irf.sigma_irf, len(tb)))
    mc_h.append(correlate(ta, tb, cfg_irf))
mc_h = np.array(mc_h)
mu_twin = expected_histogram(site2, T_short, cfg_irf)
mu_mc, var_mc = mc_h.mean(0), mc_h.var(0)
rel_mean_err = float(np.mean(np.abs(mu_mc - mu_twin) / np.maximum(mu_twin, 1)))
fano_mc = float(np.mean(var_mc / np.maximum(mu_mc, 1e-9)))
out["twin_vs_mc"] = dict(mu_twin=mu_twin.tolist(), mu_mc=mu_mc.tolist(),
                         var_mc=var_mc.tolist(), tau=cfg_irf.bin_centers.tolist(),
                         rel_mean_err=rel_mean_err, fano=fano_mc,
                         n_rep=n_rep, T_s=T_short,
                         g2_0_true=site2.g2_0)
print(f"[b] twin-vs-MC: rel mean err={rel_mean_err*100:.2f}%, Fano={fano_mc:.3f}")

# headline validation numbers
out["summary"] = dict(
    max_mad=max(v["mad"] for v in val_curves.values()),
    max_chi2=max(v["chi2_red"] for v in val_curves.values()),
    max_nrmse=max(v["nrmse"] for v in val_curves.values()),
    rel_mean_err_pct=rel_mean_err * 100,
    fano=fano_mc,
)
with open("/home/claude/sparq/results/exp1_validation.json", "w") as f:
    json.dump(out, f)
print("saved. total", time.time() - t0, "s")
