"""Figure 2 — three-way twin validation (5 panels)."""
import sys, json
sys.path.insert(0, "/home/claude/sparq/figures")
sys.path.insert(0, "/home/claude/sparq")
import numpy as np
import matplotlib.pyplot as plt
from style import C, INK, INK2, MUTED, BASE, panel_label, despine
from sparq.datasets import load_fisequr, rebin_real, CFG
from sparq.estimators import fit_g2_histogram
from sparq.physics import g2_measured

with open("/home/claude/sparq/results/exp1_validation.json") as f:
    e1 = json.load(f)

fig, axes = plt.subplots(1, 5, figsize=(7.05, 2.05))
plt.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.195,
                    wspace=0.55)

cols = {"NV-like": C["blue"], "hBN-like": C["aqua"], "GaN-like": C["yellow"]}
for i, (name, d) in enumerate(e1["mc_vs_exact"].items()):
    ax = axes[i]
    tau = np.array(d["tau"])
    g2mc = np.array(d["g2_mc"])
    g2ex = np.array(d["g2_exact"])
    counts = np.array(d["counts"])
    sd = np.sqrt(np.maximum(counts, 1)) / d["flat"]
    ax.errorbar(tau[::3], g2mc[::3], yerr=sd[::3], fmt="o", ms=1.6,
                lw=0, elinewidth=0.55, color=cols[name], alpha=0.85,
                zorder=2, label="MC stream")
    ax.plot(tau, g2ex, color=INK, lw=1.0, zorder=3, label="exact")
    ax.axhline(1.0, color=BASE, lw=0.5, zorder=1)
    ax.set_xlabel(r"$\tau$ (ns)")
    if i == 0:
        ax.set_ylabel(r"$g^{(2)}(\tau)$")
    ax.set_title(name.replace("-like", ""), color=cols[name], pad=3)
    ax.text(0.96, 0.10, rf"$\chi^2_\nu={d['chi2_red']:.2f}$",
            transform=ax.transAxes, ha="right", fontsize=6.2, color=INK2)
    ax.set_ylim(-0.08, max(2.0, g2ex.max() * 1.15))
    despine(ax)
    panel_label(ax, f"({'abc'[i]})", dx=-0.34, dy=1.22)
axes[0].legend(loc="lower left", fontsize=5.8, handlelength=1.0,
               borderpad=0.25, labelspacing=0.25)

# (d) histogram twin vs stream twin
ax = axes[3]
d = e1["twin_vs_mc"]
tau = np.array(d["tau"])
ax.plot(tau, d["mu_mc"], "o", ms=1.6, color=C["violet"], alpha=0.85,
        label="stream twin")
ax.plot(tau, d["mu_twin"], color=INK, lw=1.0, label="histogram twin")
ax.set_xlabel(r"$\tau$ (ns)")
ax.set_ylabel("mean counts/bin")
ax.text(0.96, 0.10,
        f"err {e1['summary']['rel_mean_err_pct']:.1f}%\n"
        f"Fano {e1['summary']['fano']:.2f}",
        transform=ax.transAxes, ha="right", fontsize=6.2, color=INK2)
ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 1.02), ncol=1,
          fontsize=5.8, frameon=False, handlelength=1.0, borderpad=0.1,
          labelspacing=0.15)
despine(ax)
panel_label(ax, "(d)", dx=-0.40, dy=1.22)

# (e) real pulsed quantum-dot series (comb) + pulsed-twin model fit
from sparq.physics import HBTConfig
from sparq.pulsed import expected_hist_pulsed, calibrate_comb, g2_peak_area
from scipy.optimize import curve_fit

ax = axes[4]
CFGR = HBTConfig(tau_max=45.5, n_bins=91, sigma_irf=0.35)
series = load_fisequr()
s = [x for x in series if x["name"].startswith("10uW_12000")][0]
dly, tot = s["delay"], s["total"]
center, phase, P = calibrate_comb(dly, tot)
g2h = g2_peak_area(dly, tot, center)
hist, _ = rebin_real(dly, tot, CFGR, center=center)
T = s["T_total"]
tau = CFGR.bin_centers


def model(t, rate, g20, tau_e, a, bg, sig, off):
    return expected_hist_pulsed(CFGR, T, rate, g20, tau_e, a=a, tau2=200.0,
                                bg_frac=bg, sigma_pair=sig, center_off=off)


sd = np.sqrt(np.maximum(hist, 1))
popt, _ = curve_fit(
    lambda t, *p: model(t, *p), tau, hist, sigma=sd,
    p0=(6e3, 0.5, 1.5, 0.2, 0.2, 0.5, 0.0),
    bounds=([1e3, 0, 0.3, 0, 0, 0.25, -1], [3e4, 1.2, 5, 1, 1, 1.2, 1]),
    maxfev=6000)
ax.plot(tau, hist, "o", ms=1.6, color=C["orange"], alpha=0.8,
        label="QD data (pulsed)")
ax.plot(tau, model(tau, *popt), color=INK, lw=1.0, label="pulsed twin")
ax.set_xlabel(r"$\tau$ (ns)")
ax.set_ylabel("counts/bin")
ax.text(0.96, 0.88, rf"$g^{{(2)}}(0)={g2h:.2f}$", transform=ax.transAxes,
        ha="right", fontsize=6.2, color=INK2,
        bbox=dict(fc="white", ec="none", pad=0.6))
ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 1.02), ncol=1,
          fontsize=5.8, frameon=False, handlelength=1.0, borderpad=0.1,
          labelspacing=0.15)
despine(ax)
panel_label(ax, "(e)", dx=-0.40, dy=1.22)

fig.savefig("/home/claude/sparq/figures/fig2_validation.pdf")
fig.savefig("/home/claude/sparq/figures/fig2_validation.png")
print("fig2 done, real-series g2(0) fit:", g2h)
