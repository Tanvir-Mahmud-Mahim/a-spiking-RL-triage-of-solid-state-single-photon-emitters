"""Figure 3 -- estimator envelope, anytime latency, energy accounting."""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json
sys.path.insert(0, _ROOT + "/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, INK, INK2, MUTED, BASE, despine, large_fonts
from style import panel_label_large as panel_label
large_fonts()

with open(_ROOT + "/results/exp2_estimators.json") as f:
    e2 = json.load(f)
try:      # activity-regularized SNN supersedes the exp2 SNN
    with open(_ROOT + "/results/exp2b_snn.json") as f:
        e2b = json.load(f)
    e2["results"]["snn_pitl"] = e2b["snn_sparse"]
    e2["anytime"] = e2b["anytime"]
    e2["energy"] = e2b["energy"]
except FileNotFoundError:
    pass
T = np.array(e2["T_grid"])
COLS = {"fit": MUTED, "cnn_clean": C["yellow"], "cnn_pitl": C["blue"],
        "snn_pitl": C["aqua"]}
LBL = {"fit": "LM fit", "cnn_clean": "CNN, long-exposure training",
       "cnn_pitl": "CNN, variable-exposure training",
       "snn_pitl": "SNN, variable-exposure training"}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.1))
axes = axes.ravel()
# headroom at the top hosts the shared figure-level legend so no legend
# box covers data inside the panels
plt.subplots_adjust(left=0.095, right=0.985, top=0.865, bottom=0.075,
                    wspace=0.30, hspace=0.42)

# (a) accuracy vs T
ax = axes[0]
for m in ("fit", "cnn_clean", "cnn_pitl", "snn_pitl"):
    a = np.array(e2["results"][m]["acc"])
    ax.plot(T, 100 * a[:, 0], "-o", ms=3.2, color=COLS[m], label=LBL[m],
            lw=1.2, zorder=3)
    ax.fill_between(T, 100 * (a[:, 0] - a[:, 1]), 100 * (a[:, 0] + a[:, 1]),
                    color=COLS[m], alpha=0.18, lw=0)
# MC-Bayes reference (reliable in the photon-sparse regime; the finite
# 6000-site reference sample biases it low once likelihoods concentrate)
Tb = [t for t in T if t <= 1.0]
# Bayes reference with M = 48 000 reference sites, mean over five
# independent reference samples (results/exp8_parts, second review round)
import glob, os
_P = _ROOT + "/results/exp8_parts"
_seeds = {}
for fn_ in glob.glob(_P + "/C2*_seed*.json"):
    sd = fn_.split("_seed")[1][:-5]
    _seeds.setdefault(sd, {}).update(json.load(open(fn_)))
bay = np.array([np.mean([_seeds[k][str(t)]["48000"] for k in _seeds])
                for t in Tb])
ax.plot(Tb, 100 * bay, "--", color=INK, lw=1.1, label="Bayes reference",
        zorder=5)
tgt = 100 * e2["target_acc"]
ax.axhline(tgt, color=C["red"], lw=0.7, ls=":", zorder=1)
ax.text(0.02, (tgt + 0.9 - 48) / (101 - 48), "target", fontsize=8.5,
        color=C["red"], transform=ax.transAxes, ha="left")
ax.set_xscale("log")
ax.set_ylim(48, 101)
ax.set_xlabel("acquisition time (s)")
ax.set_ylabel("balanced accuracy (%)")
# legend moved outside the axes (figure-level, one row across the top)
# so the curves' photon-sparse tails stay fully visible
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.6,
           columnspacing=1.1, handletextpad=0.45)
despine(ax)
panel_label(ax, "(a)", dx=-0.17, dy=1.10)

# (b) MAE vs T
ax = axes[1]
for m in ("fit", "cnn_clean", "cnn_pitl", "snn_pitl"):
    a = np.array(e2["results"][m]["mae"])
    ax.plot(T, a[:, 0], "-o", ms=3.2, color=COLS[m], lw=1.2)
    ax.fill_between(T, a[:, 0] - a[:, 1], a[:, 0] + a[:, 1],
                    color=COLS[m], alpha=0.18, lw=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("acquisition time (s)")
ax.set_ylabel(r"$g^{(2)}(0)$ MAE")
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4]))
ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.yaxis.set_minor_formatter(NullFormatter())
despine(ax)
panel_label(ax, "(b)", dx=-0.17, dy=1.10)

# (c) anytime: accuracy vs median latency
ax = axes[2]
ths = sorted(e2["anytime"].keys(), key=float)
lat = [e2["anytime"][t]["median_ms"] for t in ths]
acc = [100 * e2["anytime"][t]["acc"] for t in ths]
ax.plot(lat, acc, "-o", color=C["aqua"], ms=4.2, lw=1.4)
offs = {"0.5": (8, -4), "0.6": (8, -10), "0.8": (6, -20),
        "0.9": (2, -20), "0.95": (-24, -20)}
for t, x, y in zip(ths, lat, acc):
    ax.annotate(rf"$\theta$={t}", (x, y), textcoords="offset points",
                xytext=offs.get(t, (5, -8)), fontsize=8.5, color=INK2)
# full-exposure reference
full_acc = 100 * np.array(e2["results"]["snn_pitl"]["acc"])[
    e2["T_grid"].index(1.0), 0]
ax.axhline(full_acc, color=BASE, lw=0.7, ls="--")
ax.text(0.03, 0.90, "full 1-s\nexposure", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color=INK2)
ax.set_xlim(230, 900)
ax.set_xlabel("median decision time (ms)")
ax.set_ylabel("balanced accuracy (%)")
despine(ax)
panel_label(ax, "(c)", dx=-0.17, dy=1.10)

# (d) energy per decision vs T
ax = axes[3]
Ts = sorted(e2["energy"].keys(), key=float)
x = np.arange(len(Ts))
snn = [e2["energy"][t]["e_snn_nJ"] for t in Ts]
fp = [e2["energy"][t]["e_cnn_fp32_nJ"] for t in Ts]
i8 = [e2["energy"][t]["e_cnn_int8_nJ"] for t in Ts]
w = 0.27
ax.bar(x - w, snn, w, color=C["aqua"], label="SNN (23.6 pJ/synop)")
ax.bar(x, i8, w, color="#9ec5f4", label="CNN (INT8, 0.23 pJ/MAC)")
ax.bar(x + w, fp, w, color=C["blue"], label="CNN (FP32, 4.6 pJ/MAC)")
for xi, v in zip(x - w, snn):
    ax.text(xi, v * 1.3, f"{v:,.0f}", ha="center", fontsize=8.0, color=INK2)
ax.set_yscale("log")
ax.set_ylim(10, max(snn + fp) * 900)
ax.set_xticks(x)
ax.set_xticklabels([f"{float(t):g} s" for t in Ts])
ax.set_xlabel("acquisition time")
ax.set_ylabel("energy/decision (nJ)")
ax.legend(loc="upper left", handlelength=1.2, borderpad=0.3,
          labelspacing=0.3, fontsize=8.5, ncol=1)
despine(ax)
panel_label(ax, "(d)", dx=-0.17, dy=1.10)

fig.savefig(_ROOT + "/figures/fig3_estimators.pdf")
fig.savefig(_ROOT + "/figures/fig3_estimators.png", dpi=300)
print("fig3 done")
