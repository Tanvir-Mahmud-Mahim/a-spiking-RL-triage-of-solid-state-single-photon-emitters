"""Figure 4 -- adjoint protocol optimization + adversarial sim-to-real."""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json
sys.path.insert(0, _ROOT + "/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, SEQ, INK, INK2, MUTED, BASE, despine, large_fonts
from style import panel_label_large as panel_label
large_fonts()

with open(_ROOT + "/results/exp3_adjoint.json") as f:
    e3 = json.load(f)
with open(_ROOT + "/results/exp4_gan.json") as f:
    e4 = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.9))
axes = axes.ravel()
plt.subplots_adjust(left=0.105, right=0.985, top=0.945, bottom=0.085,
                    wspace=0.36, hspace=0.42)

# (a) protocol trajectory in the (s, tau_max) plane
ax = axes[0]
tr = e3["trajectory"]
ss = [t["s"] for t in tr] + [e3["s_star"]]
ww = [t["tau_max"] for t in tr] + [e3["tau_max_star"]]
for i in range(len(ss) - 1):
    ax.plot(ss[i:i+2], ww[i:i+2], "-", color=SEQ[min(i + 1, len(SEQ) - 1)],
            lw=1.8, zorder=2)
ax.scatter(ss, ww, c=[SEQ[min(i, len(SEQ) - 1)] for i in range(len(ss))],
           s=24, zorder=3)
ax.scatter([1.0], [60.5], marker="s", s=50, color=MUTED, zorder=4)
# label below the square, clear of the descending light-blue
# trajectory that passes above-right of it
ax.annotate("default", (1.0, 60.5), textcoords="offset points",
            xytext=(2, -26), fontsize=8.5, color=INK2)
ax.scatter([e3["s_star"]], [e3["tau_max_star"]], marker="*", s=170,
           color=C["red"], zorder=5)
ax.annotate(rf"$\theta^*$", (e3["s_star"], e3["tau_max_star"]),
            textcoords="offset points", xytext=(-5, 11), fontsize=10,
            color=C["red"])
ax.set_xlabel("saturation parameter $s$")
ax.set_ylabel(r"window $\tau_{\max}$ (ns)")
despine(ax)
panel_label(ax, "(a)", dx=-0.20, dy=1.10)

# (b) profile Fisher information vs s (log-log) + accuracy-gain inset
ax = axes[1]
sg = np.array(e3["fisher"]["s_grid"])
fi = np.array(e3["fisher"]["fi"])
ax.plot(sg, fi / fi.max(), color=C["blue"], lw=1.6)
ax.axvline(e3["s_star"], color=C["red"], lw=0.9, ls="--")
ax.text(e3["s_star"] * 1.22, 0.0045, "optimized $s^*$", fontsize=8.5,
        color=C["red"], rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("saturation parameter $s$")
ax.set_ylabel("profile Fisher information\n(normalized)")
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5]))
ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(1e-3, 1.6)
despine(ax)
panel_label(ax, "(b)", dx=-0.20, dy=1.10)

# (c) WGAN critic gap
ax = axes[2]
its = [w["it"] for w in e4["wgan_log"]]
ax.plot(its, [w["w_sim"] for w in e4["wgan_log"]], color=MUTED, lw=1.2,
        label="simulator output")
ax.plot(its, [w["w_gan"] for w in e4["wgan_log"]], color=C["orange"],
        lw=1.2, label="GAN-refined")
ax.set_xlabel("generator iteration")
ax.set_ylabel("critic distance to experiment")
ax.legend(loc="upper right", handlelength=1.4, borderpad=0.3,
          labelspacing=0.3, fontsize=8.5)
despine(ax)
panel_label(ax, "(c)", dx=-0.20, dy=1.10)

# (d) early-estimation MAE on held-out real series
ax = axes[3]
methods = ["fit", "sim", "dr", "gan"]
lbl = {"fit": "peak-area\nanalysis", "sim": "simulation\nonly",
       "dr": "+ domain\nrandom.", "gan": "+ GAN\nrefiner"}
cols = [MUTED, C["blue"], C["yellow"], C["orange"]]
vals = [e4["results"][m]["mae_held_out"] for m in methods]
bars = ax.bar(np.arange(4), vals, 0.58, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5, color=INK2)
try:
    with open(_ROOT + "/results/exp4c_floor.json") as f:
        floor = json.load(f)["floor_mae"]
    fl = ax.axhline(floor, color=INK, lw=1.0, ls="--")
    # label in the clear region above the short bars; a white-boxed label
    # sitting on the dashed line would visibly punch through the bars
    ax.legend([fl], ["same network,\nsimulated 30-s windows"],
              loc="upper right", fontsize=8.5, handlelength=1.8,
              borderpad=0.3)
except FileNotFoundError:
    pass
ax.set_xticks(np.arange(4))
ax.set_xticklabels([lbl[m] for m in methods], fontsize=8.5)
ax.set_ylim(0, 0.315)
ax.set_ylabel("$g^{(2)}(0)$ MAE\n(30-s experimental windows)")
despine(ax)
panel_label(ax, "(d)", dx=-0.20, dy=1.10)

fig.savefig(_ROOT + "/figures/fig4_adjoint_gan.pdf")
fig.savefig(_ROOT + "/figures/fig4_adjoint_gan.png", dpi=300)
print("fig4 done")
