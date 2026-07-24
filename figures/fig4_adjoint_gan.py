"""Figure 4 — adjoint protocol optimization + adversarial sim-to-real."""
import sys, json
sys.path.insert(0, "/home/claude/sparq/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, SEQ, INK, INK2, MUTED, BASE, panel_label, despine

with open("/home/claude/sparq/results/exp3_adjoint.json") as f:
    e3 = json.load(f)
with open("/home/claude/sparq/results/exp4_gan.json") as f:
    e4 = json.load(f)

fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.0))
plt.subplots_adjust(left=0.075, right=0.988, top=0.87, bottom=0.20,
                    wspace=0.55)

# (a) protocol trajectory in the (s, tau_max) plane
ax = axes[0]
tr = e3["trajectory"]
ss = [t["s"] for t in tr] + [e3["s_star"]]
ww = [t["tau_max"] for t in tr] + [e3["tau_max_star"]]
for i in range(len(ss) - 1):
    ax.plot(ss[i:i+2], ww[i:i+2], "-", color=SEQ[min(i + 1, len(SEQ) - 1)],
            lw=1.4, zorder=2)
ax.scatter(ss, ww, c=[SEQ[min(i, len(SEQ) - 1)] for i in range(len(ss))],
           s=12, zorder=3)
ax.scatter([1.0], [60.5], marker="s", s=28, color=MUTED, zorder=4)
ax.annotate("default", (1.0, 60.5), textcoords="offset points",
            xytext=(4, 4), fontsize=5.4, color=INK2)
ax.scatter([e3["s_star"]], [e3["tau_max_star"]], marker="*", s=90,
           color=C["red"], zorder=5)
ax.annotate(rf"$\theta^*$", (e3["s_star"], e3["tau_max_star"]),
            textcoords="offset points", xytext=(6, -2), fontsize=6.5,
            color=C["red"])
ax.set_xlabel("saturation parameter $s$")
ax.set_ylabel(r"window $\tau_{\max}$ (ns)")
despine(ax)
panel_label(ax, "(a)", dx=-0.34, dy=1.16)

# (b) profile Fisher information vs s (log-log) + accuracy-gain inset
ax = axes[1]
sg = np.array(e3["fisher"]["s_grid"])
fi = np.array(e3["fisher"]["fi"])
ax.plot(sg, fi / fi.max(), color=C["blue"], lw=1.3)
ax.axvline(e3["s_star"], color=C["red"], lw=0.9, ls="--")
ax.text(e3["s_star"] * 1.22, 0.0045, "adjoint $s^*$", fontsize=6.4,
        color=C["red"], rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("saturation parameter $s$")
ax.set_ylabel("profile Fisher\ninfo. (norm.)")
despine(ax)
panel_label(ax, "(b)", dx=-0.38, dy=1.16)

# (c) WGAN critic gap
ax = axes[2]
its = [w["it"] for w in e4["wgan_log"]]
ax.plot(its, [w["w_sim"] for w in e4["wgan_log"]], color=MUTED, lw=1.2,
        label="raw twin")
ax.plot(its, [w["w_gan"] for w in e4["wgan_log"]], color=C["orange"],
        lw=1.2, label="GAN-refined")
ax.set_xlabel("generator iteration")
ax.set_ylabel("critic distance to real")
ax.legend(loc="upper right", handlelength=1.2, borderpad=0.2,
          labelspacing=0.25, fontsize=5.4)
despine(ax)
panel_label(ax, "(c)", dx=-0.32, dy=1.16)

# (d) early-estimation MAE on held-out real series
ax = axes[3]
methods = ["fit", "sim", "dr", "gan"]
lbl = {"fit": "peak-area\nanalysis", "sim": "twin\nonly",
       "dr": "+ dom.\nrand.", "gan": "+ WGAN"}
cols = [MUTED, C["blue"], C["yellow"], C["orange"]]
vals = [e4["results"][m]["mae_held_out"] for m in methods]
bars = ax.bar(np.arange(4), vals, 0.58, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=5.2, color=INK2)
try:
    with open("/home/claude/sparq/results/exp4c_floor.json") as f:
        floor = json.load(f)["floor_mae"]
    ax.axhline(floor, color=INK, lw=0.8, ls="--")
    ax.text(1.55, floor - 0.012, "information floor\n(in-domain twin)",
            fontsize=5.2, color=INK, ha="center", va="top",
            bbox=dict(fc="white", ec="none", pad=0.4))
except FileNotFoundError:
    pass
ax.set_xticks(np.arange(4))
ax.set_xticklabels([lbl[m] for m in methods], fontsize=5.4)
ax.set_ylim(0, 0.315)
ax.set_ylabel("$g^{(2)}(0)$ MAE\n(30-s real windows)")
despine(ax)
panel_label(ax, "(d)", dx=-0.40, dy=1.16)

fig.savefig("/home/claude/sparq/figures/fig4_adjoint_gan.pdf")
fig.savefig("/home/claude/sparq/figures/fig4_adjoint_gan.png")
print("fig4 done")
