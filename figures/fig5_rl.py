"""Figure 5 -- reinforcement-learned triage."""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json
sys.path.insert(0, _ROOT + "/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, INK, INK2, MUTED, BASE, despine, large_fonts
from style import panel_label_large as panel_label
from matplotlib.gridspec import GridSpec
large_fonts()

with open(_ROOT + "/results/exp5_rl.json") as f:
    e5 = json.load(f)
with open(_ROOT + "/results/exp5_dwell.json") as f:
    dw = json.load(f)
with open(_ROOT + "/results/exp5b_oracle.json") as f:
    orc = json.load(f)

# (a) learning curves across the full width; (b) and (c) below
fig = plt.figure(figsize=(7.0, 6.0))
gs = GridSpec(2, 2, figure=fig, left=0.095, right=0.985, top=0.885,
              bottom=0.08, wspace=0.30, hspace=0.45)
axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])]

# (a) learning curves: throughput (top) with recall encoded via the
# replay ablation in panel (b); here PER vs uniform throughput + bounds
ax = axes[0]
for key, col, lbl in (("per", C["violet"], "RL agent, prioritized replay"),
                      ("uniform", C["yellow"], "RL agent, uniform replay")):
    curves = e5["curves"][key]
    steps = [p["step"] for p in curves[0]]
    vals = np.array([[p["good_per_min"] for p in c] for c in curves])
    ax.plot(steps, vals.mean(0), "-o", ms=3.2, color=col, lw=1.5, label=lbl)
    ax.fill_between(steps, vals.min(0), vals.max(0), color=col, alpha=0.20,
                    lw=0)
    # mark final recall next to the curve end
    rec = np.mean([c[-1]["recall"] for c in curves])
    dy = 20 if key == "uniform" else -14
    ax.annotate(f"recall {rec:.2f}", (steps[-1], vals.mean(0)[-1]),
                textcoords="offset points", xytext=(4, dy), fontsize=8.5,
                color=col, ha="right", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.4))
# baselines (gate-compliant: precision >= 0.88, recall >= 0.85)
def gate(s):
    return s["precision"][0] >= 0.88 and s["recall"][0] >= 0.85
best_raster = min((v for k, v in e5["baselines"].items()
                   if k.startswith("raster") and gate(v)),
                  key=lambda s: s["time_s"][0])
best_heur = min((v for k, v in e5["baselines"].items()
                 if k.startswith("heuristic") and gate(v)),
                key=lambda s: s["time_s"][0])
ax.axhline(best_raster["good_per_min"][0], color=MUTED, lw=0.9, ls="--")
ax.text(0.28, best_raster["good_per_min"][0] - 0.30, "fixed-dwell raster",
        fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform(),
        ha="left", va="top")
ax.axhline(best_heur["good_per_min"][0], color=C["aqua"], lw=0.9, ls="--")
ax.text(0.03, best_heur["good_per_min"][0] + 0.18, "threshold controller",
        fontsize=8.5, color=C["aqua"], transform=ax.get_yaxis_transform(),
        ha="left")
ax.axhline(orc["good_per_min"][0], color=INK, lw=0.9, ls=":")
ax.text(0.03, orc["good_per_min"][0] - 0.62, "oracle stopping (bound)",
        fontsize=8.5, color=INK, transform=ax.get_yaxis_transform(),
        ha="left")
ax.set_xlim(2000, 47000)
ax.set_ylim(0, 13.2)
ax.set_xlabel("training steps")
ax.set_ylabel("certified good / min")
h0, l0 = ax.get_legend_handles_labels()
fig.legend(h0, l0, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=2, frameon=False, fontsize=8.5, handlelength=1.6,
           columnspacing=1.0, handletextpad=0.45)
despine(ax)
panel_label(ax, "(a)", dx=-0.075, dy=1.12)

# (b) time vs recall, precision-gated
ax = axes[1]
rasters = sorted(((k, v) for k, v in e5["baselines"].items()
                  if k.startswith("raster")),
                 key=lambda kv: float(kv[0].split("_")[1]))
xs = [v["time_s"][0] for _, v in rasters]
ys = [v["recall"][0] for _, v in rasters]
ok = [gate(v) for _, v in rasters]
ax.plot(xs, ys, "-", color=MUTED, lw=1.0, zorder=2)
for x, y, o, (k, v) in zip(xs, ys, ok, rasters):
    ax.scatter([x], [y], s=26, facecolor=MUTED if o else "white",
               edgecolor=MUTED, lw=0.8, zorder=3)
    _off = {"0.5": (7, -2), "4.0": (-10, 8)}.get(k.split("_")[1], (4, -12))
    ax.annotate(k.split("_")[1] + " s", (x, y), textcoords="offset points",
                xytext=_off, fontsize=7.5, color=INK2)
heur = sorted(((k, v) for k, v in e5["baselines"].items()
               if k.startswith("heuristic")),
              key=lambda kv: float(kv[0].split("_")[1]))
xs = [v["time_s"][0] for _, v in heur]
ys = [v["recall"][0] for _, v in heur]
ok = [gate(v) for _, v in heur]
ax.plot(xs, ys, "-", color=C["aqua"], lw=1.0, zorder=2)
for x, y, o in zip(xs, ys, ok):
    ax.scatter([x], [y], s=26, facecolor=C["aqua"] if o else "white",
               edgecolor=C["aqua"], lw=0.8, zorder=3)
sac = e5["best_per"]
ax.scatter([sac["time_s"][0]], [sac["recall"][0]], marker="*", s=220,
           color=C["violet"], zorder=5)
ax.annotate("RL, prioritized", (sac["time_s"][0], sac["recall"][0]),
            textcoords="offset points", xytext=(4, -16), fontsize=8.5,
            color=C["violet"], fontweight="bold")
for f in e5["final"]["uniform"]:
    ax.scatter([f["time_s"][0]], [f["recall"][0]], marker="s", s=34,
               facecolor=C["yellow"] if gate(f) else "white",
               edgecolor=C["yellow"], lw=1.0, zorder=4)
ax.annotate("RL, uniform\n(recall below gate)",
            (np.mean([f["time_s"][0] for f in e5["final"]["uniform"]]),
             np.mean([f["recall"][0] for f in e5["final"]["uniform"]])),
            textcoords="offset points", xytext=(8, -18), fontsize=8.0,
            color=C["yellow"])
ax.scatter([orc["time_s"][0]], [orc["recall"][0]], marker="D", s=40,
           facecolor="white", edgecolor=INK, lw=1.0, zorder=5)
ax.annotate("oracle", (orc["time_s"][0], orc["recall"][0]),
            textcoords="offset points", xytext=(6, 4), fontsize=8.5,
            color=INK)
ax.set_xlabel("measurement time per 48-site field (s)")
ax.set_ylabel("recall of good emitters")
ax.text(0.985, 0.06, "filled: quality gate met",
        transform=ax.transAxes, ha="right", fontsize=8.0, color=INK2)
despine(ax)
panel_label(ax, "(b)", dx=-0.18, dy=1.12)

# (c) dwell allocation
ax = axes[2]
bins = np.linspace(0, 12, 25)
ax.hist(dw["dwell_rej"], bins=bins, density=True, alpha=0.75,
        color=MUTED, label="rejected sites")
ax.hist(dw["dwell_cert"], bins=bins, density=True, alpha=0.65,
        color=C["green"], label="certified sites")
ax.axvline(2.0, color=INK, lw=0.9, ls=":")
ax.text(2.25, 0.92, "fixed raster\n(2 s each)", fontsize=8.0, color=INK2,
        transform=ax.get_xaxis_transform(), va="top")
ax.set_xlabel("dwell per site (s)")
ax.set_ylabel("density")
ax.legend(loc="upper right", handlelength=1.4, borderpad=0.3,
          labelspacing=0.3, fontsize=8.5)
despine(ax)
panel_label(ax, "(c)", dx=-0.18, dy=1.12)

fig.savefig(_ROOT + "/figures/fig5_rl.pdf")
fig.savefig(_ROOT + "/figures/fig5_rl.png", dpi=300)
print("fig5 done")
