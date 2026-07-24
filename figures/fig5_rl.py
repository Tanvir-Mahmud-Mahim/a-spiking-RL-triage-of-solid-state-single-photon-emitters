"""Figure 5 — reinforcement-learned triage."""
import sys, json
sys.path.insert(0, "/home/claude/sparq/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, INK, INK2, MUTED, BASE, panel_label, despine

with open("/home/claude/sparq/results/exp5_rl.json") as f:
    e5 = json.load(f)
with open("/home/claude/sparq/results/exp5_dwell.json") as f:
    dw = json.load(f)
with open("/home/claude/sparq/results/exp5b_oracle.json") as f:
    orc = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.10))
plt.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.19,
                    wspace=0.38)

# (a) learning curves: throughput (top) with recall encoded via the
# replay ablation in panel (b); here PER vs uniform throughput + bounds
ax = axes[0]
for key, col, lbl in (("per", C["violet"], "SAC + PER"),
                      ("uniform", C["yellow"], "SAC, uniform replay")):
    curves = e5["curves"][key]
    steps = [p["step"] for p in curves[0]]
    vals = np.array([[p["good_per_min"] for p in c] for c in curves])
    ax.plot(steps, vals.mean(0), "-o", ms=2.0, color=col, lw=1.2, label=lbl)
    ax.fill_between(steps, vals.min(0), vals.max(0), color=col, alpha=0.20,
                    lw=0)
    # mark final recall next to the curve end
    rec = np.mean([c[-1]["recall"] for c in curves])
    dy = 9 if key == "uniform" else -14
    ax.annotate(f"recall {rec:.2f}", (steps[-1], vals.mean(0)[-1]),
                textcoords="offset points", xytext=(4, dy), fontsize=6.0,
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
ax.text(0.03, best_raster["good_per_min"][0] - 0.42, "raster (at gate)",
        fontsize=6.0, color=INK2, transform=ax.get_yaxis_transform(),
        ha="left", bbox=dict(fc="white", ec="none", pad=0.4))
ax.axhline(best_heur["good_per_min"][0], color=C["aqua"], lw=0.9, ls="--")
ax.text(0.03, best_heur["good_per_min"][0] + 0.18, "gated controller",
        fontsize=6.0, color=C["aqua"], transform=ax.get_yaxis_transform(),
        ha="left")
ax.axhline(orc["good_per_min"][0], color=INK, lw=0.9, ls=":")
ax.text(0.03, orc["good_per_min"][0] - 0.62, "oracle stopping",
        fontsize=6.0, color=INK, transform=ax.get_yaxis_transform(),
        ha="left")
ax.set_xlim(2000, 47000)
ax.set_ylim(0, 13.2)
ax.set_xlabel("environment steps")
ax.set_ylabel("certified good / min")
ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.86), handlelength=1.2,
          borderpad=0.35, labelspacing=0.35, fontsize=6.0)
despine(ax)
panel_label(ax, "(a)", dx=-0.24, dy=1.13)

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
    ax.scatter([x], [y], s=16, facecolor=MUTED if o else "white",
               edgecolor=MUTED, lw=0.8, zorder=3)
    ax.annotate(k.split("_")[1] + " s", (x, y), textcoords="offset points",
                xytext=(3, -8), fontsize=4.8, color=INK2)
heur = sorted(((k, v) for k, v in e5["baselines"].items()
               if k.startswith("heuristic")),
              key=lambda kv: float(kv[0].split("_")[1]))
xs = [v["time_s"][0] for _, v in heur]
ys = [v["recall"][0] for _, v in heur]
ok = [gate(v) for _, v in heur]
ax.plot(xs, ys, "-", color=C["aqua"], lw=1.0, zorder=2)
for x, y, o in zip(xs, ys, ok):
    ax.scatter([x], [y], s=16, facecolor=C["aqua"] if o else "white",
               edgecolor=C["aqua"], lw=0.8, zorder=3)
sac = e5["best_per"]
ax.scatter([sac["time_s"][0]], [sac["recall"][0]], marker="*", s=120,
           color=C["violet"], zorder=5)
ax.annotate("SAC + PER", (sac["time_s"][0], sac["recall"][0]),
            textcoords="offset points", xytext=(6, -3), fontsize=6,
            color=C["violet"], fontweight="bold")
for f in e5["final"]["uniform"]:
    ax.scatter([f["time_s"][0]], [f["recall"][0]], marker="s", s=20,
               facecolor=C["yellow"] if gate(f) else "white",
               edgecolor=C["yellow"], lw=1.0, zorder=4)
ax.annotate("SAC, uniform\n(recall-deficient)",
            (np.mean([f["time_s"][0] for f in e5["final"]["uniform"]]),
             np.mean([f["recall"][0] for f in e5["final"]["uniform"]])),
            textcoords="offset points", xytext=(8, -14), fontsize=5.2,
            color=C["yellow"])
ax.scatter([orc["time_s"][0]], [orc["recall"][0]], marker="D", s=26,
           facecolor="white", edgecolor=INK, lw=1.0, zorder=5)
ax.annotate("oracle", (orc["time_s"][0], orc["recall"][0]),
            textcoords="offset points", xytext=(5, 4), fontsize=5.6,
            color=INK)
ax.set_xlabel("measurement time per 48-site field (s)")
ax.set_ylabel("recall of good emitters")
ax.text(0.985, 0.06, "filled: quality gate met",
        transform=ax.transAxes, ha="right", fontsize=5.2, color=INK2)
despine(ax)
panel_label(ax, "(b)", dx=-0.26, dy=1.13)

# (c) dwell allocation
ax = axes[2]
bins = np.linspace(0, 12, 25)
ax.hist(dw["dwell_rej"], bins=bins, density=True, alpha=0.75,
        color=MUTED, label="rejected sites")
ax.hist(dw["dwell_cert"], bins=bins, density=True, alpha=0.65,
        color=C["green"], label="certified sites")
ax.axvline(2.0, color=INK, lw=0.9, ls=":")
ax.text(2.15, 0.92, "fixed raster\n(2 s each)", fontsize=5.2, color=INK2,
        transform=ax.get_xaxis_transform(), va="top")
ax.set_xlabel("dwell per site (s)")
ax.set_ylabel("density")
ax.legend(loc="upper right", handlelength=1.2, borderpad=0.2,
          labelspacing=0.25, fontsize=5.4)
despine(ax)
panel_label(ax, "(c)", dx=-0.24, dy=1.13)

fig.savefig("/home/claude/sparq/figures/fig5_rl.pdf")
fig.savefig("/home/claude/sparq/figures/fig5_rl.png")
print("fig5 done")
