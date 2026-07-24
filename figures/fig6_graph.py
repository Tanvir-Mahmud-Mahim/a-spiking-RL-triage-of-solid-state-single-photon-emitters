"""Figure 6 — graph-encoded cross-platform transfer (single column)."""
import sys, json
sys.path.insert(0, "/home/claude/sparq/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, INK, INK2, MUTED, BASE, panel_label, despine

with open("/home/claude/sparq/results/exp6b_graph.json") as f:
    e6 = json.load(f)

plats = ["NV", "hBN", "GaN", "SiV"]
Ts = [0.3, 3.0]
models = [("uncond_syn", MUTED, "unconditioned (synthetic training)"),
          ("graph_syn", C["yellow"], "graph-conditioned (synthetic training)"),
          ("oracle_real", C["blue"], "oracle (trained on the real four)")]

fig, ax = plt.subplots(figsize=(3.42, 2.35))
plt.subplots_adjust(left=0.13, right=0.99, top=0.78, bottom=0.155)

x = np.arange(len(plats))
w = 0.26
for j, (m, col, lbl) in enumerate(models):
    means = []
    sds = []
    for p in plats:
        acc = np.array([e6["results"][m][f"{p}@{t}"]["acc"] for t in Ts])
        means.append(100 * acc[:, 0].mean())
        sds.append(100 * np.sqrt((acc[:, 1] ** 2).mean()))
    ax.bar(x + (j - 1) * w, means, w, color=col, label=lbl,
           yerr=sds, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK2))
ax.set_xlabel("evaluation platform (zero-shot for the synthetic-trained"
              " models)", fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels(plats)
ax.set_ylim(55, 95)
ax.set_ylabel("balanced accuracy (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1,
          frameon=False, handlelength=1.2, borderpad=0.1,
          labelspacing=0.25, fontsize=6.2)
despine(ax)

fig.savefig("/home/claude/sparq/figures/fig6_graph.pdf")
fig.savefig("/home/claude/sparq/figures/fig6_graph.png")
print("fig6 done")
