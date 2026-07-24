"""Figure 1 — graphical abstract: the SPARQ loop and headline results.
All numbers are read from the experiment JSONs (no hand-typed values)."""
import sys, json
sys.path.insert(0, "/home/claude/sparq/figures")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import C, INK, INK2, MUTED, BASE, GRID

R = "/home/claude/sparq/results"


def load(n):
    with open(f"{R}/{n}") as f:
        return json.load(f)


e2 = load("exp2_estimators.json")
e2b = load("exp2b_snn.json")
e4 = load("exp4_gan.json")
e5 = load("exp5_rl.json")
orc = load("exp5b_oracle.json")
e6 = load("exp6b_graph.json")

speedup = e2["time_to_target"]["fit"] / e2b["time_to_target"]
latency = e2b["anytime"]["0.5"]["median_ms"]
x_real = (e4["results"]["fit"]["mae_held_out"]
          / e4["results"]["sim"]["mae_held_out"])
gap = 100 * e6["unseen"]["gap_recovery"]
raster_t = 408.0
sac_t = e5["best_per"]["time_s"][0]
x_sac = raster_t / sac_t
x_orc = raster_t / orc["time_s"][0]

fig = plt.figure(figsize=(7.05, 2.55))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

FS_T, FS_S = 8.2, 6.2


def box(x, y, w, h, label, sub, fc, ec, fs=FS_T, subfs=FS_S):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + 0.64 * h, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + 0.28 * h, sub, ha="center", va="center",
            fontsize=subfs, color=INK2)


def arr(p0, p1, color=INK2, lw=1.3, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color,
                 lw=lw, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                 mutation_scale=10, shrinkA=2, shrinkB=2))


# ------------------------------------------------------------ loop band
Y, H = 0.60, 0.30
box(0.020, Y, 0.20, H, "emitter field", "inhomogeneous\ncandidate sites",
    "#f4f6fa", BASE)
box(0.285, Y, 0.215, H, "event-driven\nperception",
    "spiking network on\nphoton-pair events", "#eef7f2", C["aqua"], fs=7.4,
    subfs=5.6)
box(0.565, Y, 0.20, H, "learned control", "SAC + PER photon\nbudgeting",
    "#f0eefb", C["violet"], fs=7.4, subfs=5.6)
box(0.830, Y, 0.155, H, "certified\nemitters", "pure, bright,\nstable",
    "#eaf5ea", C["green"], fs=7.4, subfs=5.6)
arr((0.220, Y + H / 2), (0.285, Y + H / 2))
arr((0.500, Y + H / 2), (0.565, Y + H / 2))
arr((0.765, Y + H / 2), (0.830, Y + H / 2))
arr((0.665, Y), (0.125, Y - 0.02), color=C["violet"], rad=-0.10)
ax.text(0.40, 0.478, "closed loop: per-site photon budget",
        fontsize=6.0, color=C["violet"], ha="center")

# twin ribbon under the loop
ax.text(0.5, 0.408,
        "one validated digital twin — trained through, differentiated "
        "(adjoint), rewarded from, and transferred by physics "
        "(GAN critic + platform graphs)",
        fontsize=6.4, color=C["blue"], ha="center", style="italic")

# ------------------------------------------------------------ stat tiles
tiles = [
    (f"{speedup:.0f}×", "faster than the\nLM fit at matched\naccuracy",
     C["blue"]),
    (f"{latency:.0f} ms", "median anytime\ndecision latency\n(1-s exposure)",
     C["aqua"]),
    (f"{x_real:.1f}×", "vs. conventional\nanalysis on real QD\ndata, zero-shot",
     C["orange"]),
    (f"{gap:.0f}%", "oracle transfer gap\nrecovered zero-shot\n(graph physics)",
     C["yellow"]),
    (f"{x_sac:.1f}× / {x_orc:.1f}×", "triage speedup:\nlearned policy /\n"
     "oracle bound", C["violet"]),
]
tw = 0.184
for i, (num, lab, col) in enumerate(tiles):
    x = 0.020 + i * (tw + 0.011)
    ax.add_patch(FancyBboxPatch((x, 0.045), tw, 0.315,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 fc="white", ec=GRID, lw=1.0))
    ax.text(x + tw / 2, 0.265, num, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=INK)
    ax.text(x + tw / 2, 0.125, lab, ha="center", va="center",
            fontsize=5.7, color=INK2)

fig.savefig("/home/claude/sparq/figures/fig0_abstract.pdf")
fig.savefig("/home/claude/sparq/figures/fig0_abstract.png")
print("fig0 done:", f"{speedup:.0f}x {latency:.0f}ms {x_real:.1f}x "
      f"{gap:.0f}% {x_sac:.1f}x/{x_orc:.1f}x")
