"""Figure 1 -- experimental pipeline, early decision, and training/transfer.

(a) Confocal Hanbury Brown--Twiss (HBT) measurement of one candidate site
    and the per-site decision loop.
(b) Output of the trained spiking estimator (results/fig1_traces.json)
    after every 31-ms time slice for two simulated sites.
(c) Supporting infrastructure: the three-level photon-counting simulator
    that supplies training data, and the level-structure graphs used for
    transfer between emitter classes.
"""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import sys
sys.path.insert(0, _ROOT + "/figures")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Rectangle, Polygon, Wedge)
from style import C, INK, INK2, MUTED, GRID, panel_label, despine

FS = 7.6        # box titles
FS_S = 6.9      # secondary text

W, H = 7.0, 4.62
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, h, title, sub=None, ec=INK2, fc="white", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 ec=ec, fc=fc, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=FS, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=FS_S, color=INK2, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=FS, color=INK, fontweight="bold",
                linespacing=1.15)


def arrow(p0, p1, color=INK2, lw=1.0, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color,
                 lw=lw, mutation_scale=9, shrinkA=1, shrinkB=1,
                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# ======================================================== panel (a)
ax.text(0.05, H - 0.10, "(a)", fontsize=9.5, fontweight="bold", va="top")

# sample: confocal field of candidate sites
sx, sy, sw = 0.25, 3.30, 1.15
ax.add_patch(Rectangle((sx, sy), sw, sw, fc="#f4f4f0", ec=INK2, lw=0.8))
rng = np.random.default_rng(7)
pts = rng.uniform([sx + 0.1, sy + 0.1], [sx + sw - 0.1, sy + sw - 0.1],
                  size=(15, 2))
for i, (px, py) in enumerate(pts):
    ax.add_patch(Circle((px, py), 0.035 if i % 4 else 0.045,
                        fc=C["aqua"] if i % 4 == 0 else "#bdbcb4",
                        ec="none"))
tgt = pts[4]
ax.add_patch(Circle(tgt, 0.085, fc="none", ec=C["red"], lw=1.1))
ax.text(sx + sw / 2, sy - 0.10, "sample: candidate sites",
        ha="center", va="top", fontsize=FS_S, color=INK2)

# objective (trapezoid, optical axis horizontal)
ox = 1.62
ax.add_patch(Polygon([[ox, sy + 0.40], [ox, sy + 0.75],
                      [ox + 0.22, sy + 0.86], [ox + 0.22, sy + 0.29]],
                     closed=True, fc="#e6e5df", ec=INK2, lw=0.8))
ax.text(ox + 0.11, sy + 0.92, "objective", ha="center", va="bottom",
        fontsize=FS_S, color=INK2)
yax = sy + 0.575                                   # optical axis
ax.plot([tgt[0] + 0.09, ox], [tgt[1], yax], color=C["red"], lw=0.7,
        ls=(0, (2, 1.5)))

# dichroic mirror and excitation laser (from below)
dx = 2.25
ax.plot([dx - 0.12, dx + 0.12], [yax - 0.12, yax + 0.12], color=INK,
        lw=1.6)
ax.text(dx + 0.02, yax + 0.20, "DM", ha="center", fontsize=FS_S,
        color=INK2)
box(dx - 0.33, sy - 0.02, 0.66, 0.28, "CW laser")
arrow((dx, sy + 0.26), (dx, yax - 0.02), color=C["blue"], lw=1.1)
arrow((dx - 0.02, yax), (ox + 0.24, yax), color=C["blue"], lw=1.1)

# fluorescence to the 50:50 beamsplitter
bx = 2.95
arrow((dx + 0.05, yax + 0.035), (bx - 0.12, yax + 0.035),
      color=C["red"], lw=1.1)
ax.add_patch(Rectangle((bx - 0.12, yax - 0.12), 0.24, 0.24, fc="#eef3fb",
                       ec=INK2, lw=0.8))
ax.plot([bx - 0.12, bx + 0.12], [yax - 0.12, yax + 0.12], color=INK2,
        lw=0.8)
ax.text(bx, yax - 0.20, "50:50 BS", ha="center", va="top",
        fontsize=FS_S, color=INK2)


def spad(x, y, rot):
    ax.add_patch(Wedge((x, y), 0.13, rot - 90, rot + 90, fc="#3b3a37",
                       ec=INK, lw=0.6))


# SPAD A (transmitted) and SPAD B (reflected)
arrow((bx + 0.12, yax), (bx + 0.50, yax), color=C["red"], lw=1.0)
spad(bx + 0.52, yax, 180)
ax.text(bx + 0.57, yax - 0.20, "SPAD A", ha="center", va="top",
        fontsize=FS_S, color=INK2)
arrow((bx, yax + 0.12), (bx, yax + 0.40), color=C["red"], lw=1.0)
spad(bx, yax + 0.42, 270)
ax.text(bx - 0.20, yax + 0.47, "SPAD B", ha="right", va="center",
        fontsize=FS_S, color=INK2)

# time tagger
tx, tw = 4.15, 0.78
box(tx, yax - 0.30, tw, 0.60, "time\ntagger")
arrow((bx + 0.66, yax), (tx - 0.02, yax), lw=0.9)
ax.add_patch(FancyArrowPatch((bx + 0.14, yax + 0.47),
                             (tx + tw / 2, yax + 0.32), arrowstyle="-|>",
                             color=INK2, lw=0.9, mutation_scale=9,
                             connectionstyle="angle,angleA=0,angleB=90,"
                             "rad=0.0", shrinkA=1, shrinkB=1))

# coincidence histogram accumulated slice by slice (schematic)
hx, hy, hw, hh = 5.78, sy + 0.12, 1.10, 0.92
ah = fig.add_axes([hx / W, hy / H, hw / W, hh / H])
tau = np.linspace(-60, 60, 61)
g2 = 1 - 0.9 * np.exp(-np.abs(tau) / 12) + 0.35 * np.exp(-np.abs(tau) / 250)
r = np.random.default_rng(4)
for n, col, lab in ((8, "#9ec5f4", "0.1 s"), (80, C["blue"], "1 s")):
    h = r.poisson(n * g2)
    ah.step(tau, h / n, where="mid", color=col, lw=0.9, label=lab)
ah.set_xlim(-60, 60)
ah.set_ylim(0, 2.6)
ah.set_xticks([-50, 0, 50])
ah.set_yticks([0, 1, 2])
ah.set_xlabel(r"delay $\tau$ (ns)", fontsize=FS_S, labelpad=1)
ah.set_ylabel(r"$g^{(2)}(\tau)$", fontsize=FS_S, labelpad=1)
ah.tick_params(labelsize=6.3, length=2, pad=1)
ah.grid(False)
despine(ah)
ah.legend(fontsize=6.0, frameon=False, loc="upper left",
          handlelength=1.0, borderaxespad=0.1, labelspacing=0.2,
          ncol=2, columnspacing=0.8)
arrow((tx + tw + 0.02, yax), (hx - 0.36, yax), lw=0.9)

# decision loop (second row, right to left)
ry, rh = 2.24, 0.62
box(5.55, ry, 1.35, rh, "early-decision estimator",
    sub="P(single emitter), updated\nduring the exposure",
    ec=C["aqua"], fc="#eef8f2")
box(3.85, ry, 1.35, rh, "exposure controller",
    sub="expose longer, certify,\nor reject", ec=C["violet"],
    fc="#f1effa")
arrow((hx + hw / 2 - 0.10, sy - 0.20), (hx + hw / 2 - 0.10, ry + rh + 0.02),
      lw=0.9)
arrow((5.53, ry + rh / 2), (5.22, ry + rh / 2), lw=0.9)
ax.add_patch(FancyArrowPatch((3.83, ry + rh / 2), (sx + sw / 2, sy - 0.30),
             arrowstyle="-|>", color=C["violet"], lw=1.1,
             mutation_scale=9, shrinkA=1, shrinkB=1,
             connectionstyle="angle,angleA=180,angleB=-90,rad=0.0"))
ax.text(2.30, ry + rh / 2 + 0.06, "next site, or continue exposing "
        "this site", ha="center", va="bottom", fontsize=FS_S,
        color=C["violet"])

# ======================================================== panel (b)
ax.text(0.05, 1.93, "(b)", fontsize=9.5, fontweight="bold", va="top")
tr = json.load(open(_ROOT + "/results/fig1_traces.json"))
ab = fig.add_axes([0.60 / W, 0.42 / H, 2.55 / W, 1.30 / H])
t_ms = (np.arange(32) + 1) * tr["slice_ms"]
for name, col, lab in (("single", C["aqua"],
                        r"single emitter, $g^{(2)}(0)=%.2f$"),
                       ("multi", C["red"],
                        r"three emitters, $g^{(2)}(0)=%.2f$")):
    P = np.array(tr[name]["p_pure"])
    for k, row in enumerate(P):
        ab.plot(t_ms, row, color=col, lw=0.9, alpha=0.85,
                label=(lab % tr[name]["g2_0"]) if k == 0 else None)
for yv in (0.75, 0.25):
    ab.axhline(yv, color=INK2, lw=0.6, ls=(0, (3, 2)))
ab.text(1005, 0.78, "certify", ha="right", va="bottom", fontsize=6.3,
        color=INK2)
ab.text(1005, 0.22, "reject", ha="right", va="top", fontsize=6.3,
        color=INK2)
ab.set_xlim(0, 1010)
ab.set_ylim(-0.03, 1.03)
ab.set_xlabel("exposure time (ms)", fontsize=FS_S, labelpad=1)
ab.set_ylabel("estimated P(single)", fontsize=FS_S, labelpad=1)
ab.tick_params(labelsize=6.3, length=2, pad=1)
ab.grid(False)
despine(ab)
ab.legend(fontsize=6.0, frameon=False, loc="lower left",
          bbox_to_anchor=(0.0, 1.0), ncol=1, borderaxespad=0.1,
          handlelength=1.2, labelspacing=0.2)

# ======================================================== panel (c)
ax.text(3.55, 1.93, "(c)", fontsize=9.5, fontweight="bold", va="top")


def levels(x0, y0, col, label):
    """three-level scheme: ground g, excited e, shelving s."""
    ax.plot([x0, x0 + 0.26], [y0, y0], color=INK, lw=1.0)             # g
    ax.plot([x0, x0 + 0.26], [y0 + 0.46, y0 + 0.46], color=INK, lw=1.0)
    ax.plot([x0 + 0.32, x0 + 0.52], [y0 + 0.28, y0 + 0.28], color=INK,
            lw=1.0)                                                   # s
    arrow((x0 + 0.07, y0 + 0.01), (x0 + 0.07, y0 + 0.45), color=col,
          lw=0.9)
    arrow((x0 + 0.17, y0 + 0.45), (x0 + 0.17, y0 + 0.01), color=col,
          lw=0.9, ls=(0, (2, 1.2)))
    arrow((x0 + 0.24, y0 + 0.45), (x0 + 0.38, y0 + 0.29), color=MUTED,
          lw=0.7)
    arrow((x0 + 0.42, y0 + 0.27), (x0 + 0.24, y0 + 0.02), color=MUTED,
          lw=0.7)
    ax.text(x0 + 0.26, y0 - 0.09, label, ha="center", va="top",
            fontsize=FS_S, color=INK)


levels(3.62, 1.02, C["blue"], "three-level emitter")
box(4.42, 1.02, 1.00, 0.50, "photon-counting\nsimulator")
arrow((4.18, 1.27), (4.40, 1.27), lw=0.9)
box(5.72, 1.02, 1.18, 0.50, "training data for\nestimator, controller",
    ec=MUTED)
arrow((5.44, 1.27), (5.70, 1.27), lw=0.9)

# level-structure graphs for transfer between emitter classes
for i, (name, col) in enumerate((("NV", C["blue"]), ("hBN", C["aqua"]),
                                 ("GaN", C["yellow"]), ("SiV", C["red"]))):
    gx, gy = 3.72 + i * 0.36, 0.42
    for (x1, y1), (x2, y2) in (((0, 0), (0, 0.28)), ((0, 0.28), (0.2, 0.16)),
                               ((0.2, 0.16), (0, 0))):
        ax.plot([gx + x1, gx + x2], [gy + y1, gy + y2], color=col, lw=0.9)
    for px, py in ((0, 0), (0, 0.28), (0.2, 0.16)):
        ax.add_patch(Circle((gx + px, gy + py), 0.035, fc="white", ec=col,
                            lw=0.9))
    ax.text(gx + 0.07, gy - 0.07, name, ha="center", va="top",
            fontsize=6.3, color=INK)
box(5.26, 0.34, 0.86, 0.46, "graph encoder")
arrow((5.18, 0.57), (5.24, 0.57), lw=0.9)
ax.text(6.18, 0.57, "conditions the\nestimator on the\nemitter class",
        ha="left", va="center", fontsize=FS_S, color=INK2,
        linespacing=1.1)
ax.text(4.35, 0.19, "level-structure graphs", ha="center", va="top",
        fontsize=FS_S, color=INK2)

fig.savefig(_ROOT + "/figures/fig1_concept.pdf")
fig.savefig(_ROOT + "/figures/fig1_concept.png", dpi=300)
print("fig1 written")
