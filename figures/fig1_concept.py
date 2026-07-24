"""Figure — SPARQ architecture (three panels, publication layout)."""
import sys
sys.path.insert(0, "/home/claude/sparq/figures")
sys.path.insert(0, "/home/claude/sparq")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Rectangle)
from style import C, INK, INK2, MUTED, BASE, panel_label

rng = np.random.default_rng(3)

FS_MAIN = 7.4      # box titles
FS_SUB = 5.9       # box subtitles
FS_NOTE = 6.4      # free annotations


def box(ax, x, y, w, h, label, fc="#ffffff", ec=INK2, fs=FS_MAIN, lw=1.0,
        tc=None, sub=None, subfs=FS_SUB, r=0.018):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.008,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(b)
    if sub:
        ax.text(x + w / 2, y + 0.66 * h, label, ha="center", va="center",
                fontsize=fs, color=tc or INK, fontweight="bold")
        ax.text(x + w / 2, y + 0.30 * h, sub, ha="center", va="center",
                fontsize=subfs, color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc or INK, fontweight="bold")
    return b


def arrow(ax, p0, p1, color=INK2, lw=1.2, style="-|>", con="arc3,rad=0.0",
          ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                        connectionstyle=con, linestyle=ls,
                        mutation_scale=10, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


fig = plt.figure(figsize=(7.05, 5.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0],
                      height_ratios=[1.05, 1.0],
                      left=0.005, right=0.995, top=0.975, bottom=0.015,
                      wspace=0.05, hspace=0.14)

# ======================================================= (a) closed loop
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 0.66)
ax.axis("off")
panel_label(ax, "(a)", dx=0.001, dy=1.0)

# confocal field
fx, fy, fw, fh = 0.035, 0.195, 0.285, 0.375
ax.add_patch(Rectangle((fx, fy), fw, fh, fc="#f4f6fa", ec=BASE, lw=0.9))
pts = rng.uniform([fx + 0.022, fy + 0.032],
                  [fx + fw - 0.022, fy + fh - 0.032], (24, 2))
purity = rng.random(24)
for (px, py), q in zip(pts, purity):
    good = q > 0.72
    ax.add_patch(Circle((px, py), 0.0080 + 0.006 * q,
                        fc=C["aqua"] if good else "#c9cdd6",
                        ec="none", alpha=0.95 if good else 0.75))
ax.add_patch(Circle(tuple(pts[np.argmax(purity)]), 0.030, fc="none",
                    ec=C["red"], lw=1.3))
ax.text(fx + fw / 2, fy - 0.038, "confocal emitter field",
        ha="center", va="top", fontsize=FS_NOTE, color=INK, fontweight="bold")
ax.text(fx + fw / 2, fy - 0.082, "NV / hBN / GaN / SiV",
        ha="center", va="top", fontsize=FS_SUB, color=INK2)
ax.text(fx + fw / 2, fy + fh + 0.022, "laser scan", ha="center",
        fontsize=FS_SUB, color=MUTED)

# HBT
bx, by, bw, bh = 0.405, 0.285, 0.215, 0.20
box(ax, bx, by, bw, bh, "HBT detection",
    sub="50:50 BS, two SPADs,\ntime tagger")
arrow(ax, (fx + fw, 0.385), (bx, 0.385))

# event stream
exx, exy, exw, exh = 0.695, 0.305, 0.155, 0.155
ax.add_patch(Rectangle((exx, exy), exw, exh, fc="#ffffff", ec=BASE, lw=0.9))
et = rng.uniform(exx + 0.010, exx + exw - 0.010, 32)
ey = rng.uniform(exy + 0.022, exy + exh - 0.022, 32)
ax.scatter(et, ey, s=3.0, color=C["blue"], marker="|", linewidths=0.9)
ax.text(exx + exw / 2, exy - 0.038, "photon-pair\nevents", ha="center",
        va="top", fontsize=FS_SUB, color=INK2)
arrow(ax, (bx + bw, 0.385), (exx, 0.385))

# SNN
sx, sy_, sw, sh = 0.925, 0.275, 0.245, 0.22
box(ax, sx, sy_, sw, sh, "spiking front-end",
    sub="event-driven LIF network\nanytime posterior",
    fc="#eef7f2", ec=C["aqua"])
arrow(ax, (exx + exw, 0.385), (sx, 0.385))

# agent
gx, gy_, gw, gh = 1.255, 0.275, 0.235, 0.22
box(ax, gx, gy_, gw, gh, "SAC agent + PER",
    sub="dwell 0.25 / 1 / 4 s\nreject / certify / move",
    fc="#f0eefb", ec=C["violet"])
arrow(ax, (sx + sw, 0.385), (gx, 0.385))

# certified output (up-right)
ax.text(1.09, 0.625, "certified single-photon emitters",
        fontsize=FS_NOTE, color=C["green"], ha="center", fontweight="bold")
arrow(ax, (gx + gw / 2, gy_ + gh), (1.20, 0.605), color=C["green"], lw=1.2,
      con="arc3,rad=-0.15")

# actuation arc back to field, routed well below all blocks
arrow(ax, (gx + gw / 2, gy_), (fx + fw - 0.01, fy - 0.004),
      color=C["violet"], lw=1.4, con="arc3,rad=-0.22")
ax.text(0.88, 0.022, "actuation: per-site photon budget",
        fontsize=FS_NOTE, color=C["violet"], ha="center",
        bbox=dict(fc="white", ec="none", pad=0.8))

# ================================================ (b) twin as training env
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.62)
ax.axis("off")
panel_label(ax, "(b)", dx=0.001, dy=1.02)

bw2, bh2 = 0.285, 0.205
box(ax, 0.025, 0.355, bw2, bh2, "stochastic twin",
    sub="exact coincidence statistics\n(validated to theory)",
    fc="#eef3fb", ec=C["blue"])
box(ax, 0.025, 0.055, bw2, bh2, "real photon data",
    sub="quantum-dot HBT series\n(sps-quality, open)",
    fc="#fdf1ec", ec=C["orange"])
box(ax, 0.375, 0.055, 0.26, bh2, "WGAN-GP critic",
    sub="sim-to-real diagnostic",
    fc="#fdf1ec", ec=C["orange"])
box(ax, 0.375, 0.355, 0.26, bh2, "physics-in-the-\nloop training",
    fc="#eef7f2", ec=C["aqua"], fs=7.0)
box(ax, 0.715, 0.355, 0.265, bh2, "adjoint protocol",
    sub=r"$\nabla_{\theta}$ through the physics",
    fc="#eef3fb", ec=C["blue"])
box(ax, 0.715, 0.055, 0.265, bh2, "SAC + PER training",
    sub="rewards from twin truth",
    fc="#f0eefb", ec=C["violet"])

arrow(ax, (0.025 + bw2, 0.46), (0.375, 0.46))
arrow(ax, (0.025 + bw2, 0.155), (0.375, 0.155))
arrow(ax, (0.168, 0.355), (0.168, 0.26))
arrow(ax, (0.505, 0.26), (0.505, 0.355), color=C["orange"])
arrow(ax, (0.635, 0.46), (0.715, 0.46))
arrow(ax, (0.635, 0.42), (0.715, 0.19), con="arc3,rad=0.2")
ax.text(0.848, 0.595, r"$\theta^{*}$: power, window", fontsize=FS_SUB,
        color=C["blue"], ha="center")

# ==================================================== (c) graph transfer
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.62)
ax.axis("off")
panel_label(ax, "(c)", dx=0.001, dy=1.02)


def level_diagram(ax, x0, y0, name, col, t1):
    w = 0.155
    ax.plot([x0, x0 + w * 0.45], [y0, y0], color=INK, lw=1.3)
    ax.plot([x0, x0 + w * 0.45], [y0 + 0.135, y0 + 0.135], color=INK, lw=1.3)
    ax.plot([x0 + w * 0.62, x0 + w], [y0 + 0.052, y0 + 0.052], color=MUTED,
            lw=1.3)
    arrow(ax, (x0 + w * 0.12, y0 + 0.004), (x0 + w * 0.12, y0 + 0.130),
          color=col, lw=1.0)
    arrow(ax, (x0 + w * 0.33, y0 + 0.130), (x0 + w * 0.33, y0 + 0.004),
          color=col, lw=1.0)
    arrow(ax, (x0 + w * 0.50, y0 + 0.125), (x0 + w * 0.78, y0 + 0.060),
          color=MUTED, lw=0.9)
    ax.text(x0 + w / 2, y0 - 0.033, name, ha="center", fontsize=6.6,
            color=INK, fontweight="bold")
    ax.text(x0 + w / 2, y0 - 0.075, t1, ha="center", fontsize=5.2,
            color=INK2)


level_diagram(ax, 0.035, 0.455, "NV", C["blue"], r"$\tau_1$ 8–25 ns")
level_diagram(ax, 0.265, 0.455, "hBN", C["aqua"], r"$\tau_1$ 1.5–5 ns")
level_diagram(ax, 0.035, 0.175, "GaN", C["yellow"], r"$\tau_1$ 0.6–2 ns")
level_diagram(ax, 0.265, 0.175, "SiV", C["red"], r"$\tau_1$ 0.8–2 ns")
ax.text(0.245, 0.032, "level-structure template graphs\n(+ procedurally"
        " generated synthetic platforms)", fontsize=5.2, color=INK2,
        ha="center")

box(ax, 0.545, 0.28, 0.20, 0.20, "GNN\nencoder", fc="#fbf6ea",
    ec=C["yellow"], fs=6.8)
arrow(ax, (0.455, 0.52), (0.545, 0.42), con="arc3,rad=-0.15")
arrow(ax, (0.455, 0.25), (0.545, 0.34), con="arc3,rad=0.12")
box(ax, 0.795, 0.28, 0.185, 0.20, r"$z_{\rm platform}$",
    sub="conditions\nthe estimator", fc="#ffffff", ec=INK2, fs=6.8,
    subfs=5.2)
arrow(ax, (0.745, 0.38), (0.795, 0.38))
ax.text(0.765, 0.10, "zero-shot to platforms\nnever seen in training",
        fontsize=6.0, color=C["green"], ha="center", fontweight="bold")

fig.savefig("/home/claude/sparq/figures/fig1_concept.pdf")
fig.savefig("/home/claude/sparq/figures/fig1_concept.png")
print("fig1 done")
