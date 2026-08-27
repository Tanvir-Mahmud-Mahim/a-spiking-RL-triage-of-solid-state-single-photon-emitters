"""ACS Photonics Table-of-Contents graphic: exactly 3.25 x 1.75 in."""
import sys
sys.path.insert(0, "/home/claude/sparq/figures")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from style import C, INK, INK2, MUTED

fig = plt.figure(figsize=(3.25, 1.75))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 3.25)
ax.set_ylim(0, 1.75)
ax.axis("off")


def box(x, y, w, h, ec, fc, lw=1.2, r=0.06):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={r}",
                 ec=ec, fc=fc, lw=lw))


def arrow(x0, y0, x1, y1, color=INK, lw=1.4, rad=0.0):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=1, shrinkB=1))


# ---- left: emitter field ------------------------------------------------
box(0.10, 0.62, 0.72, 0.80, INK, "#f4f4f0")
rngf = np.random.default_rng(5)
pts = rngf.uniform([0.17, 0.70], [0.75, 1.34], size=(14, 2))
good = [1, 5, 9, 12]
for i, (px, py) in enumerate(pts):
    ax.scatter([px], [py], s=26 if i in good else 16,
               color=C["aqua"] if i in good else "#b9b8af", zorder=3)
ax.scatter([pts[9, 0]], [pts[9, 1]], s=95, facecolor="none",
           edgecolor=C["red"], lw=1.3, zorder=4)
ax.text(0.46, 0.47, "emitter field", ha="center", fontsize=8.0,
        color=INK, fontweight="bold")

# ---- middle: spiking estimator -----------------------------------------
box(1.22, 0.62, 0.86, 0.80, C["aqua"], "#eef8f2")
# spike raster icon
rs = np.random.default_rng(3)
for row, yy in enumerate((1.24, 1.10, 0.96, 0.82)):
    for xx in np.sort(rs.uniform(1.32, 1.96, 4 + (row % 2))):
        ax.plot([xx, xx], [yy - 0.045, yy + 0.045], color=C["aqua"],
                lw=1.3, solid_capstyle="round")
ax.text(1.65, 0.70, "decides in ~0.3 s", ha="center", fontsize=6.8,
        color=INK2)
ax.text(1.65, 0.47, "spiking network", ha="center", fontsize=8.0,
        color=INK, fontweight="bold")

# ---- right: decision ----------------------------------------------------
box(2.48, 0.62, 0.66, 0.80, C["violet"], "#f1effa")
ax.text(2.81, 1.22, "certify", ha="center", fontsize=8.0,
        color="#0b7a44", fontweight="bold")
ax.text(2.81, 1.01, "reject", ha="center", fontsize=8.0,
        color=C["red"], fontweight="bold")
ax.text(2.81, 0.76, "6$\\times$ faster", ha="center", fontsize=8.2,
        color=INK, fontweight="bold")

# ---- arrows -------------------------------------------------------------
arrow(0.84, 1.02, 1.20, 1.02)
ax.text(1.02, 1.12, "photons", ha="center", fontsize=6.8, color=INK2)
arrow(2.10, 1.02, 2.46, 1.02)
# closed-loop return
arrow(2.83, 0.60, 0.50, 0.615, color=C["violet"], lw=1.5, rad=-0.22)
ax.text(1.63, 0.075, "closed loop: adaptive exposure per site",
        ha="center", fontsize=7.4, color=C["violet"])

fig.savefig("/home/claude/sparq/figures/toc_graphic.pdf")
fig.savefig("/home/claude/sparq/figures/toc_graphic.png", dpi=400)
print("toc done: 3.25 x 1.75 in")
