"""House figure style for the SPARQ manuscript (validated palette,
publication-grade matplotlib defaults)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

# validated categorical palette (fixed order, light mode)
C = dict(blue="#2a78d6", aqua="#1baf7a", yellow="#eda100", green="#008300",
         violet="#4a3aa7", red="#e34948", magenta="#e87ba4", orange="#eb6834")
SERIES = [C["blue"], C["aqua"], C["yellow"], C["green"], C["violet"],
          C["red"], C["magenta"], C["orange"]]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#ffffff"

# sequential blue ramp (light -> dark)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
       "#0d366b"]

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 7.8,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 7.8,
    "axes.titlesize": 8.4,
    "axes.titleweight": "bold",
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "axes.prop_cycle": cycler(color=SERIES),
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 6.8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "none",
    "legend.facecolor": "white",
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def panel_label(ax, s, dx=-0.14, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left", color=INK)


def legend_above(ax, ncol=3, fontsize=6.8, y=1.02, handlelength=1.3):
    """Legend in a row above the axes — never overlaps data."""
    return ax.legend(loc="lower left", bbox_to_anchor=(0.0, y),
                     ncol=ncol, fontsize=fontsize, frameon=False,
                     handlelength=handlelength, borderpad=0.1,
                     columnspacing=0.9, handletextpad=0.5,
                     borderaxespad=0.0)


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
