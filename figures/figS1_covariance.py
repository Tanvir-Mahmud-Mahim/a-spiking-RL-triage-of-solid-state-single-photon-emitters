"""Supporting Figure S1 -- bin-to-bin correlation matrices of repeated
photon-stream-simulator histograms (121 delay bins, 1000 repetitions of
a 0.5-s acquisition each; diagonal omitted).  Panel (f) repeats (e)
after each histogram is rescaled to the mean total coincidence count."""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
sys.path.insert(0, _ROOT + "/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import INK, INK2

R = _ROOT + "/results"
PANELS = [
    (f"{R}/exp7_parts/A_noimp.npy", "(a)", "condition A,\nideal detectors"),
    (f"{R}/exp7_parts/A_imp.npy", "(b)", "condition A,\ndead time + afterpulsing"),
    (f"{R}/exp8_parts/A2_B_noimp.npy", "(c)", "condition B,\nideal detectors"),
    (f"{R}/exp8_parts/A2_B_imp.npy", "(d)", "condition B,\ndead time + afterpulsing"),
    (f"{R}/exp8_parts/A2_C_blink.npy", "(e)", "condition C (blinking),\nideal detectors"),
    (f"{R}/exp8_parts/A2_C_blink.npy", "(f)", "condition C, rescaled to\nthe mean total count"),
]
VMAX = 0.25
tau = np.linspace(-60, 60, 121)

fig = plt.figure(figsize=(6.2, 4.3))
w, h, gapx, left = 0.232, 0.335, 0.035, 0.085
rows_y = (0.545, 0.075)
axes = [fig.add_axes([left + (i % 3) * (w + gapx), rows_y[i // 3], w, h])
        for i in range(6)]
cax = fig.add_axes([left + 3 * (w + gapx) - 0.015, 0.075, 0.014, 0.805])
for i, (ax, (fn, lab, title)) in enumerate(zip(axes, PANELS)):
    H = np.load(fn)
    if lab == "(f)":
        s = H.sum(1)
        H = H / s[:, None] * s.mean()
    C = np.corrcoef(H.T)
    np.fill_diagonal(C, np.nan)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                   origin="lower", extent=[tau[0], tau[-1], tau[0], tau[-1]],
                   interpolation="nearest")
    ax.set_xticks([-50, 0, 50])
    ax.set_yticks([-50, 0, 50])
    ax.tick_params(labelsize=6.2, length=2, pad=1, colors="black")
    ax.set_xlabel(r"$\tau_i$ (ns)", fontsize=6.8, labelpad=1)
    if i % 3 == 0:
        ax.set_ylabel(r"$\tau_j$ (ns)", fontsize=6.8, labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(0.6)
    ax.set_title(title, fontsize=6.5, color=INK, pad=3, linespacing=1.1)
    ax.text(-0.02, 1.22, lab, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left")
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
cb.ax.tick_params(labelsize=6.0, length=2, pad=1, colors="black")
cb.set_label("correlation coefficient", fontsize=6.4, labelpad=2)
cb.outline.set_linewidth(0.5)
fig.savefig(_ROOT + "/figures/figS1_covariance.pdf")
fig.savefig(_ROOT + "/figures/figS1_covariance.png", dpi=300)
print("figS1 written")
