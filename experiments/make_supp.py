"""Generate paper/supp_tables.tex: all supplementary tables, straight
from the experiment JSONs and the platform priors."""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, sys
import numpy as np

sys.path.insert(0, _ROOT)
from sparq.physics import PLATFORMS

R = _ROOT + "/results"
P = _ROOT + "/paper"


def load(name):
    with open(f"{R}/{name}") as f:
        return json.load(f)


parts = []

# ---------------------------------------------------------------- S1 twin
e1 = load("exp1_validation.json")
rows = []
for name, d in e1["mc_vs_exact"].items():
    e = d["eff"]
    rows.append(f"{name} & {d['T_s']:.1f} & {d['n_coinc']:,} & "
                f"{e['tau1']:.2f} & {e['tau2']:.0f} & {e['a']:.2f} & "
                f"{d['chi2_red']:.3f} & {100*d['nrmse']:.1f}\\% & "
                f"{100*d['mad']:.1f}\\% \\\\")
parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s1}Photon-stream simulator compared with the
numerically exact master-equation $g^{(2)}(\tau)$ (no free parameters). Effective
$(\tau_1, \tau_2, a)$ from the eigen-decomposition of the simulated
rates; $\chi^2_\nu$ over the 121 delay bins with Poisson errors.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccccccc}
Regime & $T$ (s) & Coinc. & $\tau_1$ (ns) & $\tau_2$ (ns) & $a$ &
$\chi^2_\nu$ & NRMSE & MAD \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")

# ---------------------------------------------------------------- S2 priors
rows = []
for k, p in PLATFORMS.items():
    rows.append(
        f"{k} & {p.tau1_rng[0]}--{p.tau1_rng[1]} & "
        f"{p.tau2_rng[0]}--{p.tau2_rng[1]} & "
        f"{p.a_rng[0]}--{p.a_rng[1]} & "
        f"{p.rate_rng[0]}--{p.rate_rng[1]} & "
        f"{p.rho_rng[0]}--{p.rho_rng[1]} & {p.blink_p:.2f} \\\\")
parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s2}Platform parameter priors (ranges anchored to the
published photophysics cited in the main text). $\rho$ is sampled from a
70/30 bimodal mixture over its range (localized-emitter vs.
high-background spots); multiplicity $N \in \{1,2,3,4\}$ with
probabilities $(0.42, 0.30, 0.18, 0.10)$.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccccc}
Platform & $\tau_1$ (ns) & $\tau_2$ (ns) & $a$ & rate (kcps) & $\rho$ &
$P(\mathrm{blink})$ \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")

# ---------------------------------------------------------------- S3 sweeps
e2 = load("exp2_estimators.json")
try:      # the activity-regularized SNN supersedes the exp2 SNN
    e2b = load("exp2b_snn.json")
    e2["results"]["snn_pitl"] = e2b["snn_sparse"]
    e2["anytime"] = e2b["anytime"]
    e2["energy"] = e2b["energy"]
except FileNotFoundError:
    pass
T = e2["T_grid"]
names = {"fit": "LM fit", "cnn_clean": "CNN, long exp.",
         "cnn_pitl": "CNN, variable exp.", "snn_pitl": "SNN, variable exp."}
rows = []
for m, label in names.items():
    accs = " & ".join(f"${a[0]*100:.1f} \\pm {a[1]*100:.1f}$"
                      for a in e2["results"][m]["acc"])
    rows.append(f"{label} & {accs} \\\\")
bay = " & ".join(f"${e2['bayes_acc'][str(t)]*100:.1f}$" for t in T)
rows.append(f"Bayes ref.\\ ($M{{=}}6000$) & {bay} \\\\")
head = " & ".join(f"{t}\\,s" for t in T)
parts.append(r"""
\begin{table*}[!htb]
\caption{\label{tab:s3}Balanced classification accuracy (\%, mean $\pm$
1~s.d.\ over five noise seeds; boundary band excluded) versus acquisition
time on the NV prior (long exp.: trained on 30-s acquisitions only;
variable exp.: trained on $T \sim \log\mathcal{U}[0.03, 30]$\,s). The
last row is the Monte-Carlo Bayes reference under the assumed
simulation prior with $M = 6000$ reference sites (one reference
sample); its dependence on $M$ is given in Table~\ref{tab:bconv}.}
\small
\begin{ruledtabular}
\begin{tabular}{l""" + "c" * len(T) + r"""}
Estimator & """ + head + r""" \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
""")

rows = []
for m, label in names.items():
    maes = " & ".join(f"${a[0]:.3f} \\pm {a[1]:.3f}$"
                      for a in e2["results"][m]["mae"])
    rows.append(f"{label} & {maes} \\\\")
parts.append(r"""
\begin{table*}[!htb]
\caption{\label{tab:s3b}$g^{(2)}(0)$ regression MAE versus acquisition
time (mean $\pm$ 1~s.d.\ over five seeds; all sites including the
boundary band).}
\footnotesize
\begin{ruledtabular}
\begin{tabular}{l""" + "c" * len(T) + r"""}
Estimator & """ + head + r""" \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
""")

# ------------------------------------------------------------- S4 anytime/energy
rows = []
for th, d in e2["anytime"].items():
    rows.append(f"$\\theta = {th}$ & {d['median_ms']:.0f} & "
                f"{d['mean_ms']:.0f} & {100*d['acc']:.1f} & "
                f"{100*d['frac_full']:.1f}\\% \\\\")
parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s4}Early decisions of the SNN during a 1-s exposure:
median and mean time at which the output first crosses the
commitment gate $\theta$ (on $|p_1 - p_0|$), balanced accuracy of the
committed decisions, and the fraction of sites that never cross the
gate within the exposure.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccc}
Gate & Median (ms) & Mean (ms) & Bal. acc. (\%) & Never-commit \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")

rows = []
for tt, d in e2["energy"].items():
    r32 = d['e_snn_nJ'] / d['e_cnn_fp32_nJ']
    r8 = d['e_snn_nJ'] / d['e_cnn_int8_nJ']
    rows.append(f"{tt} & {d['synops_mean']/1e3:.1f} & {d['e_snn_nJ']:.0f} & "
                f"{d['e_cnn_fp32_nJ']:.0f} & {d['e_cnn_int8_nJ']:.1f} & "
                f"{r32:.2f} & {r8:.0f} \\\\")
parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s5}Operation-count energy estimate per decision:
measured SNN synaptic operations (synops) at 23.6\,pJ each (minimum
value reported for Loihi) and CNN multiply--accumulates at 4.6\,pJ
(32-bit floating point) or 0.23\,pJ (8-bit integer) each (45-nm
CMOS). The last two columns give the SNN energy divided by the CNN
energy; values above 1 mean that the SNN estimate is higher.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccccc}
$T$ (s) & synops ($10^3$) & SNN (nJ) & CNN FP32 (nJ) &
CNN INT8 (nJ) & SNN/CNN FP32 & SNN/CNN INT8 \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")

# ---------------------------------------------------------------- S6 adjoint
e3 = load("exp3_adjoint.json")
rows = []
for tt, d in e3["evals"].items():
    rows.append(f"{tt} & {100*d['default']['acc']:.1f} & "
                f"{100*d['adjoint']['acc']:.1f} & "
                f"{d['default']['mae']:.3f} & {d['adjoint']['mae']:.3f} \\\\")
parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s6}Joint optimization of the measurement setting
with the CNN (one training run per setting; 2000 evaluation sites):
balanced accuracy (\%) and $g^{(2)}(0)$ MAE at the default setting
$(s{=}1, \tau_{\max}{=}60.5\,\mathrm{ns})$ versus the optimized setting
$(s^{*}{=}""" + f"{e3['s_star']:.2f}" + r""",
\tau_{\max}^{*}{=}""" + f"{e3['tau_max_star']:.1f}" + r"""\,\mathrm{ns})$.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccc}
$T$ (s) & Acc.\ default & Acc.\ optimized & MAE default & MAE optimized \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")

# ---------------------------------------------------------------- S7 gan
try:
    e4 = load("exp4_gan.json")
    rows = []
    for s in e4["series"]:
        rows.append(f"{s['name'].replace('_', ' ')[:34]} & "
                    f"{s['g2_ref']:.3f} & {s['T_tot']:.0f} & "
                    f"{s['n_windows']} & "
                    f"{'held out' if s['held_out'] else 'train'} \\\\")
    parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s7}The eight experimental quantum-dot HBT
measurement series (sps-quality, FI-SEQUR demonstrator), analyzed as
nine records: the 2.5-$\mu$W series was recorded in two sessions
(day~1/day~2), kept separate here. $g^{(2)}(0)$ references from the
peak-area analysis of each full accumulation.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccc}
Series & $g^{(2)}(0)$ ref. & $T$ (s) & 30-s windows & Split \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")
    rows = []
    lbl = {"fit": "Peak-area analysis", "sim": "Simulation only",
           "dr": "Simulation + domain rand.",
           "gan": "Simulation + WGAN-GP refiner"}
    for m in ("fit", "sim", "dr", "gan"):
        d = e4["results"][m]
        tr = d.get("mae_train_series")
        rows.append(f"{lbl[m]} & {d['mae_held_out']:.3f} & "
                    f"{tr:.3f} \\\\" if tr is not None else
                    f"{lbl[m]} & {d['mae_held_out']:.3f} & n/a \\\\")
    parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s8}Mean absolute error of $g^{(2)}(0)$ estimated
from 30-s windows of the experimental records, relative to the
peak-area value of each record's full accumulation, for the three
held-out records and the six records used to train the refiner.}
\small
\begin{ruledtabular}
\begin{tabular}{lcc}
Method & Held-out series & Train series \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")
except FileNotFoundError:
    pass

# ---------------------------------------------------------------- S9 RL
try:
    e5 = load("exp5_rl.json")
    rows = []
    def row(name, s):
        return (f"{name} & {s['time_s'][0]:.0f} $\\pm$ {s['time_s'][1]:.0f} & "
                f"{s['precision'][0]:.3f} & {s['recall'][0]:.3f} & "
                f"{s['good_per_min'][0]:.2f} \\\\")
    def pretty(k):
        kind, par = k.split("_")
        if kind == "raster":
            return f"Fixed-dwell raster, {par} s"
        return f"Threshold controller, $|2p-1| \\geq {par}$"
    for k in sorted(e5["baselines"]):
        rows.append(row(pretty(k), e5["baselines"][k]))
    for k in ("per", "uniform"):
        for i, s in enumerate(e5["final"][k]):
            rows.append(row(f"RL agent, {'prioritized' if k=='per' else 'uniform'}"
                            f" replay, seed {i}", s))
    try:
        with open(f"{R}/exp5b_oracle.json") as f:
            rows.append(row("Oracle stopping (bound)", json.load(f)))
    except FileNotFoundError:
        pass
    parts.append(r"""
\begin{table}[!htb]
\caption{\label{tab:s9}Closed-loop triage on 30 held-out simulated
48-site fields: measurement time per field (mean $\pm$ s.d.\ over
fields), precision and recall of the certified good emitters, and
certified good emitters per minute. The threshold controller exposes
a site in 0.25-s steps until $|2p - 1|$ reaches the stated value or
the dwell reaches 5\,s; the oracle stops each site as soon as the
estimate agrees with the ground truth.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccc}
Policy & Time (s) & Precision & Recall & Good/min \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")
except FileNotFoundError:
    pass

# ---------------------------------------------------------------- S10 graph
try:
    e6 = load("exp6b_graph.json")
    plats = ["NV", "hBN", "GaN", "SiV"]
    Ts = [0.3, 3.0]
    rows = []
    lbl = {"uncond_syn": "Uncond.\ (synth.)",
           "graph_syn": "Graph-cond.\ (synth.)",
           "oracle_real": "Reference (four priors)"}
    for m in ("uncond_syn", "graph_syn", "oracle_real"):
        cells = []
        for p in plats:
            for t in Ts:
                a = e6["results"][m][f"{p}@{t}"]["acc"]
                cells.append(f"${100*a[0]:.1f}{{\\pm}}{100*a[1]:.1f}$")
        rows.append(lbl[m] + " & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"{p} {t}s" for p in plats for t in Ts)
    parts.append(r"""
\begin{table*}[!htb]
\caption{\label{tab:s10}Cross-platform transfer: balanced accuracy (\%)
on simulated data from each platform prior at two acquisition times
(mean $\pm$ 1~s.d.\ over three seeds). The four platform priors are
not seen in training by the two models trained on synthetic classes.}
\footnotesize
\begin{ruledtabular}
\begin{tabular}{l""" + "c" * len(plats) * len(Ts) + r"""}
Model & """ + head + r""" \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
""")
except FileNotFoundError:
    pass

parts.append(r'''
\begin{table*}[!htb]
\caption{\label{tab:s11}Reported results of related machine-learning and
automated approaches to single-photon-emitter characterization, as
stated by the respective authors (references as in the main text),
and the corresponding results of this work. Tasks, data, and baselines
differ between entries, so the reported numbers are not directly
comparable.}
\footnotesize
\begin{center}
\begin{tabular}{p{0.16\textwidth}p{0.22\textwidth}p{0.24\textwidth}p{0.30\textwidth}}
\hline\hline
Work & Data & Method & Reported result \\
\hline
Kudyshev \emph{et al.}, 2020 & Sparse HBT autocorrelation histograms &
 Supervised classification (single or not single), including a 1D
 convolutional network & 95\% accuracy within an integration time of
 less than 1\,s; roughly 100-fold speedup over Levenberg--Marquardt
 fitting \\[2pt]
Kudyshev \emph{et al.}, 2023 & Photon-correlation histograms in
 antibunching (quantum super-resolution) microscopy &
 Convolutional-network regression & 12-fold speed-up compared with
 fitting-based autocorrelation measurements \\[2pt]
Narun \emph{et al.}, 2022 & Photoluminescence images & Image analysis
 and regression fitting & Automated identification and classification
 of emitters in nanodiamond arrays and hBN flakes \\[2pt]
Xu \emph{et al.}, 2024 & Higher-order photon-correlation data &
 Two-dimensional convolutional-network classification of photon
 states up to $|3\rangle$ & 94\% overall accuracy; 90\% with 800
 co-detection events \\[2pt]
Kedziora \emph{et al.}, 2023 & Eight experimental HBT datasets of one
 InGaAs/GaAs quantum dot & Bootstrap data augmentation to quantify the
 uncertainty of early estimates & Significant uncertainty from Poisson
 variability; least-squares fitting comparable to Poisson likelihood
 \\[2pt]
Kedziora \emph{et al.}, 2025 & Same quantum dot, eight excitation
 contexts & Transfer learning with linear and ensemble regressors
 (trained on seven contexts, tested on the eighth) & Regressors can
 outperform least-squares fitting within trained contexts; success of
 transfer less assured \\[2pt]
This work & Simulated CW HBT histograms and time-sliced coincidence
 records; experimental data of Kedziora \emph{et al.} for testing &
 Spiking and convolutional networks; RL exposure control; graph
 conditioning on the emitter class & In simulation: $\speedupSnn\times$
 less acquisition time than a multi-start LM fit at \targetAcc\%
 balanced accuracy, and $\rlSpeedup\times$ faster field screening than
 fixed dwell. On experimental 30-s windows: $\xFitOverSim\times$ lower
 $g^{(2)}(0)$ error than peak-area analysis \\
\hline\hline
\end{tabular}
\end{center}
\end{table*}
''')


out = "\n".join(parts)
import re as _re
# add explicit outer rules inside each ruledtabular (ACS article route:
# ruledtabular is a plain centering shim, so the rules must be in the
# tabular itself)
out = _re.sub(r"\\begin\{ruledtabular\}\s*\n(\\begin\{tabular\}\{[^}]*\})",
              lambda m: "\\begin{ruledtabular}\n" + m.group(1) + "\n\\hline\\hline", out)
out = _re.sub(r"\\end\{tabular\}\s*\n\\end\{ruledtabular\}",
              lambda m: "\\hline\\hline\n\\end{tabular}\n\\end{ruledtabular}", out)
with open(f"{P}/supp_tables.tex", "w") as f:
    f.write("% AUTO-GENERATED by experiments/make_supp.py\n")
    f.write(out)
print("wrote supp_tables.tex with", len(parts), "tables")

# Also split into one file per labeled table so supplementary.tex can
# input each table next to the text that cites it (ACS: tables numbered
# S1...Sn in order of appearance).
blocks = _re.split(r"(?=\\begin\{table\*?\})", out)
import os
for b in blocks:
    m = _re.search(r"\\label\{tab:([a-z0-9]+)\}", b)
    if m and b.strip().startswith("\\begin{table"):
        with open(f"{P}/supp_tab_{m.group(1)}.tex", "w") as f:
            f.write("% AUTO-GENERATED by experiments/make_supp.py\n" + b)
    elif "tablesotasupp" in b:
        with open(f"{P}/supp_tab_sota.tex", "w") as f:
            f.write("% AUTO-GENERATED by experiments/make_supp.py\n" + b)
print("wrote per-table files")
