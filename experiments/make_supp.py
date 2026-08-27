"""Generate paper/supp_tables.tex — all supplementary tables, straight
from the experiment JSONs and the platform priors."""
import json, sys
import numpy as np

sys.path.insert(0, "/home/claude/sparq")
from sparq.physics import PLATFORMS

R = "/home/claude/sparq/results"
P = "/home/claude/sparq/paper"


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
\begin{table}[!htbp]
\caption{\label{tab:s1}Stream-twin validation against the numerically
exact master-equation $g^{(2)}(\tau)$ (no free parameters). Effective
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
\begin{table}[!htbp]
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
names = {"fit": "LM fit", "cnn_clean": "CNN (clean)",
         "cnn_pitl": "CNN (PITL)", "snn_pitl": "SNN (PITL)"}
rows = []
for m, label in names.items():
    accs = " & ".join(f"${a[0]*100:.1f} \\pm {a[1]*100:.1f}$"
                      for a in e2["results"][m]["acc"])
    rows.append(f"{label} & {accs} \\\\")
bay = " & ".join(f"${e2['bayes_acc'][str(t)]*100:.1f}$" for t in T)
rows.append(f"MC-Bayes reference & {bay} \\\\")
head = " & ".join(f"{t}\\,s" for t in T)
parts.append(r"""
\begin{table*}[!htbp]
\caption{\label{tab:s3}Balanced classification accuracy (\%, mean $\pm$
1~s.d.\ over five noise seeds; boundary band excluded) versus acquisition
time on the NV prior. The Monte-Carlo Bayes row is the
Bayes-optimal reference under the assumed simulation prior (single
seed; convergence with reference-sample size in Table~\ref{tab:bconv}).}
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
\begin{table*}[!htbp]
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
\begin{table}[!htbp]
\caption{\label{tab:s4}Anytime operation of the spiking readout at
$T{=}1$\,s: commitment latency and accuracy versus the confidence gate.}
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
    rows.append(f"{tt} & {d['synops_mean']/1e3:.1f} & {d['e_snn_nJ']:.1f} & "
                f"{d['e_cnn_fp32_nJ']/1e3:.2f} & {d['e_cnn_int8_nJ']/1e3:.2f} & "
                f"{d['adv_fp32']:.0f} & {d['adv_int8']:.0f} \\\\")
parts.append(r"""
\begin{table}[!htbp]
\caption{\label{tab:s5}Measured event-driven energy accounting per
decision (synops at 23.6\,pJ, Loihi; dense MACs at 4.6\,pJ FP32 /
1\,pJ INT8).}
\small
\begin{ruledtabular}
\begin{tabular}{lcccccc}
$T$ (s) & synops ($10^3$) & SNN (nJ) & CNN FP32 ($\mu$J) &
CNN INT8 ($\mu$J) & adv.\ FP32 & adv.\ INT8 \\
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
\begin{table}[!htbp]
\caption{\label{tab:s6}Adjoint protocol co-optimization: balanced
accuracy (\%) and $g^{(2)}(0)$ MAE at the default protocol
$(s{=}1, \tau_{\max}{=}60.5\,\mathrm{ns})$ versus the adjoint optimum
$(s^{*}{=}""" + f"{e3['s_star']:.2f}" + r""",
\tau_{\max}^{*}{=}""" + f"{e3['tau_max_star']:.1f}" + r"""\,\mathrm{ns})$.}
\small
\begin{ruledtabular}
\begin{tabular}{lcccc}
$T$ (s) & Acc.\ default & Acc.\ adjoint & MAE default & MAE adjoint \\
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
\begin{table}[!htbp]
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
    lbl = {"fit": "Peak-area analysis", "sim": "Twin only",
           "dr": "Twin + domain rand.", "gan": "Twin + WGAN-GP"}
    for m in ("fit", "sim", "dr", "gan"):
        d = e4["results"][m]
        tr = d.get("mae_train_series")
        rows.append(f"{lbl[m]} & {d['mae_held_out']:.3f} & "
                    f"{tr:.3f} \\\\" if tr is not None else
                    f"{lbl[m]} & {d['mae_held_out']:.3f} & n/a \\\\")
    parts.append(r"""
\begin{table}[!htbp]
\caption{\label{tab:s8}Early (30-s) $g^{(2)}(0)$ estimation MAE on real
data against each series' asymptotic reference.}
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
    for k in sorted(e5["baselines"]):
        rows.append(row(k.replace("_", " "), e5["baselines"][k]))
    for k in ("per", "uniform"):
        for i, s in enumerate(e5["final"][k]):
            rows.append(row(f"SAC {'PER' if k=='per' else 'uniform'} "
                            f"seed {i}", s))
    try:
        with open(f"{R}/exp5b_oracle.json") as f:
            rows.append(row("Oracle stopping (bound)", json.load(f)))
    except FileNotFoundError:
        pass
    parts.append(r"""
\begin{table}[!htbp]
\caption{\label{tab:s9}Closed-loop triage evaluation on 30 held-out
48-site fields: total measurement time per field, certification
precision and recall of good emitters, and throughput.}
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
    lbl = {"uncond_syn": "Uncond.\\ (synth.)",
           "graph_syn": "Graph-cond.\\ (synth.)",
           "oracle_real": "Oracle (real four)"}
    for m in ("uncond_syn", "graph_syn", "oracle_real"):
        cells = []
        for p in plats:
            for t in Ts:
                a = e6["results"][m][f"{p}@{t}"]["acc"]
                cells.append(f"${100*a[0]:.1f}{{\\pm}}{100*a[1]:.1f}$")
        rows.append(lbl[m] + " & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"{p} {t}s" for p in plats for t in Ts)
    parts.append(r"""
\begin{table*}[!htbp]
\caption{\label{tab:s10}Cross-platform transfer: balanced accuracy (\%)
per platform and acquisition time (mean $\pm$ 1~s.d.\ over three seeds).
All four real platforms are zero-shot for the two synthetic-trained
models.}
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

% ---- S11: capability comparison (cite-free copy of main-text sota table)
\providecommand{\scA}[1]{\parbox[t]{0.185\textwidth}{\raggedright #1\strut}}
\providecommand{\scB}[1]{\parbox[t]{0.105\textwidth}{\raggedright #1\strut}}
\providecommand{\scC}[1]{\parbox[t]{0.115\textwidth}{\raggedright #1\strut}}
\providecommand{\scD}[1]{\parbox[t]{0.100\textwidth}{\raggedright #1\strut}}
\providecommand{\scE}[1]{\parbox[t]{0.115\textwidth}{\raggedright #1\strut}}
\providecommand{\scF}[1]{\parbox[t]{0.255\textwidth}{\raggedright #1\strut}}
\newcommand{\tablesotasupp}{%
\begin{tabular}{llllll}
\hline\hline
\scA{Figure of merit} & \scB{Conventional pipeline} & \scC{Kudyshev '20} &
 \scD{Kudyshev '23} & \scE{Kedziora '23/'25} &
 \scF{\textbf{SPARQ (this work)}} \\[1.5pt]
\hline
\scA{Matched-accuracy screening speedup} & \scB{$1\times$ (ref.)} &
 \scC{${\sim}10^{2}\times$ vs.\ free fit} &
 \scD{${\sim}12\times$ vs.\ fit ($g^{(2)}$ maps)} & \scE{n/r} &
 \scF{$\speedupSnn\times$ vs.\ strong multi-start fit} \\[1.5pt]
\scA{Benchmark ceiling used} & \scB{asymptotic fit} &
 \scC{fit baseline} & \scD{fit baseline} & \scE{fit + UQ} &
 \scF{Monte-Carlo Bayes reference under the prior (tracked)} \\[1.5pt]
\scA{Median decision latency} & \scB{minutes} & \scC{${\sim}1$\,s} &
 \scD{${\sim}10$\,s} & \scE{$\geq 10$\,s} &
 \scF{\anyMedianHalf\,ms, anytime-gateable} \\[1.5pt]
\scA{Inference energy per decision} & \scB{n/r} & \scC{n/r (GPU)} &
 \scD{n/r (GPU)} & \scE{n/r} &
 \scF{\eSnnPointOneT\,nJ--\eSnnOneTuJ\,$\mu$J, event-proportional}
 \\[1.5pt]
\scA{Acquisition control} & \scB{open loop} & \scC{open loop} &
 \scD{open loop} & \scE{open loop} &
 \scF{closed loop: $\rlSpeedup\times$ vs.\ raster; oracle bound
 $\xOracleRaster\times$} \\[1.5pt]
\scA{Measurement-protocol design} & \scB{manual} & \scC{manual} &
 \scD{manual} & \scE{manual} &
 \scF{adjoint: $\tau_{\max}$ $60.5{\to}\wStar$\,ns,
 $+\adjGainSparse$\,pts} \\[1.5pt]
\scA{Early $g^{(2)}(0)$ MAE, 30-s real windows} &
 \scB{\ganMaeFit{} (peak area)} & \scC{n/r} & \scD{n/r} &
 \scE{own data, with UQ} &
 \scF{\ganMaeSim{} zero-shot (floor \floorMae)} \\[1.5pt]
\scA{Sim-to-real audit} & \scB{n/a} & \scC{none} & \scD{none} &
 \scE{bootstrap UQ} &
 \scF{floor decomposition + GAN critic (\pctRemovable\% of removable
 error closed)} \\[1.5pt]
\scA{Cross-platform transfer} & \scB{n/a} & \scC{none} & \scD{none} &
 \scE{fine-tuning, limited benefit} &
 \scF{zero-shot physics graphs: \gGapRecovery\% of oracle gap}
 \\[1.5pt]
\scA{Open release} & \scB{n/a} & \scC{none} & \scD{on request} &
 \scE{data + code} &
 \scF{code (GitHub) + benchmark data + trained models (Zenodo)} \\
\hline\hline
\end{tabular}}
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
