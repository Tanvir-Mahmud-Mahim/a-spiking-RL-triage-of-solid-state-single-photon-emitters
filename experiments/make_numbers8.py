"""Generate paper/numbers8.tex (main-text macros) and the Supporting
Information sections paper/supp_reviewer1.tex (Secs. S1.3-S1.4) and
paper/supp_reviewer2.tex (Secs. S3-S4) from
  results/exp7_reviewer.json   (first review round)
  results/exp7_parts/*         (auxiliary first-round files)
  results/exp8_r2.json         (second review round)
Every number in the generated prose is computed here from those files.
"""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import numpy as np

P = _ROOT + "/paper"
R = _ROOT + "/results"
d7 = json.load(open(f"{R}/exp7_reviewer.json"))
d8 = json.load(open(f"{R}/exp8_r2.json"))
b2exp = json.load(open(f"{R}/exp7_parts/B2_expected.json"))
rep = json.load(open(f"{R}/exp7_parts/B1_repeat_check.json"))

sweep = d7["stream_vs_exact_sweep"]
b2 = d7["twin_vs_stream_sweep"]
cov = d8["covariance"]
conv = d8["bayes_convergence_seeds"]
ps = d8["prior_shift_speedup"]
T_GRID = d8["T_grid"]
TARGET = d8["target"]


def f1(x):
    return f"{x:.1f}"


# ------------------------------------------------------------ S1.3 numbers
ncmin = min(c["n_coinc"] for c in sweep["configs"])
ncmax = max(c["n_coinc"] for c in sweep["configs"])
worst = max(sweep["configs"], key=lambda c: c["chi2_red"])
repvals = ", ".join(f"{v:.2f}" for v in rep["chi2_repeats"])
nb = [c for c in b2 if not c["blink"]]
bl = [c for c in b2 if c["blink"]][0]
ratios = [(c["rel_mean_err_pct"] / b2exp[c["lbl"]]["exp_err_pct"], c)
          for c in nb]
rmax, cmax = max(ratios, key=lambda x: x[0])
excess = np.sqrt(max(cmax["rel_mean_err_pct"] ** 2
                     - b2exp[cmax["lbl"]]["exp_err_pct"] ** 2, 0.0))
fano_nb = [c["fano"] for c in nb]

# ------------------------------------------------------------ S1.4 numbers
STAT = ["A_noimp", "A_imp", "B_noimp", "B_imp"]
abs_vals = [cov[k]["mean_abs_offdiag"] for k in STAT]
frac_vals = [100 * cov[k]["frac_above_2sigma"] for k in STAT]
exp_abs = cov["A_noimp"]["expected_mean_abs"]
blink = cov["C_blink"]
# rescaled blinking matrix: full statistics for the table row
_Hb = np.load(f"{R}/exp8_parts/A2_C_blink.npy")
_s = _Hb.sum(1)
_Hn = _Hb / _s[:, None] * _s.mean()
_Cn = np.corrcoef(_Hn.T)
_off = _Cn[~np.eye(_Cn.shape[0], dtype=bool)]
blink_n = dict(mean_bin_count=float(_Hn.mean()),
               mean_offdiag=float(_off.mean()),
               mean_abs_offdiag=float(np.abs(_off).mean()),
               p95_abs_offdiag=float(np.percentile(np.abs(_off), 95)),
               max_abs_offdiag=float(np.abs(_off).max()),
               adjacent_mean=float(np.mean(np.diag(_Cn, 1))),
               frac_above_2sigma=float(np.mean(np.abs(_off)
                                               > 2 / np.sqrt(len(_Hb)))))
assert abs(blink_n["mean_offdiag"] - blink["norm_mean_offdiag"]) < 1e-9
sum_constraint = -1.0 / (_Hb.shape[1] - 1)

# ------------------------------------------------------------ Bayes gap
e2 = json.load(open(f"{R}/exp2_estimators.json"))
e2b = json.load(open(f"{R}/exp2b_snn.json"))
_Tg = e2["T_grid"]
_gaps = []
for i, T in enumerate(_Tg):
    if T > 1.0:
        continue
    key = f"{T:g}" if f"{T:g}" in conv[sorted(conv)[0]] else str(T)
    bay = np.mean([100 * conv[s][key]["48000"] for s in conv])
    _gaps.append(abs(100 * e2b["snn_sparse"]["acc"][i][0] - bay))
    _gaps.append(abs(100 * e2["results"]["cnn_pitl"]["acc"][i][0] - bay))
bayes_gap_max = max(_gaps)

# ------------------------------------------------------------ S3 numbers
seeds = sorted(conv)
Ms = ["1500", "3000", "6000", "12000", "24000", "48000"]
Ts = [t for t in ("0.03", "0.1", "0.3", "1.0", "30.0")
      if all(t in conv[s] for s in seeds)]


def ref(T, M):
    v = np.array([100 * conv[s][T][M] for s in seeds])
    return v.mean(), v.std(ddof=1)


rise = {T: ref(T, "48000")[0] - ref(T, "6000")[0] for T in Ts}
last = {T: ref(T, "48000")[0] - ref(T, "24000")[0] for T in Ts}
sparse = [T for T in Ts if float(T) <= 0.3]
max_rise_sparse = max(abs(rise[T]) for T in sparse)
max_last_sparse = max(abs(last[T]) for T in sparse)

# ------------------------------------------------------------ S4 numbers
LBL = {"in-prior": "in-prior (control)",
       "tau1 +30%": "$\\tau_1$ range $+30$\\%",
       "tau1 -30%": "$\\tau_1$ range $-30$\\%",
       "rate x0.6": "count-rate range $\\times 0.6$",
       "tau2 +50%": "$\\tau_2$ range $+50$\\%",
       "blink 2x": "blinking probability $\\times 2$"}
ORDER = [k for k in LBL if k in ps]
iT = {t: T_GRID.index(t) for t in (0.1, 1.0)}


def acc(k, m, t):
    a = ps[k][m]["acc"][iT[t]]
    return 100 * a[0], 100 * a[1]


inp = ps["in-prior"]
drops = []
for k in ORDER:
    if k == "in-prior":
        continue
    for m in ("snn", "cnn"):
        for t in (0.1, 1.0):
            drops.append(acc("in-prior", m, t)[0] - acc(k, m, t)[0])
max_drop = max(drops)


def drop_of(k):
    return max(acc("in-prior", m, t)[0] - acc(k, m, t)[0]
               for m in ("snn", "cnn") for t in (0.1, 1.0))


def absdiff_of(k):
    return max(abs(acc("in-prior", m, t)[0] - acc(k, m, t)[0])
               for m in ("snn", "cnn") for t in (0.1, 1.0))


long_scales = [k for k in ("tau1 +30%", "tau2 +50%") if k in ps]
small_change = max(absdiff_of(k) for k in long_scales)
mid_drop = max(drop_of(k) for k in ("tau1 -30%", "blink 2x") if k in ps)
rate_drop = drop_of("rate x0.6")
fit_in_1 = 100 * ps["in-prior"]["fit"]["acc"][iT[1.0]][0]
fit_rate_1 = 100 * ps["rate x0.6"]["fit"]["acc"][iT[1.0]][0]
worst_sp = min(shifted_ := [k for k in ORDER if k != "in-prior"],
               key=lambda k: ps[k]["speedup_snn"] or 0)
sp_snn = {k: ps[k]["speedup_snn"] for k in ORDER}
sp_cnn = {k: ps[k]["speedup_cnn"] for k in ORDER}
shifted = [k for k in ORDER if k != "in-prior"]
reached = [k for k in shifted if sp_snn[k] is not None]
not_reached = [k for k in shifted if sp_snn[k] is None]

# ------------------------------------------------------------ macros
with open(f"{P}/numbers8.tex", "w") as f:
    f.write("% AUTO-GENERATED by experiments/make_numbers8.py\n")
    m = {
        "bTwoBlinkErr": f1(bl["rel_mean_err_pct"]),
        "bTwoBlinkExp": f1(b2exp[bl["lbl"]]["exp_err_pct"]),
        "bTwoBlinkFano": f"{bl['fano']:.2f}",
        "covAbsMin": f"{min(abs_vals):.3f}",
        "covAbsMax": f"{max(abs_vals):.3f}",
        "covAbsExp": f"{exp_abs:.3f}",
        "covFracMin": f1(min(frac_vals)),
        "covFracMax": f1(max(frac_vals)),
        "covBlinkMean": f"{blink['mean_offdiag']:+.2f}",
        "bRiseTenthSixToFortyEight": f1(rise["0.1"]),
        "bRiseOneSixToFortyEight": f1(rise["1.0"]),
        "bRiseOneLastDoubling": f1(last["1.0"]),
        "bRiseThirtySixToFortyEight": f1(rise["30.0"]),
        "bRiseThirtyLastDoubling": f1(last["30.0"]),
        "dShiftMaxDrop": f1(max_drop),
        "covBlinkCV": f1(100 * blink["cv_total"]),
        "covBlinkCVPois": f1(100 * blink["cv_total_poisson"]),
        "covBlinkNormMean": f"{blink_n['mean_offdiag']:+.3f}",
        "covBlinkNormAbs": f"{blink_n['mean_abs_offdiag']:.3f}",
        "covBlinkNormFrac": f1(100 * blink_n["frac_above_2sigma"]),
        "covSumConstraint": f"{sum_constraint:+.3f}",
        "bayesGapMax": f1(bayes_gap_max),
    }
    if not not_reached:
        lo = min(sp_snn[k] for k in shifted)
        hi = max(sp_snn[k] for k in shifted)
        m["dShiftSpeedupSentence"] = (
            f"under every shifted prior tested, the SNN still reached the "
            f"target {lo:.1f} to {hi:.1f} times faster than the LM fit "
            f"(in-prior control {sp_snn['in-prior']:.1f}).")
    else:
        m["dShiftSpeedupSentence"] = (
            "under some shifted priors the LM fit did not reach the "
            "target within 30\\,s, so no speedup is defined there.")
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ------------------------------------------------------------ tables
rows_sweep = "\n".join(
    f"{c['label']} & {c['tau1']:.2f} & {c['tau2']:.0f} & {c['a']:.2f} & "
    f"{c['n_coinc']:,} & {c['chi2_red']:.2f} & {c['nrmse']:.3f} \\\\"
    for c in sweep["configs"])
rows_b2 = "\n".join(
    f"{c['lbl']} & {c['n']} & {c['rho']:.2f} & {c['rate']} & "
    f"{'yes' if c['blink'] else 'no'} & {c['rel_mean_err_pct']:.2f} & "
    f"{b2exp[c['lbl']]['exp_err_pct']:.2f} & {c['fano']:.3f} \\\\"
    for c in b2)
COND = {"A_noimp": "A, ideal", "A_imp": "A, dead time + afterp.",
        "B_noimp": "B, ideal", "B_imp": "B, dead time + afterp.",
        "C_blink": "C (blinking), ideal"}
rows_cov = "\n".join(
    f"{COND[k]} & {cov[k]['mean_bin_count']:.1f} & "
    f"{cov[k]['mean_offdiag']:+.4f} & {cov[k]['mean_abs_offdiag']:.4f} & "
    f"{cov[k]['p95_abs_offdiag']:.3f} & {cov[k]['max_abs_offdiag']:.3f} & "
    f"{cov[k]['adjacent_mean']:+.4f} & "
    f"{100*cov[k]['frac_above_2sigma']:.1f} & {cov[k]['fano']:.3f} \\\\"
    for k in COND) + "\n" + (
    f"C, rescaled to mean total & {blink_n['mean_bin_count']:.1f} & "
    f"{blink_n['mean_offdiag']:+.4f} & {blink_n['mean_abs_offdiag']:.4f} & "
    f"{blink_n['p95_abs_offdiag']:.3f} & {blink_n['max_abs_offdiag']:.3f} & "
    f"{blink_n['adjacent_mean']:+.4f} & "
    f"{100*blink_n['frac_above_2sigma']:.1f} & n/a \\\\")
rows_conv = "\n".join(
    f"{float(T):g}\\,s & " + " & ".join(
        "{:.1f} $\\pm$ {:.1f}".format(*ref(T, M)) for M in Ms) + " \\\\"
    for T in Ts)
rows_acc = "\n".join(
    f"{LBL[k]} & " + " & ".join(
        "{:.1f} $\\pm$ {:.1f}".format(*acc(k, mm, t))
        for mm in ("snn", "cnn") for t in (0.1, 1.0)) + " \\\\"
    for k in ORDER)


def fmt_t(x):
    return f"{x:.2f}" if x is not None else "n.r."


def fmt_sp(x):
    return f"{x:.1f}" if x is not None else "n/a"


rows_ttt = "\n".join(
    f"{LBL[k]} & {fmt_t(ps[k]['ttt']['fit'])} & "
    f"{fmt_t(ps[k]['ttt']['cnn'])} & {fmt_t(ps[k]['ttt']['snn'])} & "
    f"{fmt_sp(sp_cnn[k])} & {fmt_sp(sp_snn[k])} \\\\" for k in ORDER)

# ------------------------------------------------------------ SI text 1
tex1 = r"""% AUTO-GENERATED by experiments/make_numbers8.py

\subsection{Validation over sampled and boundary parameter sets}
\label{ssec:sweep}
The photon-stream versus exact comparison of the main text was
repeated for """ + str(sweep["n"]) + r""" photophysical parameter sets
$(\tau_1, \tau_2, a)$: five drawn at random from the prior of each
platform (NV, hBN, GaN, SiV) and six boundary cases that combine the
extremes of the prior ranges. As in the main-text protocol, each set
was simulated for a single emitter at a detected rate of 1500\,kcps
with $\rho = 1$ and ideal detectors, so this sweep tests the
photophysical parameters only; background, multiplicity, count rate,
detector effects, and blinking are tested in
Table~\ref{tab:b2sweep}, Table~\ref{tab:cov}, and
Figure~\ref{fig:cov}. The acquisition time was 1.2\,s, shortened in
proportion to $\tau_1$ for the fastest emitters to keep the
simulation within memory, which gave """ + f"{ncmin:,} to {ncmax:,}" + r"""
coincidences per set (Table~\ref{tab:sweep}). The reduced $\chi^2$
ranged from """ + f"{sweep['chi2_min']:.2f} to {sweep['chi2_max']:.2f} (mean {sweep['chi2_mean']:.2f})" + r""".
The largest value, """ + f"{worst['chi2_red']:.2f}" + r""", for the
shortest $\tau_1$ combined with the longest $\tau_2$ and the largest
$a$, was repeated with four new noise seeds and gave
$\chi^2_\nu = """ + repvals + r"""$, consistent with a statistical
fluctuation. We found no systematic deviation between the
photon-stream simulator and the exact solution for these parameter
sets; the sweep does not sample the complete joint space of all
simulator parameters.

Table~\ref{tab:b2sweep} compares the histogram simulator with the
photon-stream simulator under six representative compound operating
conditions that vary multiplicity, background fraction, count rate,
and blinking, together with the mean absolute deviation expected from
counting statistics alone for 60 repetitions of a 1-s acquisition. For
the five stationary conditions the measured deviation is within a
factor of """ + f"{rmax:.1f}" + r""" of this statistical expectation
(the largest ratio occurs at the highest count rate, where the
expectation is smallest), which limits any systematic difference to
about """ + f"{excess:.0f}" + r"""\% of the mean counts, and the Fano
factors lie between """ + f"{min(fano_nb):.2f} and {max(fano_nb):.2f}" + r""".
The blinking condition deviates by """ + f1(bl["rel_mean_err_pct"]) + r"""\%
against a statistical expectation of
""" + f1(b2exp[bl["lbl"]]["exp_err_pct"]) + r"""\%, with a Fano factor
of """ + f"{bl['fano']:.2f}" + r""". The histogram simulator represents
blinking only through the duty cycle of the telegraph process, which
reproduces the mean singles rate but neither the rate fluctuations
between acquisitions nor, exactly, the resulting bunching pedestal.
Statements that the histogram simulator reproduces the coincidence
statistics therefore apply to stationary, non-blinking sources; for
blinking sources it is an approximation, and the photon-stream
simulator, which models blinking event by event, was used for all
validation.

\begin{table}[!htb]
\caption{\label{tab:sweep}Photon-stream simulator compared with the
numerically exact master-equation $g^{(2)}(\tau)$ for random and
boundary photophysical parameter sets (single emitter, detected rate
1500\,kcps, $\rho = 1$, ideal detectors; 121 delay bins, Poisson
errors; NRMSE: normalized root-mean-square error).}
\begin{center}
\small
\begin{tabular}{lcccccc}
\hline\hline
Parameter set & $\tau_1$ (ns) & $\tau_2$ (ns) & $a$ & Coinc. &
$\chi^2_\nu$ & NRMSE \\
\hline
""" + rows_sweep + r"""
\hline\hline
\end{tabular}
\end{center}
\end{table}

\begin{table}[!htb]
\caption{\label{tab:b2sweep}Histogram simulator compared with the
photon-stream simulator under compound operating conditions
($\tau_1 = 14$\,ns, $\tau_2 = 250$\,ns, $a = 0.8$; 60 repeated 1-s
acquisitions each; IRF jitter; blinking condition:
$t_{\rm on} = 20$\,ms, $t_{\rm off} = 8$\,ms). Measured: mean absolute
relative deviation of the per-bin mean counts. Expected: the same
quantity predicted by Poisson counting statistics alone for 60
repetitions.}
\begin{center}
\small
\begin{tabular}{lccccccc}
\hline\hline
Condition & $N$ & $\rho$ & Rate (kcps) & Blinking & Measured (\%) &
Expected (\%) & Fano \\
\hline
""" + rows_b2 + r"""
\hline\hline
\end{tabular}
\end{center}
\end{table}

\subsection{Bin-to-bin correlations of the photon-stream simulator}
\label{ssec:cov}
The histogram simulator treats every delay bin as an independent
Poisson variable. A unit Fano factor does not establish that the bins
are independent, so we measured their correlations directly. For each
condition, """ + str(cov["A_noimp"]["n_rep"]) + r""" acquisitions of
""" + f"{cov['A_noimp']['acq_T_s']:g}" + r"""\,s were simulated with the
photon-stream simulator ($\tau_1 = 14$\,ns, $\tau_2 = 250$\,ns,
$a = 0.8$, IRF jitter $\sigma_{\rm IRF} = 0.35$\,ns), and the
$121 \times 121$ correlation matrix of the per-bin counts was
computed. Condition A has two emitters, $\rho = 0.8$, and 500\,kcps;
condition B has one emitter, $\rho = 0.95$, and 200\,kcps. Both were
simulated with ideal detectors and with 45-ns dead time and 2\%
afterpulsing (80-ns exponential delay). Condition C is a blinking
single emitter ($\rho = 0.85$, 400\,kcps, $t_{\rm on} = 20$\,ms,
$t_{\rm off} = 8$\,ms) with ideal detectors. Figure~\ref{fig:cov}
shows the correlation matrices and Table~\ref{tab:cov} their
statistics. For """ + str(cov["A_noimp"]["n_rep"]) + r""" uncorrelated
samples the expected mean absolute correlation coefficient is
$\sqrt{2/\pi}/\sqrt{n_{\rm rep}} = """ + f"{exp_abs:.3f}" + r"""$, and a
fraction of 4.6\% of the coefficients is expected to exceed
$2/\sqrt{n_{\rm rep}}$ by chance. In the four stationary cases the
mean absolute coefficient
(""" + f"{min(abs_vals):.3f}--{max(abs_vals):.3f}" + r""") and the
fraction beyond $2/\sqrt{n_{\rm rep}}$
(""" + f"{min(frac_vals):.1f}--{max(frac_vals):.1f}" + r"""\%) agree with
these expectations, the mean coefficient and the nearest-neighbor
coefficients are close to zero, and the maps show no structure. No
statistically significant bin-to-bin correlations were therefore
detected under the tested stationary conditions, with or without
detector dead time and afterpulsing; this does not exclude
correlations below the sampling precision
($1/\sqrt{n_{\rm rep}} = """ + f"{cov['A_noimp']['noise_floor']:.3f}" + r"""$)
or under conditions that were not tested. In the blinking case, by
contrast, all bins are positively correlated (mean coefficient
""" + f"{blink['mean_offdiag']:+.3f}" + r""", Fano factor
""" + f"{blink['fano']:.2f}" + r""") [Figure~\ref{fig:cov}(e)]. The
fraction of the 0.5-s acquisition that the emitter spends in the on
state varies from one acquisition to the next, so the total number of
coincidences fluctuates by """ + f"{100*blink['cv_total']:.1f}" + r"""\%
(relative standard deviation) instead of the
""" + f"{100*blink['cv_total_poisson']:.1f}" + r"""\% expected from
Poisson statistics, and this common factor multiplies every bin. When
each histogram is rescaled to the mean total count, the correlations
disappear [Figure~\ref{fig:cov}(f)]: the mean coefficient becomes
""" + f"{blink_n['mean_offdiag']:+.4f}" + r""", equal to the value
$-1/(n_{\rm bins}-1) = """ + f"{sum_constraint:+.4f}" + r"""$ imposed by
the fixed total alone, and """ + f"{100*blink_n['frac_above_2sigma']:.1f}" + r"""\%
of the coefficients exceed $2/\sqrt{n_{\rm rep}}$. The blinking-induced
correlation is therefore a fluctuation of the overall scale of the
histogram rather than a coupling between particular delays. The
histogram simulator, which assigns blinking emitters a fixed duty
cycle, does not reproduce this extra variance of the overall scale;
this is the same effect that limits it for blinking sources
(Table~\ref{tab:b2sweep}).

\begin{figure}[!htb]
\centering
\includegraphics[width=0.9\textwidth]{figS1_covariance.pdf}
\caption{\label{fig:cov}Correlation matrices of the per-bin
coincidence counts of """ + str(cov["A_noimp"]["n_rep"]) + r""" repeated
""" + f"{cov['A_noimp']['acq_T_s']:g}" + r"""-s photon-stream acquisitions
(diagonal omitted; color scale $\pm 0.25$). (a),(b)~Condition
A (two emitters, $\rho = 0.8$, 500\,kcps) with ideal detectors and
with dead time and afterpulsing. (c),(d)~Condition B (one emitter,
$\rho = 0.95$, 200\,kcps), same detector settings. (e)~Condition C,
a blinking single emitter with ideal detectors. (f)~Condition C after
each histogram is rescaled to the mean total coincidence count.}
\end{figure}

\begin{table}[!htb]
\caption{\label{tab:cov}Statistics of the off-diagonal elements of the
bin-to-bin correlation matrices of Figure~\ref{fig:cov}
(""" + str(cov["A_noimp"]["n_rep"]) + r""" repetitions each; expected
for uncorrelated data: mean $r = 0$, mean $|r| = """ + f"{exp_abs:.3f}" + r"""$,
4.6\% of coefficients with $|r| > 2/\sqrt{n_{\rm rep}}$).
Adj.: mean coefficient of adjacent bins. Last row: condition C after
each histogram is rescaled to the mean total count, for which the fixed
total alone gives a mean $r$ of """ + f"{sum_constraint:+.4f}" + r""".}
\begin{center}
\small
\begin{tabular}{lcccccccc}
\hline\hline
Condition & Counts/bin & Mean $r$ & Mean $|r|$ & p95 $|r|$ &
Max $|r|$ & Adj. & $>2\sigma$ (\%) & Fano \\
\hline
""" + rows_cov + r"""
\hline\hline
\end{tabular}
\end{center}
\end{table}
"""

# ------------------------------------------------------------ SI text 2
s6000_1 = 100 * conv["77"]["1.0"]["6000"]
tex2 = r"""\section{Monte-Carlo Bayes reference and its dependence on the
reference-sample size}
\label{sec:bayesconv}
The Monte-Carlo Bayes reference is the decision obtained from the
posterior under the same prior that the estimators are trained on:
for each evaluation acquisition, exact Poisson likelihoods of the
observed histogram and singles count are evaluated for $M$ sites
drawn from the prior, and the site is classified by posterior mass.
It is the Bayes-optimal decision under the assumed simulation prior,
not a universal physical limit. Because the posterior is computed
from a finite sample of the prior, the reference accuracy is expected
to increase with $M$, and more so at long acquisition times, where the
likelihood is concentrated on a smaller fraction of the sample.
Table~\ref{tab:bconv} lists the reference accuracy for $M = 1500$ to
$48\,000$, averaged over """ + str(len(seeds)) + r""" independent
reference samples, for the fixed 1200-site NV evaluation population
and noise seed of the main text. Between $M = 6000$ and $48\,000$ the
mean changes by at most """ + f1(max_rise_sparse) + r""" points at
$T \leq 0.3$\,s (at most """ + f1(max_last_sparse) + r""" points between
$M = 24\,000$ and $48\,000$); at $T = 1$\,s it increases by
""" + f1(rise["1.0"]) + r""" points, of which """ + f1(last["1.0"]) + r"""
between $M = 24\,000$ and $48\,000$; and at $T = 30$\,s it increases by
""" + f1(rise["30.0"]) + r""" points, of which """ + f1(last["30.0"]) + r"""
in the last doubling. At $T = 30$\,s the reference is therefore still
increasing with $M$ over the tested range, which is why trained
estimators can exceed the finite-$M$ value at long acquisition times;
we use the reference only for $T \leq 1$\,s. The main-text
Figure~3(a) shows the $M = 48\,000$ mean. The single-sample
$M = 6000$ values listed in Table~\ref{tab:s3} (last row), which were
plotted in the previous versions of Figure~3(a), are lower by up to
""" + f1(ref("1.0", "48000")[0] - s6000_1) + r""" points at $T \leq 1$\,s.

\begin{table}[!htb]
\caption{\label{tab:bconv}Monte-Carlo Bayes reference (balanced
accuracy, \%) versus reference-sample size $M$: mean $\pm$ standard
deviation over """ + str(len(seeds)) + r""" independent reference
samples; fixed 1200-site NV evaluation population and noise seed.}
\begin{center}
\footnotesize
\begin{tabular}{lcccccc}
\hline\hline
$T$ & $M{=}1500$ & $M{=}3000$ & $M{=}6000$ & $M{=}12\,000$ &
$M{=}24\,000$ & $M{=}48\,000$ \\
\hline
""" + rows_conv + r"""
\hline\hline
\end{tabular}
\end{center}
\end{table}

\section{Robustness to prior misspecification}
\label{sec:priorshift}
The estimators are trained on the stated NV prior, which a real
sample will not match exactly. To test the sensitivity, the trained
SNN and CNN were evaluated without retraining on populations of 1200
sites drawn from deliberately shifted priors: the antibunching-time
range scaled by $\pm 30$\%, the count-rate range scaled by $0.6$, the
bunching-time range scaled by $1.5$, and the blinking probability
doubled. For each prior, balanced accuracy was computed at all seven
acquisition times of the main text with five noise seeds for the
networks and two noise seeds (400 sites) for the LM fit, as in the
main-text protocol, and the time to reach the """ + f"{100*TARGET:.0f}" + r"""\%
target was interpolated in $\log T$. Table~\ref{tab:pshift} gives the
balanced accuracy at 0.1 and 1\,s, and Table~\ref{tab:pshiftttt} the
times to target and the resulting acquisition-time ratios relative to
the LM fit. Because each shifted prior also generates a different
population of sites, the changes combine prior misspecification with a
change in the intrinsic difficulty of the population. Lengthening the
antibunching-time or bunching-time range changed the network accuracy
at 0.1 and 1\,s by at most """ + f1(small_change) + r""" points;
shortening the antibunching-time range or doubling the blinking
probability reduced it by up to """ + f1(mid_drop) + r""" points; and
the reduced count-rate range reduced it by up to """ + f1(rate_drop) + r"""
points. The LM fit, which does not use the prior, also lost accuracy at
the reduced count rate (""" + f1(fit_rate_1) + r"""\% versus
""" + f1(fit_in_1) + r"""\% at 1\,s), so part of this decrease reflects
the smaller number of coincidences per acquisition rather than
misspecification alone. """ + (
    (r"""Under every shifted prior the SNN reached the target
""" + f"{min(sp_snn[k] for k in shifted):.1f} to {max(sp_snn[k] for k in shifted):.1f}" + r"""
times faster than the LM fit (in-prior control
""" + f"{sp_snn['in-prior']:.1f}" + r"""; smallest ratio for the
""" + LBL[worst_sp].replace("range", "range shifted by") .replace("shifted by $-30$\\%", "shortened by 30\\%") + r"""). The
acquisition-time reduction is therefore not specific to the training
prior within the range of shifts tested, although its size depends on
the prior. These values are based on one evaluation population per
prior and five noise seeds for the networks (two for the LM fit).""")
    if not not_reached else
    (r"""Under some shifted priors the LM fit did not reach the target
within 30\,s (n.r.\ in Table~\ref{tab:pshiftttt}); there the speedup is
not defined, and our conclusion for those priors is restricted to the
classification accuracy.""")) + r""" These results cover the listed
shifts only. The experimental quantum-dot data of the main text differ
from the CW training priors in kind (pulsed excitation and a
different emitter class) and were analyzed with a separately trained
network; we do not quantify the size of that mismatch.

\begin{table}[!htb]
\caption{\label{tab:pshift}Balanced accuracy (\%, mean $\pm$ s.d.\ over
five noise seeds) of the trained estimators on deliberately shifted
evaluation priors (no retraining; 1200 sites per prior).}
\begin{center}
\small
\begin{tabular}{lcccc}
\hline\hline
 & \multicolumn{2}{c}{SNN} & \multicolumn{2}{c}{CNN} \\
Evaluation prior & $T{=}0.1$\,s & $T{=}1$\,s & $T{=}0.1$\,s &
$T{=}1$\,s \\
\hline
""" + rows_acc + r"""
\hline\hline
\end{tabular}
\end{center}
\end{table}

\begin{table}[!htb]
\caption{\label{tab:pshiftttt}Acquisition time (s) to reach
""" + f"{100*TARGET:.0f}" + r"""\% balanced accuracy on each evaluation
prior, and the ratio of the LM-fit time to the network time
(n.r.: not reached within 30\,s; n/a: not defined).}
\begin{center}
\small
\begin{tabular}{lccccc}
\hline\hline
Evaluation prior & LM fit & CNN & SNN & Ratio (CNN) & Ratio (SNN) \\
\hline
""" + rows_ttt + r"""
\hline\hline
\end{tabular}
\end{center}
\end{table}
"""

open(f"{P}/supp_reviewer1.tex", "w").write(tex1)
open(f"{P}/supp_reviewer2.tex", "w").write(tex2)
print("wrote numbers8.tex, supp_reviewer1.tex, supp_reviewer2.tex")
print("max drop", max_drop, "speedups", sp_snn)
