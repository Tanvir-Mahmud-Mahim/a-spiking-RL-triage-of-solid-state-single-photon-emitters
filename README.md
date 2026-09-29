# SPARQ: early decisions and adaptive exposure for single-photon-emitter screening

Code accompanying the article **"Closed-loop, event-driven machine learning
for autonomous triage of single-photon emitters"** by T. M. Mahim,
M. N. Islam, M. M. Rahman, and A. S. M. Mohsin (revised version).

Repository: https://github.com/Tanvir-Mahmud-Mahim/a-spiking-RL-triage-of-solid-state-single-photon-emitters

Archived data (result files and trained models): https://doi.org/10.5281/zenodo.21352758
(this DOI always points to the latest version; the version matching the
second revision is https://doi.org/10.5281/zenodo.23045625)

## What the code does

Deciding whether a spot on a sample is a single-photon emitter needs a
Hanbury Brown-Twiss (HBT) measurement: photon pairs from two detectors are
collected into a histogram of arrival-time differences, the second-order
correlation g2(tau). This is slow, because photon pairs arrive at a rate
proportional to the square of the count rate. The code studies three ways
to shorten it:

1. **Deciding early.** A spiking neural network reads the photon-pair
   histogram in short time slices and updates, after every slice, the
   probability that the site is a single emitter, so a decision can be made
   before the exposure ends.
2. **Adapting the exposure of each site.** A reinforcement-learning
   controller decides, site by site, whether to measure longer, accept
   (certify), or reject, and is compared with a simple threshold rule and
   with fixed-time scanning.
3. **One estimator for several emitter types.** A small graph that
   describes the energy levels of an emitter type (NV, hBN, GaN, SiV) tells
   the estimator which type it is measuring.

All networks are trained on simulated measurements from a standard
three-level emitter model, which is first checked against the exact
solution. The estimation step is also tested on published experimental
quantum-dot data.

## Requirements

Python 3.10 or newer with numpy, scipy, matplotlib, and PyTorch:

```bash
pip install -r requirements.txt
```

Everything runs on a CPU. The experimental quantum-dot data are read from
the open sps-quality repository; clone it next to this repository:

```bash
git clone https://github.com/UTS-CASLab/sps-quality ../sps-quality
```

## Reproduce every figure and number

```bash
./run_all.sh
```

This runs all experiments in order, then writes the numbers quoted in the
article and the Supporting Information (SI) tables, and draws all figures.
The full run takes several hours on a desktop CPU. The two revision
analyses (`exp7run.py`, `exp8_r2.py`) are split into parts that save their
progress, so they can be stopped and restarted. Random seeds are fixed in
every script.

Outputs:

* `results/` : result files (JSON) of every experiment, and intermediate
  files of the revision analyses (`exp7_parts/`, `exp8_parts/`).
* `results/models/` : trained network weights.
* `figures/` : Figures 1 to 6, Figure S1, and the Table of Contents graphic
  (PDF and PNG).
* `paper/` : LaTeX files with the numbers and SI tables used by the
  manuscript.

The result files and trained weights used in the article are also in the
Zenodo record, so the figures can be drawn without rerunning the
experiments.

Paths are taken relative to the repository folder. Set the environment
variable `SPARQ_ROOT` to use another folder, and `SPS_QUALITY` if the
sps-quality data are stored somewhere other than `../sps-quality`.

## Files

### Core library (`sparq/`)

| File | Contents |
|---|---|
| `physics.py` | Emitter model (three levels), parameter ranges of the four emitter types, the fast histogram simulator (independent Poisson counts in each bin; exact for steady, non-blinking emitters), and the slower photon-by-photon simulator with blinking, detector dead time, afterpulsing, and timing jitter |
| `exact.py` | Exact g2(tau) of the three-level model from its rate equations |
| `pulsed.py` | Pulsed-excitation version of the simulator and the conventional peak-area analysis, used for the quantum-dot data |
| `datasets.py` | Generators of simulated training and test measurements; reader for the sps-quality data |
| `estimators.py` | The conventional least-squares fit, the convolutional network (CNN), and the spiking network (SNN), with training and scoring |
| `twin_torch.py` | Differentiable version of the histogram simulator, used to tune the excitation power and correlation window, and the Fisher-information calculation |
| `rl_env.py` | Simulated field of candidate sites for the adaptive-exposure study, and the fixed-time and threshold baselines |
| `sac_per.py` | Reinforcement-learning agent (soft actor-critic with prioritized replay) |
| `gnn.py` | Energy-level graphs of the emitter types and the graph encoder |

### Experiments (`experiments/`)

| Script | What it produces | Article |
|---|---|---|
| `exp1_validate_twin.py` | Simulator checked against the exact solution; fast simulator checked against the photon-by-photon simulator | Fig. 2, Table S1 |
| `exp2_estimators.py` | Accuracy and g2(0) error versus measurement time for the fit, CNN, and SNN; time to reach 90% accuracy | Fig. 3(a,b), Table 1, Tables S9, S10 |
| `exp2b_snn_sparse.py` | SNN trained to use few spikes: early decisions and energy estimate | Fig. 3(c,d), Tables S11, S12 |
| `exp3_adjoint.py` | Joint tuning of excitation power and correlation window | Fig. 4(a,b), Table S13 |
| `exp4_gan.py` | Test on the experimental quantum-dot records | Fig. 4(c,d), Tables S14, S15 |
| `exp4c_floor.py` | Error of the same network on simulated 30-s windows (reference line) | Fig. 4(d) |
| `exp5_rl.py`, `exp5b_oracle.py` | Adaptive exposure: learning, baselines, and the ideal-stopping bound | Fig. 5, Table S16 |
| `exp6_graph.py`, `exp6b_graph_synth.py` | Transfer between emitter types | Fig. 6, Table S17 |
| `exp7run.py` | First-revision checks: validation over sampled and boundary parameter sets and under mixed operating conditions | Tables S2, S3 |
| `exp8_r2.py` | Second-revision checks: bin-to-bin correlations, dependence of the Bayes reference on sample size, results under shifted parameter ranges | Fig. S1, Tables S4, S6, S7, S8 |
| `fig1_traces.py` | SNN output traces shown in Figure 1 | Fig. 1(b) |
| `recompute_energy.py` | Energy per decision from the counted operations | Fig. 3(d), Table 1, Table S12 |
| `make_numbers.py`, `make_numbers7.py`, `make_numbers8.py` | Numbers quoted in the text, and SI Sections S1.3, S1.4, S3, S4 | |
| `make_supp.py` | The remaining SI tables | |

### Figures (`figures/`)

`fig1_concept.py` to `fig6_graph.py`, `figS1_covariance.py`, and
`toc_graphic.py` draw the figures from the result files; `style.py` sets
the common style (Times New Roman or Liberation Serif, black axes).

## Notes

* **What is simulated and what is measured.** All closed-loop and transfer
  results are simulations. Only the estimation step is tested on
  experimental data (one InGaAs/GaAs quantum dot from the sps-quality
  dataset).
* **Normalization of g2.** No estimator uses the edge of the correlation
  window to normalize the histogram. The uncorrelated level g2 = 1 is
  computed from the measured single-detector count rate, so shortening the
  window does not change it.
* **Blinking.** The fast histogram simulator treats blinking only through
  its average on-time, so it is exact only for steady, non-blinking
  emitters. The photon-by-photon simulator models blinking photon by
  photon and is used for all checks.
* **Energy estimates** count the operations of each network and multiply by
  published energies per operation: 23.6 pJ per synaptic operation (Intel
  Loihi; Davies et al., IEEE Micro 38, 82 (2018),
  doi:10.1109/MM.2018.112130359), and 4.6 pJ (32-bit floating point) or
  0.23 pJ (8-bit integer) per multiply-accumulate at 45 nm (Horowitz, ISSCC
  2014, doi:10.1109/ISSCC.2014.6757323). They are order-of-magnitude
  estimates.

## Data source

The experimental quantum-dot HBT measurements are from the openly licensed
[sps-quality](https://github.com/UTS-CASLab/sps-quality) repository
(Kedziora et al., Mach. Learn.: Sci. Technol. 4, 045042 (2023),
doi:10.1088/2632-2153/ad0d11). They are not redistributed here.

## Changes in version 3 (second revision)

* New analyses (`exp8_r2.py`): bin-to-bin correlation maps at three
  operating conditions (Figure S1), dependence of the Bayes reference on the
  number of reference sites (up to 48 000, five repeats), and time to the
  90% target under shifted parameter ranges (five noise seeds).
* Energy: the 8-bit multiply-accumulate energy is now 0.23 pJ (0.2 pJ
  multiply plus 0.03 pJ add; Horowitz 2014) instead of 1 pJ. Accuracy
  results are unchanged.
* Figure 1 redrawn as a schematic of the measurement; new Figure S1; black
  tick labels in all figures.
* Paths are relative to the repository, so the code runs from any folder.
* Removed the unused `figures/fig0_abstract.py`.

## License

Code: Apache-2.0 (see `LICENSE`). Data on Zenodo: CC-BY 4.0.
