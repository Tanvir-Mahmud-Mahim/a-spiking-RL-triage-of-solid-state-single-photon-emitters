#!/bin/bash
# Run the full SPARQ experiment suite in order (CPU; several hours).
# The experimental quantum-dot data are read from ../sps-quality or from
# the directory given by the SPS_QUALITY environment variable.
set -e
cd "$(dirname "$0")"
mkdir -p results paper
python3 experiments/exp1_validate_twin.py
python3 experiments/exp2_estimators.py
python3 experiments/exp2b_snn_sparse.py
python3 experiments/exp3_adjoint.py
python3 experiments/exp4_gan.py
python3 experiments/exp4c_floor.py
python3 experiments/exp5_rl.py
python3 experiments/exp5b_oracle.py
python3 experiments/exp6_graph.py
python3 experiments/exp6b_graph_synth.py
python3 experiments/fig1_traces.py
# first-revision analyses (resumable parts)
for c in noimp imp; do python3 experiments/exp7run.py a --cond $c --reps 1000; done
python3 experiments/exp7run.py astats
for g in 0 1; do python3 experiments/exp7run.py b1 --group $g; done
for g in 0 1 2; do python3 experiments/exp7run.py b2 --group $g; done
python3 experiments/exp7run.py c
python3 experiments/exp7run.py d
python3 experiments/exp7run.py merge
# second-revision analyses (resumable parts)
for c in B_noimp B_imp C_blink; do python3 experiments/exp8_r2.py a2 --cond $c --reps 1000; done
python3 experiments/exp8_r2.py a2stats
for s in 77 78 79 80 81; do
  python3 experiments/exp8_r2.py c2 --seed $s
  python3 experiments/exp8_r2.py c2x --seed $s
done
for p in "in-prior" "tau1 +30%" "tau1 -30%" "rate x0.6" "tau2 +50%" "blink 2x"; do
  python3 experiments/exp8_r2.py d2 --prior "$p"
done
python3 experiments/exp8_r2.py merge
python3 experiments/recompute_energy.py
# manuscript numbers, SI tables, and figures
python3 experiments/make_numbers.py
python3 experiments/make_numbers7.py
python3 experiments/make_numbers8.py
python3 experiments/make_supp.py
for f in figures/fig*.py figures/toc_graphic.py; do python3 "$f"; done
echo "ALL DONE"
