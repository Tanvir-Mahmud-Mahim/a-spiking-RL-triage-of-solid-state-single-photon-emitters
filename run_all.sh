#!/bin/bash
# Run the full SPARQ experiment suite in order.
set -e
cd "$(dirname "$0")"
mkdir -p results
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
python3 experiments/make_numbers.py
python3 experiments/make_supp.py
echo "ALL DONE"
