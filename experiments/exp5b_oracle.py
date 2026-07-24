"""Exp 5b — clairvoyant (oracle-stopping) bound for per-site triage.

For each site, expose in 0.25-s steps and stop at the FIRST moment the
running posterior agrees with the ground truth (cap 5 s); decide
accordingly. No realizable controller can triage faster at equal
quality using the same posterior, so this bounds the entire
policy family (heuristics and learned agents alike).

Writes results/exp5b_oracle.json using the same evaluation fields and
noise seeds as exp5.
"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, "/home/claude/sparq")
from sparq.estimators import TriageCNN
from sparq.datasets import CFG
from sparq.rl_env import TriageEnv, A_MEAS0, A_REJECT, A_CERTIFY

est = TriageCNN(CFG.n_bins)
est.load_state_dict(torch.load(
    "/home/claude/sparq/results/models/cnn_triage.pt"))
est.eval()

field_rng = np.random.default_rng(4242)
env = TriageEnv(est, seed=999)
EVAL_FIELDS = [env.new_field(field_rng) for _ in range(30)]


def run_oracle(env, field, noise_rng, T_cap=5.0):
    env.reset(field=field, noise_rng=noise_rng)
    while not env.done:
        truth = env.field[env.idx].is_good
        agrees = (env.p_good > 0.5) == truth
        if (env.dwell >= 0.25 and agrees) or env.dwell >= T_cap:
            env.step(A_CERTIFY if env.p_good > 0.5 else A_REJECT)
        else:
            env.step(A_MEAS0)
    return env.summary()


outs = [run_oracle(env, f, np.random.default_rng(7000 + i))
        for i, f in enumerate(EVAL_FIELDS)]
keys = ("time_s", "precision", "recall", "good_per_min", "n_false")
summary = {k: [float(np.mean([o[k] for o in outs])),
               float(np.std([o[k] for o in outs]))] for k in keys}
print("oracle stopping:", {k: round(v[0], 3) for k, v in summary.items()})
with open("/home/claude/sparq/results/exp5b_oracle.json", "w") as f:
    json.dump(summary, f)
