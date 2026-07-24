"""Experiment 5 — closed-loop triage with discrete-action SAC + PER.

The agent triages 48-site confocal fields inside the validated twin,
perceiving the physics-in-the-loop estimator's posterior and choosing
dwell/reject/certify actions.  Compared against fixed-dwell raster
policies and an uncertainty-gated heuristic; a PER-vs-uniform replay
ablation quantifies the value of prioritization.

Writes results/exp5_rl.json.
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/claude/sparq")
from sparq.estimators import HistCNN
from sparq.datasets import CFG
from sparq.rl_env import (TriageEnv, run_raster, run_adaptive_heuristic,
                          run_policy, N_ACTIONS, OBS_DIM)
from sparq.sac_per import DiscreteSAC

t0 = time.time()
TOTAL_STEPS = 40_000
UPDATE_EVERY = 2
WARMUP = 2_000
EVAL_EVERY = 4_000
N_EVAL_FIELDS = 30
SEEDS = (0, 1)

import os
from sparq.estimators import TriageCNN, train_triage

EST_PATH = "/home/claude/sparq/results/models/cnn_triage.pt"
if os.path.exists(EST_PATH):
    est = TriageCNN(CFG.n_bins)
    est.load_state_dict(torch.load(EST_PATH))
    est.eval()
else:
    print("== training triage estimator (purity + good + regression)")
    est = train_triage(steps=1500)
    torch.save(est.state_dict(), EST_PATH)

# fixed evaluation fields (shared across all policies)
field_rng = np.random.default_rng(4242)
env_proto = TriageEnv(est, seed=999)
EVAL_FIELDS = [env_proto.new_field(field_rng) for _ in range(N_EVAL_FIELDS)]


def eval_agent(agent, seed=0):
    env = TriageEnv(est, seed=seed)
    outs = []
    for i, f in enumerate(EVAL_FIELDS):
        nr = np.random.default_rng(7000 + i)
        outs.append(run_policy(env, f, agent, nr))
    return summarize(outs)


def summarize(outs):
    keys = ("time_s", "precision", "recall", "good_per_min", "n_false")
    return {k: [float(np.mean([o[k] for o in outs])),
                float(np.std([o[k] for o in outs]))] for k in keys}


def train_sac(per, seed):
    agent = DiscreteSAC(OBS_DIM, N_ACTIONS, per=per, seed=seed)
    env = TriageEnv(est, seed=seed)
    obs = env.reset()
    curve = []
    ep_ret, ep_rets = 0.0, []
    for step in range(TOTAL_STEPS):
        if step < WARMUP:
            a = agent.rng.integers(0, N_ACTIONS)
        else:
            a = agent.act(obs)
        nobs, r, done, _ = env.step(a)
        agent.buf.push((obs, a, r, nobs, float(done)))
        ep_ret += r
        obs = nobs
        if done:
            ep_rets.append(ep_ret)
            ep_ret = 0.0
            obs = env.reset()
        if step >= WARMUP and step % UPDATE_EVERY == 0:
            agent.update()
        if (step + 1) % EVAL_EVERY == 0:
            s = eval_agent(agent, seed=seed)
            curve.append(dict(step=step + 1, **{k: v[0] for k, v in s.items()},
                              train_ret=float(np.mean(ep_rets[-20:]))
                              if ep_rets else 0.0))
            print(f"  [{'PER' if per else 'UNI'} s{seed}] step {step+1}: "
                  f"t={s['time_s'][0]:.0f}s P={s['precision'][0]:.3f} "
                  f"R={s['recall'][0]:.3f} g/min={s['good_per_min'][0]:.2f}")
    return agent, curve


# ---------------------------------------------------------------- baselines
env = TriageEnv(est, seed=1)
baselines = {}
for T_fix in (0.5, 1.0, 2.0, 4.0, 8.0):
    outs = [run_raster(env, f, T_fix, np.random.default_rng(7000 + i))
            for i, f in enumerate(EVAL_FIELDS)]
    baselines[f"raster_{T_fix}"] = summarize(outs)
    print(f"raster {T_fix}s:", {k: round(v[0], 3)
                                for k, v in baselines[f'raster_{T_fix}'].items()})
for margin in (0.8, 0.9, 0.95):
    outs = [run_adaptive_heuristic(env, f, np.random.default_rng(7000 + i),
                                   margin=margin)
            for i, f in enumerate(EVAL_FIELDS)]
    baselines[f"heuristic_{margin}"] = summarize(outs)
    print(f"heuristic m={margin}:", {k: round(v[0], 3)
                                     for k, v in baselines[f'heuristic_{margin}'].items()})

# ---------------------------------------------------------------- training
curves = {"per": [], "uniform": []}
final = {"per": [], "uniform": []}
agents = {}
for per in (True, False):
    key = "per" if per else "uniform"
    for seed in SEEDS:
        print(f"== SAC {'with PER' if per else 'uniform replay'}, seed {seed}")
        agent, curve = train_sac(per, seed)
        curves[key].append(curve)
        final[key].append(eval_agent(agent, seed=seed))
        agents[(key, seed)] = agent

# best PER agent by good_per_min
best_i = int(np.argmax([f["good_per_min"][0] for f in final["per"]]))
best_agent = agents[("per", SEEDS[best_i])]
best = final["per"][best_i]
print("best SAC-PER:", {k: round(v[0], 3) for k, v in best.items()})

# ------------------------------------------- dwell-allocation behavior
dwell_cert, dwell_rej = [], []
env_b = TriageEnv(est, seed=5)
for i, f in enumerate(EVAL_FIELDS):
    obs = env_b.reset(field=f, noise_rng=np.random.default_rng(8000 + i))
    site_dwell = 0.0
    while not env_b.done:
        idx_before = env_b.idx
        d_before = env_b.dwell
        a = best_agent.act(obs, greedy=True)
        obs, _, _, _ = env_b.step(a)
        if env_b.done or env_b.idx != idx_before:      # site concluded
            (dwell_cert if (env_b.certified and
                            env_b.certified[-1][0] == idx_before)
             else dwell_rej).append(d_before if a in (3, 4) else d_before)
ratio = (np.mean(dwell_cert) / max(np.mean(dwell_rej), 1e-9)
         if dwell_cert and dwell_rej else float("nan"))
with open("/home/claude/sparq/results/exp5_dwell.json", "w") as f:
    json.dump(dict(ratio=float(ratio),
                   dwell_cert=[float(x) for x in dwell_cert],
                   dwell_rej=[float(x) for x in dwell_rej]), f)
print(f"dwell certify/reject ratio: {ratio:.2f} "
      f"({np.mean(dwell_cert):.2f}s vs {np.mean(dwell_rej):.2f}s)")

out = dict(total_steps=TOTAL_STEPS, n_eval_fields=N_EVAL_FIELDS,
           baselines=baselines, curves=curves,
           final={k: v for k, v in final.items()},
           best_per=best,
           p_good_field=float(np.mean(
               [s.is_good for f in EVAL_FIELDS for s in f])))
with open("/home/claude/sparq/results/exp5_rl.json", "w") as f:
    json.dump(out, f)
torch.save(best_agent.pi.state_dict(),
           "/home/claude/sparq/results/models/sac_pi.pt")
print(f"saved ({time.time()-t0:.0f}s)")
