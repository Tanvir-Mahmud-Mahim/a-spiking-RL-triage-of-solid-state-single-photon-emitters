"""Experiment 3 — adjoint (pathwise-gradient) co-optimization of the
measurement protocol through the differentiable twin.

The protocol theta = (log saturation parameter s, log window tau_max) is
optimized jointly with the estimator by backpropagating the task loss
through the stochastic physics (reparameterized Poisson).  Compared
against the default protocol (s = 1, tau_max = 60.5 ns) with an equally
trained estimator.  A numerical Fisher-information sweep shows the
adjoint optimum tracks the information-theoretic optimum of the physics.

Writes results/exp3_adjoint.json.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/sparq")
from sparq.physics import sample_site
from sparq.twin_torch import (torch_expected_hist, reparam_counts,
                              base_tensors, fisher_info_g2zero)
from sparq.estimators import HistCNN, balanced_accuracy

t0 = time.time()
rng = np.random.default_rng(42)
gen = torch.Generator().manual_seed(42)

STEPS = 900
BATCH = 192
T_TRAIN = ("logu", 0.03, 3.0)      # protocol matters most when starved
N_BINS = 121


def sample_bases(rng, batch):
    sites = [sample_site(rng, "NV") for _ in range(batch)]
    y = np.array([1 if s.g2_0 < 0.5 else 0 for s in sites], np.int64)
    g2 = np.array([s.g2_0 for s in sites], np.float32)
    v = np.abs(g2 - 0.5) > 0.1
    return (base_tensors(sites), torch.from_numpy(y), torch.from_numpy(g2),
            torch.from_numpy(v))


def make_inputs(base, th_s, th_w, T, gen):
    mu = torch_expected_hist(th_s, th_w, base, T, N_BINS)
    h = reparam_counts(mu, gen)
    r_tot = (base["rate_kcps"] * 1e3)[:, None]
    width = 2.0 * torch.exp(th_w) / N_BINS
    exp_flat = (0.5 * r_tot) ** 2 * (width * 1e-9) * T * N_BINS
    k = torch.arange(N_BINS, dtype=torch.float32)
    centers = ((k + 0.5) / N_BINS * 2.0 - 1.0)[None, :] * torch.exp(th_w)
    central = (h * (centers.abs() < 12.0)).sum(1, keepdim=True)
    aux = torch.cat([
        torch.full((h.shape[0], 1), float(np.log10(T))),
        torch.log10(r_tot),
        torch.log10(1.0 + h.sum(1, keepdim=True)),
        torch.log10(1.0 + exp_flat),
        torch.log10(1.0 + central),
    ], 1)
    return h, aux


def train(protocol_free, seed):
    torch.manual_seed(seed)
    net = HistCNN(N_BINS)
    th_s = torch.tensor(0.0, requires_grad=protocol_free)
    th_w = torch.tensor(float(np.log(60.5)), requires_grad=protocol_free)
    opt = torch.optim.Adam([
        {"params": net.parameters(), "lr": 1e-3},
        {"params": [th_s, th_w], "lr": 3e-2},
    ] if protocol_free else [{"params": net.parameters(), "lr": 1e-3}])
    r = np.random.default_rng(seed)
    traj = []
    for step in range(STEPS):
        base, y, g2, v = sample_bases(r, BATCH)
        lo, hi = T_TRAIN[1], T_TRAIN[2]
        T = float(np.exp(r.uniform(np.log(lo), np.log(hi))))
        h, aux = make_inputs(base, th_s, th_w, T, gen)
        logits, reg = net(h, aux)
        loss = F.cross_entropy(logits[v], y[v]) + 0.5 * F.mse_loss(reg, g2)
        # keep the window physical
        if protocol_free:
            loss = loss + 0.1 * (torch.relu(th_w - np.log(120.0)) ** 2
                                 + torch.relu(np.log(15.0) - th_w) ** 2
                                 + torch.relu(th_s - np.log(8.0)) ** 2
                                 + torch.relu(np.log(1 / 8.0) - th_s) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 150 == 0:
            traj.append(dict(step=step, s=float(torch.exp(th_s)),
                             tau_max=float(torch.exp(th_w)),
                             loss=float(loss)))
            print(f"  step {step}: s={traj[-1]['s']:.2f}, "
                  f"w={traj[-1]['tau_max']:.1f}, loss={float(loss):.3f}")
    return net, th_s.detach(), th_w.detach(), traj


@torch.no_grad()
def eval_protocol(net, th_s, th_w, T, n=2000, seed=7):
    r = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    base, y, g2, v = sample_bases(r, n)
    h, aux = make_inputs(base, th_s, th_w, T, g)
    logits, reg = net(h, aux)
    pred = logits.argmax(1).numpy()
    v = v.numpy()
    return (balanced_accuracy(y.numpy()[v], pred[v]),
            float(np.mean(np.abs(reg.numpy() - g2.numpy()))))


print("== default protocol (fixed s=1, tau_max=60.5)")
net0, s0, w0, _ = train(False, seed=100)
print("== adjoint-optimized protocol")
net1, s1, w1, traj = train(True, seed=100)
print(f"optimized: s* = {float(torch.exp(s1)):.2f}, "
      f"tau_max* = {float(torch.exp(w1)):.1f} ns")

evals = {}
for T in (0.03, 0.1, 0.3, 1.0):
    a0, m0 = eval_protocol(net0, s0, w0, T)
    a1, m1 = eval_protocol(net1, s1, w1, T)
    evals[str(T)] = dict(default=dict(acc=a0, mae=m0),
                         adjoint=dict(acc=a1, mae=m1))
    print(f"T={T}: default {a0:.3f}/{m0:.3f}  adjoint {a1:.3f}/{m1:.3f}")

# Fisher-information sweep over s at the population mean site
r = np.random.default_rng(3)
sites = [sample_site(r, "NV") for _ in range(400)]
base_mean = base_tensors(sites)
s_grid = np.geomspace(1 / 8, 8, 25)
fi = [fisher_info_g2zero(s, base_mean) / len(sites) for s in s_grid]
s_star_fi = float(s_grid[int(np.argmax(fi))])
print(f"FI-optimal s = {s_star_fi:.2f}")

out = dict(steps=STEPS, evals=evals, trajectory=traj,
           s_star=float(torch.exp(s1)), tau_max_star=float(torch.exp(w1)),
           fisher=dict(s_grid=s_grid.tolist(), fi=fi, s_star=s_star_fi))
with open("/home/claude/sparq/results/exp3_adjoint.json", "w") as f:
    json.dump(out, f)
print(f"saved ({time.time()-t0:.0f}s)")
