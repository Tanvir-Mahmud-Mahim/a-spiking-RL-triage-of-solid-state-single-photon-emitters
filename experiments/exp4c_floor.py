"""Exp 4c — information-floor decomposition of the sim-to-real error:
in-domain twin MAE at matched 30-s count statistics, restricted to the
g2(0) range of the held-out real series. Writes results/exp4c_floor.json."""
import sys, json
sys.path.insert(0, "/home/claude/sparq")
import numpy as np, torch
import torch.nn.functional as F
from sparq.physics import HBTConfig
from sparq.pulsed import expected_hist_pulsed
from sparq.estimators import HistCNN

CFG = HBTConfig(tau_max=45.5, n_bins=91, sigma_irf=0.35)
torch.manual_seed(7)

def sample_qd_params(rng):
    return dict(rate_cps=float(np.exp(rng.uniform(np.log(2e3), np.log(1.8e4)))),
                g2_0=float(rng.uniform(0.02, 0.95)),
                tau_e=float(np.exp(rng.uniform(np.log(0.7), np.log(4.0)))),
                a=float(rng.uniform(0.0, 0.6)),
                tau2=float(np.exp(rng.uniform(np.log(40.0), np.log(500.0)))),
                bg_frac=float(rng.uniform(0.0, 0.45)),
                sigma_pair=float(rng.uniform(0.3, 0.9)),
                center_off=float(rng.uniform(-0.5, 0.5)))

def sim_batch(rng, batch, T=30.0):
    H = np.zeros((batch, CFG.n_bins), np.float32); g2 = np.zeros(batch, np.float32)
    for i in range(batch):
        p = sample_qd_params(rng)
        H[i] = rng.poisson(np.maximum(expected_hist_pulsed(CFG, T, **p), 0))
        g2[i] = p["g2_0"]
    return H, g2

def make_aux(H, T):
    k = np.abs(CFG.bin_centers) < 4.0
    return np.stack([np.full(len(H), np.log10(T)), np.log10(1.0+H.sum(1)),
                     np.log10(1.0+H[:,k].sum(1))],1).astype(np.float32)

net = HistCNN(CFG.n_bins, n_aux=3)
opt = torch.optim.Adam(net.parameters(), 1e-3)
r = np.random.default_rng(5)
for step in range(900):
    H, g2 = sim_batch(r, 128)
    _, reg = net(torch.from_numpy(H), torch.from_numpy(make_aux(H, 30.0)))
    loss = F.mse_loss(reg, torch.from_numpy(g2))
    opt.zero_grad(); loss.backward(); opt.step()
net.eval()
errs = []
with torch.no_grad():
    for _ in range(20):
        H, g2 = sim_batch(r, 512)
        _, reg = net(torch.from_numpy(H), torch.from_numpy(make_aux(H, 30.0)))
        e = np.abs(reg.numpy()-g2)
        m = (g2 > 0.30) & (g2 < 0.56)
        errs.append(e[m])
errs = np.concatenate(errs)
out = dict(floor_mae=float(errs.mean()), n=len(errs))
with open("/home/claude/sparq/results/exp4c_floor.json", "w") as f:
    json.dump(out, f)
print(out)
