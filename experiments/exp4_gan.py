"""Experiment 4 — adversarial sim-to-real: WGAN-GP refinement of the
(pulsed-excitation) twin against real experimental HBT data
(UTS-CASLab sps-quality, FI-SEQUR InGaAs/GaAs quantum dot, 80 MHz).

Pipeline:
 1. Load the nine measurement series; calibrate the pulse comb; obtain
    each series' reference g2(0) by the conventional peak-area analysis
    of the full accumulation.
 2. Build 30-s early-estimation windows (3 x 10-s snapshots), rebinned
    onto the twin grid centered on the suppressed peak.
 3. Train a WGAN-GP whose generator refines pulsed-twin histograms so
    their distribution matches real windows (train split: 6 series).
 4. Train three g2(0) regressors: sim-only, sim + domain randomization,
    sim + GAN-refined; evaluate early estimation on the 3 held-out
    series against their asymptotic references; compare with the
    conventional peak-area analysis of the same truncated windows.

Writes results/exp4_gan.json.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/sparq")
from sparq.datasets import load_fisequr, rebin_real
from sparq.estimators import HistCNN
from sparq.physics import HBTConfig
from sparq.pulsed import (expected_hist_pulsed, calibrate_comb,
                          g2_peak_area, T_REP_NS)

CFG = HBTConfig(tau_max=45.5, n_bins=91, sigma_irf=0.35)

torch.set_num_threads(2)
t0 = time.time()
rng = np.random.default_rng(17)
torch.manual_seed(17)

HELD_OUT = {"10uW_12000cps", "2p5uW_4000cps_day1", "8uW_5100cps"}
WIN_SNAPS = 3                    # 3 x 10 s = 30 s early windows
N_GAN_ITERS = 1500
N_CRITIC = 2
EST_STEPS = 1300
BATCH = 128

# ---------------------------------------------------------------- real data
series = load_fisequr()
real = []
for s in series:
    d, tot = s["delay"], s["total"]
    center, phase, P = calibrate_comb(d, tot)
    g2_ref = g2_peak_area(d, tot, center)
    n_win = s["counts"].shape[1] // WIN_SNAPS
    wins, wins_raw = [], []
    for w in range(n_win):
        h = s["counts"][:, w * WIN_SNAPS:(w + 1) * WIN_SNAPS].sum(1)
        hw, _ = rebin_real(d, h, CFG, center=center)
        wins.append(hw)
        wins_raw.append(h)
    real.append(dict(name=s["name"], g2_ref=float(g2_ref), center=center,
                     delay=d, windows=np.array(wins, np.float32),
                     windows_raw=np.array(wins_raw, np.float32),
                     T_win=WIN_SNAPS * 10.0, T_tot=s["T_total"],
                     held_out=s["name"] in HELD_OUT))
    print(f"{s['name'][:40]:42s} g2_ref={g2_ref:.3f} center={center:.1f} "
          f"windows={n_win} held_out={s['name'] in HELD_OUT} "
          f"cts/win={np.mean([w.sum() for w in wins]):.0f}")

train_wins = np.concatenate([r["windows"] for r in real if not r["held_out"]])
print("real training windows:", train_wins.shape)

# ---------------------------------------------------------------- sim domain
# pulsed-twin prior spanning the platform's published photophysics
def sample_qd_params(rng):
    return dict(
        rate_cps=float(np.exp(rng.uniform(np.log(2e3), np.log(1.8e4)))),
        g2_0=float(rng.uniform(0.02, 0.95)),
        tau_e=float(np.exp(rng.uniform(np.log(0.7), np.log(4.0)))),
        a=float(rng.uniform(0.0, 0.6)),
        tau2=float(np.exp(rng.uniform(np.log(40.0), np.log(500.0)))),
        bg_frac=float(rng.uniform(0.0, 0.45)),
        sigma_pair=float(rng.uniform(0.3, 0.9)),
        center_off=float(rng.uniform(-0.5, 0.5)),
    )


def sim_batch(rng, batch, T=30.0, distort=False):
    H = np.zeros((batch, CFG.n_bins), np.float32)
    g2 = np.zeros(batch, np.float32)
    for i in range(batch):
        p = sample_qd_params(rng)
        mu = expected_hist_pulsed(CFG, T, **p)
        if distort:   # generic domain randomization
            tilt = 1.0 + rng.uniform(-0.15, 0.15) * \
                (CFG.bin_centers / CFG.tau_max)
            bump = 1.0 + rng.uniform(0, 0.3) * np.exp(
                -np.abs(np.abs(CFG.bin_centers) - rng.uniform(20, 45)) / 6.0)
            mu = mu * tilt * bump
        H[i] = rng.poisson(np.maximum(mu, 0))
        g2[i] = p["g2_0"]
    return H, g2


def make_aux(H, T):
    """Features computable identically for sim and real windows."""
    k = np.abs(CFG.bin_centers) < 4.0
    return np.stack([np.full(len(H), np.log10(T)),
                     np.log10(1.0 + H.sum(1)),
                     np.log10(1.0 + H[:, k].sum(1))], 1).astype(np.float32)

# ---------------------------------------------------------------- WGAN-GP
class Gen(nn.Module):
    """Residual refiner in log space, zero-initialized to the identity."""

    def __init__(self, k=91, z=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k + z, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, k))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.z = z

    def forward(self, h_log, z):
        return F.relu(h_log + self.net(torch.cat([h_log, z], 1)))


class Critic(nn.Module):
    def __init__(self, k=91):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)


G, D = Gen(CFG.n_bins), Critic(CFG.n_bins)
optG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.9))
optD = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.9))
real_t = torch.log1p(torch.from_numpy(train_wins))

def gp(D, xr, xf):
    e = torch.rand(xr.shape[0], 1)
    xh = (e * xr + (1 - e) * xf).requires_grad_(True)
    d = D(xh)
    g = torch.autograd.grad(d.sum(), xh, create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()

print("== WGAN-GP training")
w_dist_log = []
for it in range(N_GAN_ITERS):
    for _ in range(N_CRITIC):
        idx = rng.integers(0, len(real_t), BATCH)
        xr = real_t[idx]
        Hs, _ = sim_batch(rng, BATCH, T=30.0)
        hlog = torch.log1p(torch.from_numpy(Hs))
        z = torch.randn(BATCH, G.z)
        with torch.no_grad():
            xf = G(hlog, z)                  # refined, log space
        loss_d = D(xf).mean() - D(xr).mean() + 10.0 * gp(D, xr, xf)
        optD.zero_grad(); loss_d.backward(); optD.step()
    Hs, _ = sim_batch(rng, BATCH, T=30.0)
    hlog = torch.log1p(torch.from_numpy(Hs))
    z = torch.randn(BATCH, G.z)
    xf = G(hlog, z)
    loss_g = -D(xf).mean()
    optG.zero_grad(); loss_g.backward(); optG.step()
    if it % 150 == 0:
        with torch.no_grad():
            idx = rng.integers(0, len(real_t), 512)
            Hs, _ = sim_batch(rng, 512, T=30.0)
            xs = torch.log1p(torch.from_numpy(Hs))
            xf = G(xs, torch.randn(512, G.z))
            w_sim = float(D(real_t[idx]).mean() - D(xs).mean())
            w_gan = float(D(real_t[idx]).mean() - D(xf).mean())
        w_dist_log.append(dict(it=it, w_sim=w_sim, w_gan=w_gan))
        print(f"  it {it}: critic gap sim {w_sim:.3f} -> refined {w_gan:.3f}")

# ---------------------------------------------------------------- estimators
def train_estimator(mode, seed=5):
    torch.manual_seed(seed)
    net = HistCNN(CFG.n_bins, n_aux=3)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    r = np.random.default_rng(seed)
    for step in range(EST_STEPS):
        H, g2 = sim_batch(r, BATCH, T=30.0,
                          distort=(mode == "dr" and step % 2 == 0))
        Ht = torch.from_numpy(H)
        if mode == "gan" and step % 2 == 0:
            with torch.no_grad():
                Ht = torch.expm1(
                    G(torch.log1p(Ht), torch.randn(len(H), G.z)))
        aux = torch.from_numpy(make_aux(Ht.numpy(), 30.0))
        _, reg = net(Ht, aux)
        loss = F.mse_loss(reg, torch.from_numpy(g2))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def eval_real(net, subset):
    errs, per_series = [], {}
    for rr_ in real:
        if rr_["held_out"] != subset:
            continue
        H = torch.from_numpy(rr_["windows"])
        aux = torch.from_numpy(make_aux(rr_["windows"], rr_["T_win"]))
        _, reg = net(H, aux)
        e = np.abs(reg.numpy() - rr_["g2_ref"])
        errs.append(e)
        per_series[rr_["name"]] = dict(mae=float(e.mean()),
                                       bias=float(np.mean(
                                           reg.numpy() - rr_["g2_ref"])),
                                       n=len(e))
    return float(np.concatenate(errs).mean()), per_series


nets = {m: train_estimator(m) for m in ("sim", "dr", "gan")}
res = {}
for m, net in nets.items():
    mae_ho, per = eval_real(net, subset=True)
    mae_tr, _ = eval_real(net, subset=False)
    res[m] = dict(mae_held_out=mae_ho, mae_train_series=mae_tr,
                  per_series=per)
    print(f"{m}: held-out MAE {mae_ho:.4f}, train-series MAE {mae_tr:.4f}")

# conventional peak-area analysis on the same held-out windows
fit_errs, n_fail = [], 0
for rr_ in real:
    if not rr_["held_out"]:
        continue
    for h in rr_["windows_raw"][:120]:
        g2h = g2_peak_area(rr_["delay"], h, rr_["center"])
        if not np.isfinite(g2h):
            n_fail += 1
            g2h = 1.0
        fit_errs.append(abs(np.clip(g2h, 0, 3) - rr_["g2_ref"]))
res["fit"] = dict(mae_held_out=float(np.mean(fit_errs)), n=len(fit_errs),
                  n_fail=n_fail)
print("peak-area 30-s windows: MAE", res["fit"]["mae_held_out"])

gap_closed = ((res["sim"]["mae_held_out"] - res["gan"]["mae_held_out"]) /
              max(res["sim"]["mae_held_out"], 1e-9))
out = dict(series=[dict(name=r_["name"], g2_ref=r_["g2_ref"],
                        held_out=r_["held_out"], T_tot=r_["T_tot"],
                        n_windows=len(r_["windows"])) for r_ in real],
           results=res, wgan_log=w_dist_log, gap_closed=float(gap_closed),
           win_s=WIN_SNAPS * 10.0)
with open("/home/claude/sparq/results/exp4_gan.json", "w") as f:
    json.dump(out, f)
torch.save(G.state_dict(), "/home/claude/sparq/results/models/gan_G.pt")
print(f"saved ({time.time()-t0:.0f}s)")
