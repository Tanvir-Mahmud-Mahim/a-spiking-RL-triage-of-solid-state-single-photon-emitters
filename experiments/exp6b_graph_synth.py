"""Experiment 6b — platform-randomized graph transfer.

The failure mode of naive graph conditioning (exp6: two training
platforms -> the embedding degenerates into a platform ID and hurts
zero-shot) is repaired by procedurally generating synthetic emitter
platforms during training: every batch samples fresh photophysics
templates, so the encoder must learn the mapping from level-structure
graph to photon statistics rather than memorize embeddings.

Models: (i) unconditioned, trained on the synthetic platform
distribution; (ii) graph-conditioned, same data; (iii) oracle trained on
the four real platforms. Zero-shot evaluation of (i) and (ii) on
NV, hBN, GaN, SiV — none of which appear in their training.

Writes results/exp6b_graph.json (supersedes exp6 in the paper).
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/claude/sparq")
from sparq.datasets import make_batch, make_eval_set, CFG
from sparq.estimators import HistCNN, balanced_accuracy
from sparq.gnn import GraphEncoder, template_graph
from sparq.physics import Platform, PLATFORMS, sample_site

t0 = time.time()
STEPS = 1300
BATCH = 256
PLATS_PER_BATCH = 4
ALL_PLATFORMS = ["NV", "hBN", "GaN", "SiV"]
T_EVAL = [0.3, 3.0]
N_EVAL = 1200
N_SEEDS = 3
COND = 8


def synth_platform(rng):
    """Procedurally generated emitter platform spanning the physical
    space of room-temperature solid-state SPEs."""
    t1_lo = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
    t1_hi = t1_lo * float(np.exp(rng.uniform(0.4, 1.4)))
    # shelving must be slower than antibunching (physical ordering)
    t2_lo = max(3.0 * t1_hi,
                float(np.exp(rng.uniform(np.log(10.0), np.log(400.0)))))
    t2_hi = t2_lo * float(np.exp(rng.uniform(0.6, 2.0)))
    a_lo = float(rng.uniform(0.0, 0.8))
    a_hi = a_lo + float(rng.uniform(0.2, 2.0))
    r_lo = float(np.exp(rng.uniform(np.log(25.0), np.log(250.0))))
    r_hi = r_lo * float(np.exp(rng.uniform(0.8, 2.2)))
    blink = float(rng.uniform(0.0, 0.4))
    return Platform("SYN", (t1_lo, t1_hi), (t2_lo, t2_hi), (a_lo, a_hi),
                    (r_lo, r_hi), (0.55, 0.99), blink,
                    (2, 250), (0.5, 50))


def gen_synth(rng, batch):
    """Batch drawn from PLATS_PER_BATCH fresh synthetic platforms."""
    per = batch // PLATS_PER_BATCH
    parts, plats = [], []
    for _ in range(PLATS_PER_BATCH):
        p = synth_platform(rng)
        from sparq import physics
        physics.PLATFORMS["SYN"] = p        # sample_site lookup
        parts.append(make_batch(rng, per, platform="SYN"))
        plats += [p] * per
    out = {}
    for k in ("hist", "aux", "y_cls", "y_g2", "y_valid"):
        out[k] = np.concatenate([q[k] for q in parts])
    out["plats"] = plats
    return out


def gen_real(rng, batch, platforms):
    per = batch // len(platforms)
    parts = [make_batch(rng, per, platform=p) for p in platforms]
    out = {}
    for k in ("hist", "aux", "y_cls", "y_g2", "y_valid"):
        out[k] = np.concatenate([q[k] for q in parts])
    out["plats"] = sum([[PLATFORMS[p]] * per for p in platforms], [])
    return out


def train(conditioned, data, seed):
    torch.manual_seed(seed)
    net = HistCNN(CFG.n_bins, cond_dim=COND if conditioned else 0)
    enc = GraphEncoder(out_dim=COND) if conditioned else None
    params = list(net.parameters()) + (list(enc.parameters()) if enc else [])
    opt = torch.optim.Adam(params, 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(seed)
    for step in range(STEPS):
        b = gen_synth(rng, BATCH) if data == "synth" else \
            gen_real(rng, BATCH, ALL_PLATFORMS)
        hist = torch.from_numpy(b["hist"])
        aux = torch.from_numpy(b["aux"])
        y = torch.from_numpy(b["y_cls"])
        yg = torch.from_numpy(b["y_g2"])
        vv = torch.from_numpy(b["y_valid"])
        cond = None
        if conditioned:
            # one embedding per distinct platform object in the batch
            uniq = {}
            zs = []
            for p in b["plats"]:
                key = id(p)
                if key not in uniq:
                    node, edges, ef = template_graph(p)
                    uniq[key] = enc(torch.from_numpy(node),
                                    torch.from_numpy(edges),
                                    torch.from_numpy(ef))
                zs.append(uniq[key])
            cond = torch.stack(zs)
        logits, reg = net(hist, aux, cond)
        loss = F.cross_entropy(logits[vv], y[vv]) \
            + 0.5 * F.mse_loss(reg, yg)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 300 == 0:
            print(f"  step {step} loss {float(loss):.4f}", flush=True)
    net.eval()
    return net, enc


@torch.no_grad()
def eval_platform(net, enc, platform, T, seed):
    site_rng = np.random.default_rng(500 + seed)
    sites = [sample_site(site_rng, platform) for _ in range(N_EVAL)]
    r = np.random.default_rng(900 + seed)
    ev = make_eval_set(r, N_EVAL, T, platform=platform, sites=sites)
    hist = torch.from_numpy(ev["hist"])
    aux = torch.from_numpy(ev["aux"])
    cond = None
    if enc is not None:
        node, edges, ef = template_graph(platform)
        z = enc(torch.from_numpy(node), torch.from_numpy(edges),
                torch.from_numpy(ef))
        cond = z[None].repeat(N_EVAL, 1)
    logits, reg = net(hist, aux, cond)
    vv = ev["y_valid"]
    acc = balanced_accuracy(ev["y_cls"][vv], logits.argmax(1).numpy()[vv])
    mae = float(np.mean(np.abs(reg.numpy() - ev["y_g2"])))
    return acc, mae


models = {}
print("== unconditioned, synthetic-platform training")
models["uncond_syn"] = train(False, "synth", 21)
print("== graph-conditioned, synthetic-platform training")
models["graph_syn"] = train(True, "synth", 21)
print("== oracle, real platforms")
models["oracle_real"] = train(False, "real", 21)

results = {}
for name, (net, enc) in models.items():
    results[name] = {}
    for p in ALL_PLATFORMS:
        for T in T_EVAL:
            accs, maes = [], []
            for s in range(N_SEEDS):
                a, m = eval_platform(net, enc, p, T, s)
                accs.append(a); maes.append(m)
            results[name][f"{p}@{T}"] = dict(
                acc=[float(np.mean(accs)), float(np.std(accs))],
                mae=[float(np.mean(maes)), float(np.std(maes))])
    print(name, {k: round(v['acc'][0], 3) for k, v in results[name].items()},
          flush=True)

def mean_over(name, plats):
    vals = [results[name][f"{p}@{T}"]["acc"][0] for p in plats for T in T_EVAL]
    return float(np.mean(vals))

gap = ((mean_over("graph_syn", ALL_PLATFORMS)
        - mean_over("uncond_syn", ALL_PLATFORMS)) /
       max(mean_over("oracle_real", ALL_PLATFORMS)
           - mean_over("uncond_syn", ALL_PLATFORMS), 1e-9))
out = dict(results=results, steps=STEPS,
           unseen=dict(uncond=mean_over("uncond_syn", ALL_PLATFORMS),
                       graph=mean_over("graph_syn", ALL_PLATFORMS),
                       oracle=mean_over("oracle_real", ALL_PLATFORMS),
                       gap_recovery=float(gap)))
print("zero-shot summary:", out["unseen"])
with open("/home/claude/sparq/results/exp6b_graph.json", "w") as f:
    json.dump(out, f)
torch.save(models["graph_syn"][0].state_dict(),
           "/home/claude/sparq/results/models/cnn_graph_syn.pt")
print(f"saved ({time.time()-t0:.0f}s)")
