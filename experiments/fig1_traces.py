"""Anytime outputs of the trained spiking estimator (snn_sparse.pt) for
two simulated NV-prior sites, used in main-text Figure 1(b).
Writes results/fig1_traces.json."""
import os as _os
_ROOT = _os.environ.get("SPARQ_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, sys
import numpy as np
import torch
sys.path.insert(0, _ROOT)
from sparq.physics import EmitterSite
from sparq.datasets import make_batch, CFG, N_SLICES
from sparq.estimators import SpikingG2Net

snn = SpikingG2Net(CFG.n_bins)
snn.load_state_dict(torch.load("results/models/snn_sparse.pt",
                               map_location="cpu"))
snn.eval()
base = dict(tau1=14.0, tau2=250.0, a=0.8, blinking=False, t_on_ms=1,
            t_off_ms=1, platform="NV")
cases = {
    "single": EmitterSite(dict(base, rate_kcps=200, rho=0.95), 1),
    "multi": EmitterSite(dict(base, rate_kcps=200, rho=0.80), 3),
}
out = {"slice_ms": 1000.0 / N_SLICES, "T_s": 1.0}
for name, site in cases.items():
    rng = np.random.default_rng(2025)
    b = make_batch(rng, 5, sites=[site] * 5, T_fixed=1.0)
    with torch.no_grad():
        logits, _ = snn(torch.from_numpy(b["stream"]),
                        torch.from_numpy(b["aux"]))
        p = torch.softmax(logits, -1)[..., 1].numpy()
    out[name] = dict(g2_0=float(site.g2_0), p_pure=p.tolist(),
                     coinc_per_slice=float(b["stream"].sum((1, 2)).mean()
                                           / N_SLICES))
    print(name, "g2(0)=%.3f" % site.g2_0, "final p:",
          np.round(p[:, -1], 3), "coinc/slice %.1f"
          % out[name]["coinc_per_slice"])
json.dump(out, open("results/fig1_traces.json", "w"), indent=1)
