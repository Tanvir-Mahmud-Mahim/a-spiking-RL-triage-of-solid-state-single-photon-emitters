"""Re-price the dense-baseline energy entries of the stored results with
the per-operation energies of Horowitz (ISSCC 2014, 45-nm CMOS):
  32-bit floating-point MAC = 3.7 pJ (multiply) + 0.9 pJ (add) = 4.6 pJ
   8-bit integer MAC        = 0.2 pJ (multiply) + 0.03 pJ (add) = 0.23 pJ
The 8-bit value used in the original submission (1 pJ) had no source in
the cited reference and is replaced here. The spiking-network entries
(measured synaptic operations x 23.6 pJ, the minimum energy per synaptic
spike operation reported for Loihi, Davies et al. 2018) are unchanged.
The dense multiply-accumulate count is an exact property of the network
(HistCNN.macs_per_inference) and does not depend on training.
Rewrites the "energy" block of results/exp2b_snn.json and
results/exp2_estimators.json in place."""
import json

E_SYNOP = 23.6e-12
E_MAC_FP32 = 4.6e-12
E_MAC_INT8 = 0.23e-12

for fn in ("results/exp2b_snn.json", "results/exp2_estimators.json"):
    d = json.load(open(fn))
    for T, e in d["energy"].items():
        macs = e["macs_cnn"]
        e_snn = e["synops_mean"] * E_SYNOP
        e["e_snn_nJ"] = e_snn * 1e9
        e["e_cnn_fp32_nJ"] = macs * E_MAC_FP32 * 1e9
        e["e_cnn_int8_nJ"] = macs * E_MAC_INT8 * 1e9
        e["adv_fp32"] = macs * E_MAC_FP32 / e_snn
        e["adv_int8"] = macs * E_MAC_INT8 / e_snn
    d["energy_constants_J"] = dict(synop=E_SYNOP, mac_fp32=E_MAC_FP32,
                                   mac_int8=E_MAC_INT8)
    json.dump(d, open(fn, "w"))
    print(fn, {T: (round(e["e_snn_nJ"], 1), round(e["e_cnn_fp32_nJ"], 1),
                   round(e["e_cnn_int8_nJ"], 1)) for T, e in
               d["energy"].items()})
