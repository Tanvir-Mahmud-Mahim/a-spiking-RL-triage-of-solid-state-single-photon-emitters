"""Generate paper/numbers.tex (macros) and paper/supp_tables.tex from the
experiment result JSONs — every number in the manuscript originates here."""
import json, sys
import numpy as np

R = "/home/claude/sparq/results"
P = "/home/claude/sparq/paper"


def load(name):
    with open(f"{R}/{name}") as f:
        return json.load(f)


def fmt(x, d=1):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100*x:.{d}f}"


macros = {}

# ------------------------------------------------------------------ exp1
e1 = load("exp1_validation.json")
chis = {k: v["chi2_red"] for k, v in e1["mc_vs_exact"].items()}
macros["valChiNv"] = fmt(chis["NV-like"], 2)
macros["valChiHbn"] = fmt(chis["hBN-like"], 2)
macros["valChiGan"] = fmt(chis["GaN-like"], 2)
macros["valChiMin"] = fmt(min(chis.values()), 2)
macros["valChiMax"] = fmt(max(chis.values()), 2)
macros["valNrmseMax"] = fmt(100 * e1["summary"]["max_nrmse"], 1)
macros["valTwinErr"] = fmt(e1["summary"]["rel_mean_err_pct"], 1)
macros["valFano"] = fmt(e1["summary"]["fano"], 2)

# ------------------------------------------------------------------ exp2
e2 = load("exp2_estimators.json")
T = e2["T_grid"]
iT = {t: i for i, t in enumerate(T)}
acc = lambda m, t: e2["results"][m]["acc"][iT[t]][0]
mae = lambda m, t: e2["results"][m]["mae"][iT[t]][0]
macros["targetAcc"] = fmt(100 * e2["target_acc"], 0)
macros["pGoodPct"] = fmt(100 * e2["p_good"], 0)
for name, key in (("Fit", "fit"), ("Cnn", "cnn_pitl"),
                  ("CnnClean", "cnn_clean"), ("Snn", "snn_pitl")):
    macros[f"acc{name}PointOThree"] = pct(acc(key, 0.03))
    macros[f"acc{name}PointOne"] = pct(acc(key, 0.1))
    macros[f"acc{name}One"] = pct(acc(key, 1.0))
    macros[f"acc{name}Asym"] = pct(acc(key, 30.0))
    macros[f"mae{name}One"] = fmt(mae(key, 1.0), 3)
    ttt = e2["time_to_target"][key]
    macros[f"ttt{name}"] = (fmt(ttt, 2) if ttt == ttt else "$>$30")
macros["cleanDropPointOne"] = fmt(
    100 * (acc("cnn_pitl", 0.1) - acc("cnn_clean", 0.1)), 1)
sp_snn = e2["time_to_target"]["fit"] / e2["time_to_target"]["snn_pitl"]
sp_cnn = e2["time_to_target"]["fit"] / e2["time_to_target"]["cnn_pitl"]
macros["speedupSnn"] = fmt(sp_snn, 0)
macros["speedupCnn"] = fmt(sp_cnn, 0)
any90 = e2["anytime"]["0.9"]
macros["anyMedianNinety"] = fmt(any90["median_ms"], 0)
macros["anyAccNinety"] = pct(any90["acc"])
en1 = e2["energy"]["1.0"]
macros["synopsOneK"] = fmt(en1["synops_mean"] / 1e3, 1)
macros["eSnnOneT"] = fmt(en1["e_snn_nJ"], 1)
macros["eCnnFpT"] = fmt(en1["e_cnn_fp32_nJ"] / 1e3, 2)
macros["eCnnIntT"] = fmt(en1["e_cnn_int8_nJ"] / 1e3, 2)
macros["advFpX"] = fmt(en1["adv_fp32"], 0)
macros["advIntX"] = fmt(en1["adv_int8"], 0)

# ------------------------------------------------------------------ exp2b
# activity-regularized SNN supersedes the exp2 SNN rows
try:
    e2b = load("exp2b_snn.json")
    accs = e2b["snn_sparse"]["acc"]
    maes = e2b["snn_sparse"]["mae"]
    macros["accSnnPointOThree"] = pct(accs[iT[0.03]][0])
    macros["accSnnPointOne"] = pct(accs[iT[0.1]][0])
    macros["accSnnOne"] = pct(accs[iT[1.0]][0])
    macros["accSnnAsym"] = pct(accs[iT[30.0]][0])
    macros["maeSnnOne"] = fmt(maes[iT[1.0]][0], 3)
    macros["tttSnn"] = fmt(e2b["time_to_target"], 2)
    macros["speedupSnn"] = fmt(
        e2["time_to_target"]["fit"] / e2b["time_to_target"], 0)
    a50 = e2b["anytime"]["0.5"]
    a60 = e2b["anytime"]["0.6"]
    a90 = e2b["anytime"]["0.9"]
    macros["anyMedianHalf"] = fmt(a50["median_ms"], 0)
    macros["anyAccHalf"] = pct(a50["acc"])
    macros["anyMedianSixty"] = fmt(a60["median_ms"], 0)
    macros["anyAccSixty"] = pct(a60["acc"])
    macros["anyMedianNinety"] = fmt(a90["median_ms"], 0)
    macros["anyAccNinety"] = pct(a90["acc"])
    en1 = e2b["energy"]["1.0"]
    enp1 = e2b["energy"]["0.1"]
    macros["synopsOneK"] = fmt(en1["synops_mean"] / 1e3, 1)
    macros["eSnnOneT"] = fmt(en1["e_snn_nJ"], 0)
    macros["eSnnOneTuJ"] = fmt(en1["e_snn_nJ"] / 1e3, 1)
    macros["eSnnPointOneT"] = fmt(enp1["e_snn_nJ"], 0)
    macros["eCnnFpT"] = fmt(en1["e_cnn_fp32_nJ"] / 1e3, 2)
    macros["eCnnIntT"] = fmt(en1["e_cnn_int8_nJ"] / 1e3, 2)
    macros["advFpX"] = fmt(en1["adv_fp32"], 1)
    macros["advIntX"] = fmt(en1["adv_int8"], 1)
    macros["advFpPointOneX"] = fmt(enp1["adv_fp32"], 0)
    macros["advIntPointOneX"] = fmt(enp1["adv_int8"], 1)
except FileNotFoundError:
    print("exp2b missing")

# ------------------------------------------------------------------ exp3
e3 = load("exp3_adjoint.json")
macros["sStar"] = fmt(e3["s_star"], 2)
macros["wStar"] = fmt(e3["tau_max_star"], 0)
macros["sStarFi"] = fmt(e3["fisher"]["s_star"], 1)
a_def = e3["evals"]["0.1"]["default"]["acc"]
a_adj = e3["evals"]["0.1"]["adjoint"]["acc"]
macros["adjAccDefPointOne"] = pct(a_def)
macros["adjAccAdjPointOne"] = pct(a_adj)
macros["adjGainPointOne"] = fmt(100 * (a_adj - a_def), 1)
s_def = e3["evals"]["0.03"]["default"]["acc"]
s_adj = e3["evals"]["0.03"]["adjoint"]["acc"]
macros["adjAccDefSparse"] = pct(s_def)
macros["adjAccAdjSparse"] = pct(s_adj)
macros["adjGainSparse"] = fmt(100 * (s_adj - s_def), 1)

# ------------------------------------------------------------------ exp4
try:
    e4 = load("exp4_gan.json")
    res = e4["results"]
    macros["ganMaeSim"] = fmt(res["sim"]["mae_held_out"], 3)
    macros["ganMaeDr"] = fmt(res["dr"]["mae_held_out"], 3)
    macros["ganMaeGan"] = fmt(res["gan"]["mae_held_out"], 3)
    macros["ganMaeFit"] = fmt(res["fit"]["mae_held_out"], 3)
    macros["ganGapClosed"] = fmt(100 * e4["gap_closed"], 0)
    w0 = e4["wgan_log"][0]
    wN = e4["wgan_log"][-1]
    macros["ganCriticCut"] = fmt(
        100 * (1 - max(wN["w_gan"], 0) / max(w0["w_sim"], 1e-9)), 0)
    refs = [s["g2_ref"] for s in e4["series"]]
    macros["refGtwoMin"] = fmt(min(refs), 2)
    macros["refGtwoMax"] = fmt(max(refs), 2)
    macros["nRealWindows"] = str(int(sum(s["n_windows"]
                                         for s in e4["series"])))
    # information-floor decomposition
    with open(f"{R}/exp4c_floor.json") as f:
        fl = json.load(f)
    macros["floorMae"] = fmt(fl["floor_mae"], 3)
    removable = res["fit"]["mae_held_out"] - fl["floor_mae"]
    removed = res["fit"]["mae_held_out"] - res["sim"]["mae_held_out"]
    macros["pctRemovable"] = fmt(100 * removed / removable, 0)
    macros["pctAboveFloor"] = fmt(
        100 * (res["sim"]["mae_held_out"] - fl["floor_mae"])
        / fl["floor_mae"], 0)
    macros["xFitOverSim"] = fmt(
        res["fit"]["mae_held_out"] / res["sim"]["mae_held_out"], 1)
except FileNotFoundError:
    print("exp4 missing")

# ------------------------------------------------------------------ exp5
try:
    e5 = load("exp5_rl.json")
    macros["pGoodFieldPct"] = fmt(100 * e5["p_good_field"], 0)
    # matched-quality comparison: policies meeting the quality gate
    def ok(s):
        return s["precision"][0] >= 0.88 and s["recall"][0] >= 0.85
    rasters = {k: v for k, v in e5["baselines"].items()
               if k.startswith("raster") and ok(v)}
    heur = {k: v for k, v in e5["baselines"].items()
            if k.startswith("heuristic") and ok(v)}
    best_raster = min(rasters.values(), key=lambda s: s["time_s"][0]) \
        if rasters else None
    best_heur = min(heur.values(), key=lambda s: s["time_s"][0]) \
        if heur else None
    sac = e5["best_per"]
    macros["rlTimeSac"] = fmt(sac["time_s"][0], 0)
    macros["rlPrecSac"] = fmt(sac["precision"][0], 3)
    macros["rlRecSac"] = fmt(sac["recall"][0], 3)
    macros["rlGoodPerMinSac"] = fmt(sac["good_per_min"][0], 2)
    if best_raster:
        macros["rlTimeRaster"] = fmt(best_raster["time_s"][0], 0)
        macros["rlGoodPerMinRaster"] = fmt(best_raster["good_per_min"][0], 2)
        macros["rlSpeedup"] = fmt(best_raster["time_s"][0] /
                                  sac["time_s"][0], 1)
    if best_heur:
        macros["rlTimeHeur"] = fmt(best_heur["time_s"][0], 0)
        macros["rlGoodPerMinHeur"] = fmt(best_heur["good_per_min"][0], 2)
        if best_raster:
            macros["xHeurRaster"] = fmt(best_raster["time_s"][0] /
                                        best_heur["time_s"][0], 1)
    # clairvoyant oracle-stopping bound
    try:
        with open(f"{R}/exp5b_oracle.json") as f:
            orc = json.load(f)
        macros["rlTimeOracle"] = fmt(orc["time_s"][0], 0)
        if best_raster:
            macros["xOracleRaster"] = fmt(best_raster["time_s"][0] /
                                          orc["time_s"][0], 1)
        macros["pctHeurOfOracle"] = fmt(
            100 * (best_raster["time_s"][0] - best_heur["time_s"][0]) /
            (best_raster["time_s"][0] - orc["time_s"][0]), 0) \
            if (best_raster and best_heur) else "--"
    except FileNotFoundError:
        pass
    # throughput growth of the agent during training (first -> final eval)
    g0 = np.mean([c[0]["good_per_min"] for c in e5["curves"]["per"]])
    gN = np.mean([c[-1]["good_per_min"] for c in e5["curves"]["per"]])
    macros["sacTrainGain"] = fmt(gN / max(g0, 1e-9), 1)
    # replay ablation: prioritization preserves the recall tail
    macros["recPer"] = fmt(np.mean(
        [f["recall"][0] for f in e5["final"]["per"]]), 3)
    macros["recUni"] = fmt(np.mean(
        [f["recall"][0] for f in e5["final"]["uniform"]]), 3)
    macros["timeUni"] = fmt(np.mean(
        [f["time_s"][0] for f in e5["final"]["uniform"]]), 0)
    macros["dwellGoodOverBad"] = macros.get("dwellGoodOverBad", "TBD")
except FileNotFoundError:
    print("exp5 missing")

# ------------------------------------------------------------------ exp6
try:
    e6 = load("exp6b_graph.json")
    u = e6["unseen"]
    macros["gAccUncond"] = pct(u["uncond"])
    macros["gAccGraph"] = pct(u["graph"])
    macros["gAccOracle"] = pct(u["oracle"])
    macros["gGapRecovery"] = fmt(100 * u["gap_recovery"], 0)
    # the naive two-platform ablation (exp6): conditioning HURTS
    e6a = load("exp6_graph.json")
    macros["gNaiveUncond"] = pct(e6a["unseen"]["uncond"])
    macros["gNaiveGraph"] = pct(e6a["unseen"]["graph"])
except FileNotFoundError:
    print("exp6b missing")

# extra: dwell allocation ratio from exp5 detail file if present
try:
    with open(f"{R}/exp5_dwell.json") as f:
        dw = json.load(f)
    macros["dwellGoodOverBad"] = fmt(dw["ratio"], 1)
except FileNotFoundError:
    pass

with open(f"{P}/numbers.tex", "w") as f:
    f.write("% AUTO-GENERATED by experiments/make_numbers.py — do not edit\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros to paper/numbers.tex")
for k, v in sorted(macros.items()):
    print(f"  \\{k} = {v}")
