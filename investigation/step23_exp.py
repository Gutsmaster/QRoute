import sys, time, json
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_instance
from qpso_core import QPSO
from baselines import GeneticAlgorithm

df = build_edge_table(load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"))
G = build_graph(df); Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645), strongly_connected=True)
inst = make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=0)          # TUNING
INST = {"Delhi": dict(d=[inst.demands[c] for c in inst.customers], tw=[inst.time_windows[c] for c in inst.customers],
                      T=inst.time_mat[9], D=inst.dist_mat[9], E=inst.emis_mat[9])}
for kind in ("C", "R", "RC"):
    s = generate_instance(kind, n_customers=24, seed=101)                          # TUNING
    INST[kind + "1"] = dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])
SEEDS = tuple(range(6)); NP, NI = 24, 80

V = {
 "Q0  baseline":            dict(update_rule="quantum"),
 "Q+WB  writeback":         dict(update_rule="quantum", writeback_mode="sorted_reassign"),
 "Q+Bs  borda-sorted":      dict(update_rule="quantum", mbest_mode="borda_sorted"),
 "Q+Bm  borda-mean":        dict(update_rule="quantum", mbest_mode="borda_mean"),
 "Q+RN  renorm":            dict(update_rule="quantum", renormalize=True),
 "Q+RN+Bs":                 dict(update_rule="quantum", renormalize=True, mbest_mode="borda_sorted"),
 "Q+WB+Bs":                 dict(update_rule="quantum", writeback_mode="sorted_reassign", mbest_mode="borda_sorted"),
 "P0  baseline":            dict(update_rule="classical"),
 "P+WB  writeback":         dict(update_rule="classical", writeback_mode="sorted_reassign"),
 "P+RN  renorm":            dict(update_rule="classical", renormalize=True),
}
res = {v: {k: [] for k in INST} for v in V}; ga = {k: [] for k in INST}; tm = {v: [] for v in V}
for name, I in INST.items():
    for seed in SEEDS:
        for v, kw in V.items():
            t0 = time.time()
            r = QPSO(24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], n_particles=NP, n_iter=NI, seed=seed, **kw).run()
            res[v][name].append(r["gbest_fit"]); tm[v].append(time.time() - t0)
        ga[name].append(GeneticAlgorithm(24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], pop_size=NP, n_gen=NI, seed=seed,
                                         local_search=True, lamarck=True).run()["gbest_fit"])
    print("done", name, flush=True)
    json.dump({"res": res, "ga": ga, "tm": tm}, open("/tmp/step23.json", "w"))

names = list(INST)
gam = {n: np.mean(ga[n]) for n in names}
print("\nMEAN FITNESS per instance (lower=better); gap vs memetic GA (same seeds)")
print("%-24s" % "variant" + "".join("%9s" % n for n in names) + "   gapGA%   sec")
for v in V:
    row = [np.mean(res[v][n]) for n in names]
    gap = np.mean([100 * (row[i] - gam[n]) / gam[n] for i, n in enumerate(names)])
    print("%-24s" % v + "".join("%9.0f" % x for x in row) + "  %+6.1f  %5.1f" % (gap, np.mean(tm[v])))
print("%-24s" % "GA+LS memetic (ref)" + "".join("%9.0f" % gam[n] for n in names))

def flat(v): return np.array([res[v][n][i] for n in names for i in range(len(SEEDS))])
def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * ps[idx]); adj[idx] = min(1.0, run)
    return adj
def cmp(a, b):
    x, y = flat(a), flat(b); d = y - x
    rel = 100 * np.mean(d / y)
    try: p = wilcoxon(x, y)[1]
    except ValueError: p = 1.0
    return (x < y).sum(), len(x), rel, p

print("\nFAMILY A: variant vs its own baseline (paired, %d pairs; positive = variant better)" % (len(names) * len(SEEDS)))
pairsA = [(v, "Q0  baseline") for v in V if v.startswith("Q") and v != "Q0  baseline"] + \
         [("P+WB  writeback", "P0  baseline"), ("P+RN  renorm", "P0  baseline")]
outA = [cmp(a, b) for a, b in pairsA]; adjA = holm(np.array([o[3] for o in outA]))
for (a, b), (w, n, rel, p), ap in zip(pairsA, outA, adjA):
    print("  %-22s vs %-14s wins %2d/%d  mean %+6.2f%%  p=%.3f  Holm-p=%.3f" % (a, b.split()[0], w, n, rel, p, ap))
print("\nFAMILY B: quantum vs classical under IDENTICAL infrastructure (positive = quantum better)")
pairsB = [("Q0  baseline", "P0  baseline"), ("Q+WB  writeback", "P+WB  writeback"), ("Q+RN  renorm", "P+RN  renorm")]
outB = [cmp(a, b) for a, b in pairsB]; adjB = holm(np.array([o[3] for o in outB]))
for (a, b), (w, n, rel, p), ap in zip(pairsB, outB, adjB):
    print("  %-22s vs %-16s wins %2d/%d  mean %+6.2f%%  p=%.3f  Holm-p=%.3f" % (a, b, w, n, rel, p, ap))
