"""
PRE-REGISTERED CONFIRMATORY TEST (written before running; no tuning after seeing results)
Change under test: minimal-disturbance write-back (writeback_mode="sorted_reassign"), a SHARED-infrastructure
change applied identically to QPSO and to classical PSO (the memetic GA already uses Lamarckian write-back).
Data: the 8 held-out instances (2 Delhi + 6 Solomon-style, exactly as in the frozen benchmark) with FRESH
algorithm seeds 14-19 (frozen benchmark used 10-13) -> 48 paired runs per comparison.
Hypotheses (two-sided paired Wilcoxon, Holm-corrected over these 4):
  H1  QPSO+WB   vs QPSO (frozen)          H2  PSO+WB vs PSO (frozen)
  H3  QPSO+WB   vs PSO+WB (same infra)    H4  QPSO+WB vs memetic GA
No further variants will be tried on this data.
"""
import sys, time, json
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_sweep
from qpso_core import QPSO, evaluate_routes, fitness_weighted
from baselines import GeneticAlgorithm, AntColony, TunedLocalSearch, solve_ortools_cvrptw

df = build_edge_table(load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"))
G = build_graph(df); Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645), strongly_connected=True)
def unpack(i, h): return dict(d=[i.demands[c] for c in i.customers], tw=[i.time_windows[c] for c in i.customers],
                              T=i.time_mat[h], D=i.dist_mat[h], E=i.emis_mat[h])
INST = {}
for name, seed, hour in (("Delhi-9am-s7", 7, 9), ("Delhi-6pm-s11", 11, 18)):
    INST[name] = unpack(make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=seed), hour)
for s in generate_sweep(n_per_kind=2, n_customers=24, seed_base=300):
    INST[s["name"]] = dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])
SEEDS = tuple(range(14, 20)); NP, NI = 24, 80
WB = dict(writeback_mode="sorted_reassign")
def algos(I, seed):
    a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"])
    return {
     "QPSO (frozen)":  lambda: QPSO(*a, n_particles=NP, n_iter=NI, seed=seed),
     "QPSO+WB":        lambda: QPSO(*a, n_particles=NP, n_iter=NI, seed=seed, **WB),
     "PSO (frozen)":   lambda: QPSO(*a, n_particles=NP, n_iter=NI, seed=seed, update_rule="classical"),
     "PSO+WB":         lambda: QPSO(*a, n_particles=NP, n_iter=NI, seed=seed, update_rule="classical", **WB),
     "GA+LS (memetic)":lambda: GeneticAlgorithm(*a, pop_size=NP, n_gen=NI, seed=seed, local_search=True, lamarck=True),
     "GA (plain)":     lambda: GeneticAlgorithm(*a, pop_size=NP + 10, n_gen=NI, seed=seed),
     "ACO":            lambda: AntColony(*a, n_ants=NP, n_iter=int(NI * 0.8), seed=seed),
     "TunedLS":        lambda: TunedLocalSearch(*a, n_restarts=15, seed=seed),
    }
NAMES = list(algos(next(iter(INST.values())), 0).keys())
R = {n: {k: [] for k in INST} for n in NAMES}; ORT = {}
import os
if os.path.exists("/tmp/confirm.json"):   # RESUME after an interruption: design, seeds and hypotheses unchanged
    _d = json.load(open("/tmp/confirm.json")); ORT = _d["ORT"]
    for n in NAMES:
        for k, v in _d["R"][n].items():
            if k in ORT: R[n][k] = v
    print("resumed; already complete:", list(ORT), flush=True)
for iname, I in INST.items():
    if iname in ORT: continue
    for seed in SEEDS:
        for n, f in algos(I, seed).items():
            R[n][iname].append(f().run()["gbest_fit"])
    rts = solve_ortools_cvrptw(24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], time_limit_s=5)
    ORT[iname] = float(fitness_weighted(evaluate_routes(rts, I["T"], I["D"], I["E"], I["tw"], I["d"], 100)))
    print("done", iname, flush=True)
    json.dump({"R": R, "ORT": ORT}, open("/tmp/confirm.json", "w"))

inames = list(INST)
print("\nMEAN COST per held-out instance, 6 fresh seeds (lower = better)")
print("%-16s" % "algorithm" + "".join("%15s" % n[:14] for n in inames) + "   meanGap%")
allv = {n: np.array([np.mean(R[n][i]) for i in inames]) for n in NAMES}; allv["OR-Tools"] = np.array([ORT[i] for i in inames])
best = np.min(np.vstack(list(allv.values())), axis=0)
for n, v in allv.items():
    print("%-16s" % n + "".join("%15.0f" % x for x in v) + "   %6.1f" % np.mean(100 * (v - best) / best))

def flat(n): return np.array([R[n][i][s] for i in inames for s in range(len(SEEDS))])
def cmp(a, b):
    x, y = flat(a), flat(b)
    try: p = wilcoxon(x, y)[1]
    except ValueError: p = 1.0
    return (x < y).sum(), len(x), 100 * np.mean((y - x) / y), p
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
H = [("H1 QPSO+WB vs QPSO(frozen)", "QPSO+WB", "QPSO (frozen)"), ("H2 PSO+WB vs PSO(frozen)", "PSO+WB", "PSO (frozen)"),
     ("H3 QPSO+WB vs PSO+WB (same infra)", "QPSO+WB", "PSO+WB"), ("H4 QPSO+WB vs memetic GA", "QPSO+WB", "GA+LS (memetic)")]
out = [cmp(a, b) for _, a, b in H]; adj = holm(np.array([o[3] for o in out]))
print("\nPRE-REGISTERED TESTS (48 pairs; mean = % improvement of first over second, positive = first better)")
for (lab, _, _), (w, n, rel, p), ap in zip(H, out, adj):
    print("  %-38s wins %2d/%d  mean %+6.2f%%  p=%.4f  Holm-p=%.4f" % (lab, w, n, rel, p, ap))
print("\nDESCRIPTIVE (not pre-registered): QPSO+WB vs the other baselines")
for b in ("GA (plain)", "ACO", "TunedLS", "QPSO (frozen)"):
    w, n, rel, p = cmp("QPSO+WB", b); print("  vs %-16s wins %2d/%d  mean %+7.2f%%  p=%.5f" % (b, w, n, rel, p))
q = allv["QPSO+WB"]; o = allv["OR-Tools"]
print("\nQPSO+WB gap to OR-Tools per instance: " + ", ".join("%s %+.1f%%" % (i[:11], 100 * (a - b) / b) for i, a, b in zip(inames, q, o)) + "  | mean %+.1f%%" % np.mean(100 * (q - o) / o))
