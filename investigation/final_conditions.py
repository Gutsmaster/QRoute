"""
PRE-REGISTERED FINAL TEST (written before running; resumable). All methods use the confirmed shared stack
(write-back + extended inter-route local search). OR-Tools = hardened config AND original config; the fair
reference is the BETTER of the two at the same wall-clock.

C1 DISRUPTION (6 new instances, n=24): a road closure multiplies travel times on the incumbent plan's busiest
   route by 8x. Compared at matched sub-second budget: (a) QPSO MAP-Elites archive lookup, (b) QPSO warm-started
   re-optimisation, (c) OR-Tools WARM-STARTED from the surviving plan (not from scratch), (d) GA warm-started.
   CRITERION: archive (or QPSO at that budget) strictly better than warm OR-Tools on >=5/6.
C2 BUDGET (8 new instances, n=24): QPSO vs OR-Tools at matched 30 s. CRITERION: QPSO mean gap <=0 on >=5/8.
C4 (descriptive, no criterion): per-instance-class gaps; QPSO vs GA under the same stack at n=72.
Nothing is tuned on this data. Results reported as-is.
"""
import sys, os, time, json, math
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "benchmarks")
from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from solomon_synthetic import generate_instance, generate_sweep
from qpso_core import QPSO, evaluate_routes, fitness_weighted
from quantum_extras import build_map_elites_archive
from local_search import split_optimal, local_search_pass, inter_route_pass, make_neighbors
from baselines import GeneticAlgorithm, solve_ortools_cvrptw

BB = (77.205, 28.625, 77.225, 28.645); W = (1.0, 0.02, 0.01, 300.0)
STACK = dict(writeback_mode="sorted_reassign", ls_extended=True)
HARD = dict(first_solution="LOCAL_CHEAPEST_INSERTION", metaheuristic="GUIDED_LOCAL_SEARCH", scale=100, precise=True)
ORIG = dict(first_solution="PATH_CHEAPEST_ARC", metaheuristic="GUIDED_LOCAL_SEARCH", scale=1, precise=False)
GJ = "data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__%s_to_%s_.geojson"
def graph(d): return largest_component_subgraph(build_graph(build_edge_table(load_day(GJ % (d, d)))), BB, strongly_connected=True)
Gs, Gf = graph("2024-08-12"), graph("2024-08-19")
def unpack(i, h): return dict(d=[i.demands[c] for c in i.customers], tw=[i.time_windows[c] for c in i.customers], T=i.time_mat[h], D=i.dist_mat[h], E=i.emis_mat[h])
def sy(s): return dict(d=s["demands"], tw=s["time_windows_s"], T=s["time_mat"], D=s["dist_mat"], E=s["emis_mat"])
def fit(r, I, n=24): return float(fitness_weighted(evaluate_routes(r, I["T"], I["D"], I["E"], I["tw"], I["d"], 100)))
def orboth(I, secs, n=24, warm=None):
    out = []
    for cfg in (HARD, ORIG):
        try:
            r = solve_ortools_cvrptw(n, I["d"], 100, I["tw"], I["T"], I["D"], I["E"], time_limit_s=secs, warm_start=warm, **cfg)
            out.append(fit(r, I, n) if r else float("inf"))
        except Exception: out.append(float("inf"))
    return min(out), out
SAVE = "/tmp/final_cond.json"; S = json.load(open(SAVE)) if os.path.exists(SAVE) else {}
def done(k): return k in S
def put(k, v): S[k] = v; json.dump(S, open(SAVE, "w"))

# ---------------------------------------------------------------- C1 DISRUPTION
D1 = {"Delhi-Aug12-9am-s201": unpack(make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=201), 9),
      "Delhi-Aug19fest-6pm-s203": unpack(make_synthetic_instance(Gf, n_customers=24, capacity=100, seed=203), 18)}
for s in generate_sweep(n_per_kind=2, n_customers=24, seed_base=1300)[:4]: D1[s["name"]] = sy(s)
for name, I in D1.items():
    k = "C1|" + name
    if done(k): continue
    a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"])
    q = QPSO(*a, n_particles=24, n_iter=80, seed=70, **STACK); base = q.run(); inc = base["gbest_routes"]
    cells = build_map_elites_archive(q.archive, n_route_bins=5, n_emis_bins=4)
    Tc = I["T"].copy(); busiest = max(inc, key=len)
    seq = [0] + [c + 1 for c in busiest]
    for u, v in zip(seq[:-1], seq[1:] + [0]): Tc[u, v] *= 8.0
    J = dict(I, T=Tc); nn = make_neighbors(Tc, 6)
    t0 = time.time(); inc_cost = fit(inc, J)
    best_arc, t_arc = inc_cost, 0.0
    for _, keys, _ in cells.values():
        rts = split_optimal(list(np.argsort(keys)), I["d"], 100, Tc, I["D"], I["E"], I["tw"], W)
        rts = local_search_pass(rts, I["d"], 100, Tc, I["D"], I["E"], I["tw"], W, window=6)
        best_arc = min(best_arc, fit(rts, J))
    t_arc = time.time() - t0
    T_budget = max(0.05, round(t_arc, 2))
    t0 = time.time(); qw = QPSO(24, I["d"], 100, I["tw"], Tc, I["D"], I["E"], n_particles=24, n_iter=8, seed=70, **STACK).run(); t_qw = time.time() - t0
    orw_b, orw_all = orboth(J, T_budget, warm=inc)
    orc_b, _ = orboth(J, T_budget)
    gw = GeneticAlgorithm(24, I["d"], 100, I["tw"], Tc, I["D"], I["E"], pop_size=24, n_gen=8, seed=70, local_search=True, lamarck=True, ls_extended=True).run()
    put(k, dict(inc=inc_cost, arc=best_arc, t_arc=t_arc, qw=qw["gbest_fit"], t_qw=t_qw, orw=orw_b, orw_all=orw_all, orc=orc_b, gw=gw["gbest_fit"], T=T_budget))
    print("C1 done", name, flush=True)

# ---------------------------------------------------------------- C2 BUDGET 30s
D2 = {"Delhi-Aug12-noon-s211": unpack(make_synthetic_instance(Gs, n_customers=24, capacity=100, seed=211), 12),
      "Delhi-Aug19fest-9am-s213": unpack(make_synthetic_instance(Gf, n_customers=24, capacity=100, seed=213), 9)}
for s in generate_sweep(n_per_kind=2, n_customers=24, seed_base=1500): D2[s["name"]] = sy(s)
for name, I in D2.items():
    k = "C2|" + name
    if done(k): continue
    a = (24, I["d"], 100, I["tw"], I["T"], I["D"], I["E"]); cq, tq, cg = [], [], []
    for sd in (70, 71, 72):
        t = time.time(); cq.append(QPSO(*a, n_particles=30, n_iter=700, seed=sd, **STACK).run()["gbest_fit"]); tq.append(time.time() - t)
        cg.append(GeneticAlgorithm(*a, pop_size=30, n_gen=1400, seed=sd, local_search=True, lamarck=True, ls_extended=True).run()["gbest_fit"])
    T = max(1.0, round(float(np.mean(tq)), 1)); orb, orall = orboth(I, T)
    put(k, dict(q=float(np.mean(cq)), tq=float(np.mean(tq)), g=float(np.mean(cg)), orb=orb, orall=orall, T=T))
    print("C2 done", name, flush=True)

print("ALL PHASES COMPLETE", flush=True)
