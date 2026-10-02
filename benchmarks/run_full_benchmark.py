"""
Honest benchmarking suite. PROTOCOL (stated so nobody has to trust us):
  * QPSO's settings and the opponents' settings (classical-PSO inertia, GA
    local-search mode) were tuned on the TUNING set (Delhi seed-0 instance +
    Solomon-style C/R/RC seed-101 instances, algorithm seeds 0-3).
  * The numbers reported here come from a HELD-OUT set of different
    instances and different algorithm seeds -- nothing here was tuned on.
  * Every swarm/evolutionary method shares the SAME decoder (DP split), the
    same objective, and (where marked) the same local search + cache.
  * Wilcoxon signed-rank tests are paired by (instance, seed).
"""
import sys, os, time
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qpso_core import QPSO, evaluate_routes, fitness_weighted
from baselines import GeneticAlgorithm, AntColony, TunedLocalSearch, solve_ortools_cvrptw

# chosen from the GA tuning round (see outputs/tuning_logs/)
GA_MEMETIC_KW = dict(local_search=True, lamarck=True, ls_extended=True)   # same extended inter-route search as QPSO/PSO

ALGO_ORDER = ["QPSO", "PSO(same infra, tuned)", "QPSO-Basic(first pass)",
              "GA+LS(memetic)", "GA(plain)", "ACO", "TunedLS(HGS-inspired)"]


def run_all_algorithms(n_customers, demands, capacity, tw, T, D, E, seed,
                        n_particles=24, n_iter=80):
    args = (n_customers, demands, capacity, tw, T, D, E)
    algos = {
        "QPSO": lambda: QPSO(*args, n_particles=n_particles, n_iter=n_iter, seed=seed),
        "PSO(same infra, tuned)": lambda: QPSO(*args, n_particles=n_particles, n_iter=n_iter,
                                                seed=seed, update_rule="classical"),
        "QPSO-Basic(first pass)": lambda: QPSO(*args, n_particles=n_particles, n_iter=n_iter, seed=seed,
                                                use_optimal_split=False, local_search=False,
                                                savings_seed_frac=0.0, tunnel_prob=0.0,
                                                weighted_mbest=False, beta_start=1.0, beta_end=0.5),
        "GA+LS(memetic)": lambda: GeneticAlgorithm(*args, pop_size=n_particles, n_gen=n_iter,
                                                    seed=seed, **GA_MEMETIC_KW),
        "GA(plain)": lambda: GeneticAlgorithm(*args, pop_size=n_particles + 10, n_gen=n_iter, seed=seed),
        "ACO": lambda: AntColony(*args, n_ants=n_particles, n_iter=int(n_iter * 0.8), seed=seed),
        "TunedLS(HGS-inspired)": lambda: TunedLocalSearch(*args, n_restarts=15, seed=seed),
    }
    out = {}
    for name in ALGO_ORDER:
        t0 = time.time()
        res = algos[name]().run()
        out[name] = {"fit": float(res["gbest_fit"]), "time_s": time.time() - t0,
                     "metrics": {k: (float(v) if not isinstance(v, list) else [float(x) for x in v])
                                 for k, v in res["gbest_metrics"].items()}}
    return out


def benchmark_instance(name, n_customers, demands, capacity, tw, T, D, E,
                        seeds=(10, 11, 12, 13), n_particles=24, n_iter=80,
                        ortools_time_limit=5):
    per_seed = [run_all_algorithms(n_customers, demands, capacity, tw, T, D, E, s, n_particles, n_iter)
                for s in seeds]
    summary = {}
    for algo in ALGO_ORDER:
        fits = [ps[algo]["fit"] for ps in per_seed]
        summary[algo] = {"mean_fit": float(np.mean(fits)), "std_fit": float(np.std(fits)), "fits": fits,
                         "mean_time_s": float(np.mean([ps[algo]["time_s"] for ps in per_seed]))}
    # OR-Tools CVRPTW reference: BETTER of a hardened and the original configuration (same objective + soft time windows), single run each
    ort_name = "OR-Tools(GLS,%ds)" % ortools_time_limit
    HARD_OR = dict(first_solution="LOCAL_CHEAPEST_INSERTION", metaheuristic="GUIDED_LOCAL_SEARCH", scale=100, precise=True)
    ORIG_OR = dict(first_solution="PATH_CHEAPEST_ARC", metaheuristic="GUIDED_LOCAL_SEARCH", scale=1, precise=False)
    try:
        t0 = time.time(); best_or = None
        for cfg in (HARD_OR, ORIG_OR):
            routes = solve_ortools_cvrptw(n_customers, demands, capacity, tw, T, D, E, time_limit_s=ortools_time_limit, **cfg)
            if routes:
                f = float(fitness_weighted(evaluate_routes(routes, T, D, E, tw, demands, capacity)))
                best_or = f if best_or is None else min(best_or, f)
        if best_or is not None:
            summary[ort_name] = {"mean_fit": best_or, "std_fit": 0.0, "fits": [best_or], "mean_time_s": time.time() - t0}
    except Exception as e:
        print("OR-Tools failed on", name, e)
    best = min(s["mean_fit"] for s in summary.values())
    for a in summary:
        summary[a]["gap_to_best_pct"] = 100.0 * (summary[a]["mean_fit"] - best) / best
    return {"instance": name, "per_seed": per_seed, "summary": summary, "seeds": list(seeds)}


def wilcoxon_qpso_vs(results, baseline):
    q, b = [], []
    for r in results:
        for ps in r["per_seed"]:
            q.append(ps["QPSO"]["fit"]); b.append(ps[baseline]["fit"])
    try:
        stat, p = wilcoxon(q, b)
    except ValueError:
        return None
    wins = sum(1 for x, y in zip(q, b) if x < y)
    mean_gap = 100.0 * (np.mean(b) - np.mean(q)) / np.mean(b)
    return {"baseline": baseline, "p_value": float(p), "n_pairs": len(q),
            "qpso_lower_count": wins, "qpso_lower_pct": 100.0 * wins / len(q),
            "qpso_mean_improvement_pct": float(mean_gap)}


def dolan_more_profile(results, algo_names, taus=None):
    taus = np.linspace(1.0, 1.5, 51) if taus is None else taus
    ratios = {a: [] for a in algo_names}
    for r in results:
        best = min(r["summary"][a]["mean_fit"] for a in algo_names if a in r["summary"])
        for a in algo_names:
            ratios[a].append(r["summary"][a]["mean_fit"] / best)
    return {"taus": taus.tolist(),
            "profile": {a: [float(np.mean([rv <= t for rv in ratios[a]])) for t in taus] for a in algo_names}}


def ablation_quantum_term(results):
    """QPSO vs classical PSO running through the IDENTICAL class/infrastructure:
    only the position-update rule differs."""
    d = np.array([ps["PSO(same infra, tuned)"]["fit"] - ps["QPSO"]["fit"]
                  for r in results for ps in r["per_seed"]])
    base = np.mean([ps["PSO(same infra, tuned)"]["fit"] for r in results for ps in r["per_seed"]])
    try:
        p = float(wilcoxon(d)[1])
    except ValueError:
        p = float("nan")
    return {"mean_improvement": float(d.mean()), "pct_improvement_over_classical_pso": float(100 * d.mean() / base),
            "n_pairs": len(d), "n_qpso_better": int((d > 0).sum()), "n_qpso_worse": int((d < 0).sum()),
            "p_value": p}
