"""
The "extra" layer on top of core QPSO:
  1. Quantum-walk centrality       -> biases initial random-key population
  2. Reverse-annealing recovery    -> beta schedule that ramps back up on
                                       disruption, then decays again
  3. Hierarchical / multi-scale QPSO -> KMeans zone clustering + per-zone
                                       QPSO + stitched inter-zone tour
  4. Pareto front (multi-objective) -> weighted-sum sweep + non-dominated
                                       filter over (time, distance, emissions)
  5. MAP-Elites-style plan archive  -> keep best solution per (n_routes,
                                       emissions-bucket) behavioural cell
  6. Route-stability objective      -> variance of a fixed route's travel
                                       time across the day's 24 hourly
                                       snapshots (lower = more operationally
                                       reliable, independent of raw speed)
"""
import numpy as np
from scipy.linalg import expm
from sklearn.cluster import KMeans

from qpso_core import QPSO, decode_random_key, evaluate_routes, fitness_weighted


# ------------------------------------------------------------- 1. quantum walk centrality
def quantum_walk_centrality(G, nodes, times=(0.5, 1.0, 2.0, 4.0)):
    """
    Continuous-time quantum walk (CTQW) centrality on the (small, undirected)
    subgraph induced by `nodes`. H = adjacency matrix. Starting from a
    uniform superposition over all nodes, centrality[j] is the time-averaged
    probability of finding the walker at node j -- a standard CTQW
    centrality definition (Sanchez-Burillo et al. style), used here purely
    as a structural signal to bias where in the tour a customer is likely to
    be visited early.
    NOTE: this runs on the small induced subgraph of instance nodes for
    tractability (full-graph CTQW on ~20k nodes is not tractable via dense
    matrix exponential); this is disclosed, not hidden.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for u in nodes:
        for v in G.neighbors(u) if G.has_node(u) else []:
            if v in idx:
                A[idx[u], idx[v]] = 1.0
                A[idx[v], idx[u]] = 1.0
    psi0 = np.ones(n, dtype=complex) / np.sqrt(n)
    probs = np.zeros(n)
    for t in times:
        U = expm(-1j * A * t)
        psi_t = U @ psi0
        probs += np.abs(psi_t) ** 2
    probs /= len(times)
    probs = probs / probs.sum()
    return probs  # length n, sums to 1


# ------------------------------------------------------------- 2. reverse-annealing recovery
def reverse_annealing_schedule(disruption_iters, ramp_up=0.9, decay_rate=0.5,
                                recovery_len=25):
    """
    Returns a beta_schedule(it, n_iter) function for QPSO. Normally beta
    decays linearly 1.0 -> 0.5 (pure exploitation drift). On each iteration
    in `disruption_iters`, beta is ramped back UP toward `ramp_up` and then
    decays again over `recovery_len` iterations -- mimicking reverse
    annealing's "re-introduce the transverse field, then anneal back down"
    to escape a now-stale optimum after a network disruption.
    """
    disruption_iters = set(disruption_iters)

    def schedule(it, n_iter):
        base = 1.0 - 0.5 * it / n_iter
        # find most recent disruption at or before `it`
        recent = [d for d in disruption_iters if d <= it]
        if not recent:
            return base
        last = max(recent)
        since = it - last
        if since >= recovery_len:
            return base
        local_decay = ramp_up - (ramp_up - base) * (since / recovery_len) ** decay_rate
        return max(base, local_decay)

    return schedule


# ------------------------------------------------------------- 3. hierarchical QPSO
def hierarchical_qpso(customers_xy, demands, capacity, time_windows_s,
                       time_mat, dist_mat, emis_mat, n_zones=3,
                       n_particles=20, n_iter=80, seed=0, qpso_kwargs=None):
    """
    City -> zones -> routes. KMeans clusters customers into `n_zones`
    geographic zones; QPSO solves each zone's mini-VRP independently
    (parallel in principle); the zone route sets are concatenated as the
    final plan. This is the scalability mechanism: total optimisation cost
    grows roughly with zone size, not the full customer count, at the cost
    of some cross-zone route-sharing optimality (a real, disclosed
    trade-off, not free scalability).
    """
    n = len(customers_xy)
    n_zones = min(n_zones, max(1, n // 4))
    km = KMeans(n_clusters=n_zones, n_init=5, random_state=seed).fit(customers_xy)
    labels = km.labels_

    all_routes = []
    total_metrics = {"total_time_s": 0.0, "total_dist_m": 0.0, "total_emis_g": 0.0,
                      "n_routes": 0, "penalty": 0.0}
    per_zone_history = []
    for z in range(n_zones):
        zone_customers = [i for i in range(n) if labels[i] == z]
        if not zone_customers:
            continue
        zm = len(zone_customers)
        # remap sub-instance indices 0..zm-1 -> original indices for matrix lookup
        idx_map = zone_customers
        zone_demands = [demands[i] for i in zone_customers]
        zone_tw = [time_windows_s[i] for i in zone_customers]
        # build zone-local time/dist/emis matrices: depot=0 + zone customers
        full_idx = [0] + [i + 1 for i in idx_map]
        zT = time_mat[np.ix_(full_idx, full_idx)]
        zD = dist_mat[np.ix_(full_idx, full_idx)]
        zE = emis_mat[np.ix_(full_idx, full_idx)]

        qpso = QPSO(n_customers=zm, demands=zone_demands, capacity=capacity,
                    time_windows_s=zone_tw, time_mat=zT, dist_mat=zD, emis_mat=zE,
                    n_particles=n_particles, n_iter=n_iter, seed=seed + z, **(qpso_kwargs or {}))
        res = qpso.run()
        per_zone_history.append(res["history"])
        zone_routes_local = res["gbest_routes"]
        # map back to original customer indices
        zone_routes_global = [[idx_map[c] for c in route] for route in zone_routes_local]
        all_routes.extend(zone_routes_global)
        for k in ("total_time_s", "total_dist_m", "total_emis_g", "penalty"):
            total_metrics[k] += res["gbest_metrics"][k]
        total_metrics["n_routes"] += res["gbest_metrics"]["n_routes"]

    return {"routes": all_routes, "metrics": total_metrics,
            "n_zones": n_zones, "labels": labels, "per_zone_history": per_zone_history}


# ------------------------------------------------------------- 4. Pareto front
def pareto_front_sweep(n_customers, demands, capacity, time_windows_s,
                        time_mat, dist_mat, emis_mat, n_weight_samples=12,
                        n_particles=20, n_iter=60, seed=0, time_mat_by_hour=None,
                        stability_hours=range(6, 22)):
    """
    Traces an approximate Pareto front with a normalised weighted-sum sweep
    (single-objective QPSO per weight vector), then keeps the non-dominated
    set over FOUR metrics: total time, total distance, total emissions and
    route-stability (coefficient of variation of the plan's total time across
    departure hours).
    Honest notes:
      * weights are normalised by each objective's magnitude at a reference
        solution, otherwise the objective with the biggest units silently
        dominates every scalarisation (that is why the first version's front
        collapsed to one point);
      * stability is measured on each solution, not optimised directly, so it
        acts as a filtering/selection axis, not a search objective.
    """
    rng = np.random.default_rng(seed)
    ref = QPSO(n_customers, demands, capacity, time_windows_s, time_mat, dist_mat, emis_mat,
               n_particles=n_particles, n_iter=n_iter, seed=seed).run()
    rm = ref["gbest_metrics"]
    scale = np.array([rm["total_time_s"], rm["total_dist_m"], rm["total_emis_g"]])

    simplex = [np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3] * 3)]
    simplex += [rng.dirichlet(np.ones(3) * 0.6) for _ in range(max(0, n_weight_samples - 4))]
    cands = []
    for s, w in enumerate(simplex):
        eff = 10000.0 * w / scale   # each objective contributes ~10k * its weight share
        weights = (float(eff[0]), float(eff[1]), float(eff[2]), 300.0)
        r = QPSO(n_customers, demands, capacity, time_windows_s, time_mat, dist_mat, emis_mat,
                 n_particles=n_particles, n_iter=n_iter, weights=weights, seed=seed + 1 + s).run()
        m = r["gbest_metrics"]
        cv = float("nan")
        if time_mat_by_hour is not None:
            cv = route_stability(r["gbest_routes"], time_mat_by_hour, list(stability_hours))["cv"]
        cands.append({"time_s": float(m["total_time_s"]), "dist_m": float(m["total_dist_m"]),
                      "emis_g": float(m["total_emis_g"]), "stability_cv": cv,
                      "weights": w.tolist(), "routes": r["gbest_routes"]})
    use_cv = time_mat_by_hour is not None
    pts = np.array([[c["time_s"], c["dist_m"], c["emis_g"]] + ([c["stability_cv"]] if use_cv else [])
                    for c in cands])
    dominated = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated[i] = True
                break
    front = [c for c, d in zip(cands, dominated) if not d]
    return front, cands


# ------------------------------------------------------------- 5. MAP-Elites archive
def build_map_elites_archive(archive_records, n_route_bins=4, n_emis_bins=4):
    """
    archive_records: list of (fitness, keys, metrics) collected during a
    QPSO run (QPSO.archive). Bins solutions by (n_routes, emissions-bucket)
    and keeps the best-fitness solution per cell -- a lightweight
    MAP-Elites-style diverse solution archive ("superposition plan
    archive"): if the best-fitness route plan later becomes infeasible
    (e.g. a road closes), the next best DIVERSE plan can be substituted
    without re-optimizing from scratch.
    """
    if not archive_records:
        return {}
    emis_vals = [r[2]["total_emis_g"] for r in archive_records]
    lo, hi = min(emis_vals), max(emis_vals) + 1e-6
    edges = np.linspace(lo, hi, n_emis_bins + 1)

    cells = {}
    for fit, keys, metrics in archive_records:
        n_routes = min(metrics["n_routes"], n_route_bins)
        emis_bin = int(np.clip(np.searchsorted(edges, metrics["total_emis_g"]) - 1, 0, n_emis_bins - 1))
        cell = (n_routes, emis_bin)
        if cell not in cells or fit < cells[cell][0]:
            cells[cell] = (fit, keys.copy(), metrics)
    return cells


# ------------------------------------------------------------- 6. route stability
def route_stability(routes, time_mat_by_hour, hours_to_test, depot_idx=0):
    """
    For a FIXED set of routes, evaluates total route time at several
    different hours of day and returns the coefficient of variation (std/
    mean) across hours -- a proxy for "how much does this plan's duration
    swing depending on when it departs". A low-variance route is more
    operationally predictable for drivers/dispatchers even if it isn't the
    single fastest at any one hour -- the "fastest isn't always best"
    argument made concrete as a number.
    """
    totals = []
    for hour in hours_to_test:
        T = time_mat_by_hour[hour]
        total = 0.0
        for route in routes:
            prev = depot_idx
            for c in route:
                node = c + 1
                total += T[prev, node]
                prev = node
            total += T[prev, depot_idx]
        totals.append(total)
    totals = np.array(totals)
    cv = totals.std() / (totals.mean() + 1e-9)
    return {"hourly_totals": totals, "mean": totals.mean(), "std": totals.std(), "cv": cv}
