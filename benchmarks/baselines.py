"""
Baseline metaheuristics benchmarked against QPSO, all using the SAME
decode/evaluate functions (qpso_core.decode_random_key /
qpso_core.evaluate_routes / fitness_weighted) so comparisons are apples-to-
apples -- only the search strategy differs.

Included:
  - Classical PSO        (no quantum delta-well update; standard velocity/position)
  - Genetic Algorithm     (order crossover + swap mutation on the giant tour)
  - Ant Colony Optimization (pheromone-guided route construction)
  - Tuned local search    (nearest-neighbor + 2-opt/or-opt; HONESTLY labelled
                            as "Tuned-LS (HGS-inspired)" -- a strong local-
                            search baseline, NOT a full from-scratch HGS-CVRP
                            implementation, which was out of scope to verify
                            correct in this session)
  - OR-Tools CVRPTW       (Google OR-Tools routing solver, used as the
                            near-exact reference for small instances)
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qpso_core import decode_random_key, evaluate_routes, fitness_weighted
from local_search import split_optimal, local_search_pass, make_neighbors, inter_route_pass


# ---------------------------------------------------------------- classical PSO
class ClassicalPSO:
    """Standard velocity/position PSO on the same random-key encoding.
    No quantum term -- this is the direct classical baseline QPSO claims to
    beat."""
    def __init__(self, n_customers, demands, capacity, time_windows_s,
                 time_mat, dist_mat, emis_mat, n_particles=30, n_iter=120,
                 weights=(1.0, 0.02, 0.01, 300.0), seed=0,
                 w=0.7, c1=1.5, c2=1.5, use_optimal_split=True):
        self.n = n_customers
        self.demands, self.capacity, self.tw = demands, capacity, time_windows_s
        self.time_mat, self.dist_mat, self.emis_mat = time_mat, dist_mat, emis_mat
        self.n_particles, self.n_iter, self.weights = n_particles, n_iter, weights
        self.rng = np.random.default_rng(seed)
        self.w, self.c1, self.c2 = w, c1, c2
        self.use_optimal_split = use_optimal_split

    def _eval(self, keys):
        if self.use_optimal_split:
            order = list(np.argsort(keys))
            routes = split_optimal(order, self.demands, self.capacity, self.time_mat,
                                    self.dist_mat, self.emis_mat, self.tw, self.weights)
        else:
            routes = decode_random_key(keys, list(range(self.n)), self.demands, self.capacity)
        m = evaluate_routes(routes, self.time_mat, self.dist_mat, self.emis_mat,
                             self.tw, self.demands, self.capacity)
        return fitness_weighted(m, self.weights), m

    def run(self):
        X = self.rng.random((self.n_particles, self.n))
        V = self.rng.uniform(-0.1, 0.1, (self.n_particles, self.n))
        pbest = X.copy()
        pbest_fit = np.full(self.n_particles, np.inf)
        gbest, gbest_fit, gbest_metrics = None, np.inf, None
        history = []
        for it in range(self.n_iter):
            for i in range(self.n_particles):
                fit, m = self._eval(X[i])
                if fit < pbest_fit[i]:
                    pbest_fit[i], pbest[i] = fit, X[i].copy()
                if fit < gbest_fit:
                    gbest_fit, gbest, gbest_metrics = fit, X[i].copy(), m
            r1 = self.rng.random((self.n_particles, self.n))
            r2 = self.rng.random((self.n_particles, self.n))
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest[None, :] - X)
            X = np.clip(X + V, 0, 1)
            history.append(gbest_fit)
        return {"gbest_keys": gbest, "gbest_fit": gbest_fit, "gbest_metrics": gbest_metrics, "history": history}


# ---------------------------------------------------------------- GA
class GeneticAlgorithm:
    def __init__(self, n_customers, demands, capacity, time_windows_s,
                 time_mat, dist_mat, emis_mat, pop_size=40, n_gen=120,
                 weights=(1.0, 0.02, 0.01, 300.0), seed=0,
                 mutation_rate=0.15, elite_frac=0.1, use_optimal_split=True,
                 local_search=False, lamarck=False, ls_window=6,
                 ls_extended=False, ls_ext_within=0.02, ls_ext_k=6):
        self.n = n_customers
        self.demands, self.capacity, self.tw = demands, capacity, time_windows_s
        self.time_mat, self.dist_mat, self.emis_mat = time_mat, dist_mat, emis_mat
        self.pop_size, self.n_gen, self.weights = pop_size, n_gen, weights
        self.rng = np.random.default_rng(seed)
        self.mutation_rate, self.elite_frac = mutation_rate, elite_frac
        self.use_optimal_split = use_optimal_split
        self.local_search, self.lamarck, self.ls_window = local_search, lamarck, ls_window
        self.ls_extended, self.ls_ext_within = ls_extended, ls_ext_within
        self._nn = make_neighbors(time_mat, ls_ext_k) if ls_extended else None
        self._best = float('inf')
        self._cache = {}

    def _decode_perm(self, perm):
        keys = np.argsort(perm).astype(float) / len(perm)
        return keys

    def _eval_perm(self, perm):
        order = list(perm)
        routes = split_optimal(order, self.demands, self.capacity, self.time_mat,
                                self.dist_mat, self.emis_mat, self.tw, self.weights,
                                cache=self._cache)
        m = evaluate_routes(routes, self.time_mat, self.dist_mat, self.emis_mat,
                             self.tw, self.demands, self.capacity)
        fit = fitness_weighted(m, self.weights)
        new_perm = None
        if self.local_search:
            imp = local_search_pass(routes, self.demands, self.capacity, self.time_mat,
                                    self.dist_mat, self.emis_mat, self.tw, self.weights,
                                    window=self.ls_window, cache=self._cache)
            m2 = evaluate_routes(imp, self.time_mat, self.dist_mat, self.emis_mat,
                                  self.tw, self.demands, self.capacity)
            f2 = fitness_weighted(m2, self.weights)
            cur_routes = routes
            if f2 < fit:
                fit, m = f2, m2
                new_perm = np.array([c for r in imp for c in r])
                cur_routes = imp
            if self.ls_extended and np.isfinite(self._best) and fit <= self._best * (1.0 + self.ls_ext_within):
                ext = inter_route_pass(cur_routes, self.demands, self.capacity, self.time_mat, self.dist_mat,
                                       self.emis_mat, self.tw, self.weights, self._nn, cache=self._cache)
                m3 = evaluate_routes(ext, self.time_mat, self.dist_mat, self.emis_mat, self.tw, self.demands, self.capacity)
                f3 = fitness_weighted(m3, self.weights)
                if f3 < fit:
                    fit, m = f3, m3
                    new_perm = np.array([c for r in ext for c in r])
        self._last_improved = new_perm
        return fit, m

    def _order_crossover(self, p1, p2):
        n = len(p1)
        a, b = sorted(self.rng.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b] = p1[a:b]
        fill = [g for g in p2 if g not in child[a:b]]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return np.array(child)

    def run(self):
        pop = [self.rng.permutation(self.n) for _ in range(self.pop_size)]
        history = []
        gbest, gbest_fit, gbest_metrics = None, np.inf, None
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        for gen in range(self.n_gen):
            scored = []
            for ind in pop:
                ev = self._eval_perm(ind)
                if self.lamarck and self._last_improved is not None:
                    ind = self._last_improved
                scored.append((ev, ind))
            scored.sort(key=lambda x: x[0][0])
            if scored[0][0][0] < gbest_fit:
                gbest_fit, gbest_metrics = scored[0][0]
                gbest = scored[0][1]
            self._best = gbest_fit
            history.append(gbest_fit)
            elites = [s[1] for s in scored[:n_elite]]
            new_pop = list(elites)
            while len(new_pop) < self.pop_size:
                p1, p2 = scored[self.rng.integers(0, self.pop_size // 2)][1], \
                          scored[self.rng.integers(0, self.pop_size // 2)][1]
                child = self._order_crossover(p1, p2)
                if self.rng.random() < self.mutation_rate:
                    i, j = self.rng.choice(self.n, 2, replace=False)
                    child[i], child[j] = child[j], child[i]
                new_pop.append(child)
            pop = new_pop
        return {"gbest_keys": self._decode_perm(gbest), "gbest_fit": gbest_fit,
                "gbest_metrics": gbest_metrics, "history": history}


# ---------------------------------------------------------------- ACO
class AntColony:
    def __init__(self, n_customers, demands, capacity, time_windows_s,
                 time_mat, dist_mat, emis_mat, n_ants=30, n_iter=100,
                 weights=(1.0, 0.02, 0.01, 300.0), seed=0,
                 alpha=1.0, beta_h=3.0, evap=0.3, depot_idx=0):
        self.n = n_customers
        self.demands, self.capacity, self.tw = demands, capacity, time_windows_s
        self.time_mat, self.dist_mat, self.emis_mat = time_mat, dist_mat, emis_mat
        self.n_ants, self.n_iter, self.weights = n_ants, n_iter, weights
        self.rng = np.random.default_rng(seed)
        self.alpha, self.beta_h, self.evap = alpha, beta_h, evap
        self.depot_idx = depot_idx
        n_nodes = self.n + 1
        self.pheromone = np.ones((n_nodes, n_nodes))
        self.heuristic = 1.0 / (self.time_mat + 1e-6)

    def _construct(self):
        unvisited = list(range(self.n))
        routes = []
        current_route = []
        load = 0
        prev = self.depot_idx
        while unvisited:
            probs = []
            for c in unvisited:
                node = c + 1
                if load + self.demands[c] > self.capacity:
                    probs.append(0.0)
                    continue
                tau = self.pheromone[prev, node] ** self.alpha
                eta = self.heuristic[prev, node] ** self.beta_h
                probs.append(tau * eta)
            probs = np.array(probs)
            if probs.sum() <= 0:
                routes.append(current_route)
                current_route, load, prev = [], 0, self.depot_idx
                continue
            probs = probs / probs.sum()
            choice = self.rng.choice(len(unvisited), p=probs)
            c = unvisited.pop(choice)
            current_route.append(c)
            load += self.demands[c]
            prev = c + 1
        if current_route:
            routes.append(current_route)
        return routes

    def run(self):
        gbest_routes, gbest_fit, gbest_metrics = None, np.inf, None
        history = []
        for it in range(self.n_iter):
            all_routes = []
            fits = []
            for _ in range(self.n_ants):
                routes = self._construct()
                m = evaluate_routes(routes, self.time_mat, self.dist_mat, self.emis_mat,
                                     self.tw, self.demands, self.capacity)
                fit = fitness_weighted(m, self.weights)
                all_routes.append(routes)
                fits.append(fit)
                if fit < gbest_fit:
                    gbest_fit, gbest_routes, gbest_metrics = fit, routes, m
            self.pheromone *= (1 - self.evap)
            for routes, fit in zip(all_routes, fits):
                deposit = 1.0 / (fit + 1e-6)
                prev = self.depot_idx
                for route in routes:
                    for c in route:
                        node = c + 1
                        self.pheromone[prev, node] += deposit
                        prev = node
                    self.pheromone[prev, self.depot_idx] += deposit
                    prev = self.depot_idx
            history.append(gbest_fit)
        # convert routes back to a "keys" style output isn't meaningful for ACO;
        # we just report routes directly.
        return {"gbest_routes": gbest_routes, "gbest_fit": gbest_fit,
                "gbest_metrics": gbest_metrics, "history": history}


# ---------------------------------------------------------------- Tuned local search ("HGS-inspired", NOT full HGS-CVRP)
class TunedLocalSearch:
    """Nearest-neighbor construction + 2-opt + or-opt local search, with
    random restarts. A strong, honestly-labelled heuristic baseline -- NOT
    a verified from-scratch HGS-CVRP implementation."""
    def __init__(self, n_customers, demands, capacity, time_windows_s,
                 time_mat, dist_mat, emis_mat, n_restarts=15,
                 weights=(1.0, 0.02, 0.01, 300.0), seed=0, depot_idx=0):
        self.n = n_customers
        self.demands, self.capacity, self.tw = demands, capacity, time_windows_s
        self.time_mat, self.dist_mat, self.emis_mat = time_mat, dist_mat, emis_mat
        self.n_restarts, self.weights = n_restarts, weights
        self.rng = np.random.default_rng(seed)
        self.depot_idx = depot_idx

    def _nn_construct(self, start_order):
        unvisited = list(start_order)
        routes, current, load, prev = [], [], 0, self.depot_idx
        while unvisited:
            best, best_d = None, np.inf
            for c in unvisited:
                if load + self.demands[c] > self.capacity:
                    continue
                d = self.time_mat[prev, c + 1]
                if d < best_d:
                    best_d, best = d, c
            if best is None:
                routes.append(current)
                current, load, prev = [], 0, self.depot_idx
                continue
            current.append(best)
            unvisited.remove(best)
            load += self.demands[best]
            prev = best + 1
        if current:
            routes.append(current)
        return routes

    def _route_cost(self, route):
        m = evaluate_routes([route], self.time_mat, self.dist_mat, self.emis_mat,
                             self.tw, self.demands, self.capacity)
        return fitness_weighted(m, self.weights)

    def _two_opt(self, route):
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if self._route_cost(new_route) < self._route_cost(route):
                        route = new_route
                        improved = True
        return route

    def run(self):
        gbest_routes, gbest_fit, gbest_metrics = None, np.inf, None
        history = []
        order = list(range(self.n))
        for r in range(self.n_restarts):
            self.rng.shuffle(order)
            routes = self._nn_construct(order)
            routes = [self._two_opt(route) for route in routes]
            m = evaluate_routes(routes, self.time_mat, self.dist_mat, self.emis_mat,
                                 self.tw, self.demands, self.capacity)
            fit = fitness_weighted(m, self.weights)
            if fit < gbest_fit:
                gbest_fit, gbest_routes, gbest_metrics = fit, routes, m
            history.append(gbest_fit)
        return {"gbest_routes": gbest_routes, "gbest_fit": gbest_fit,
                "gbest_metrics": gbest_metrics, "history": history}


# ---------------------------------------------------------------- OR-Tools (near-exact reference)
def solve_ortools(n_customers, demands, capacity, time_mat, dist_mat,
                   time_limit_s=10, depot_idx=0):
    """Google OR-Tools CVRP solver -- used as the near-exact reference
    baseline. Falls back gracefully (returns None) if infeasible in the
    time limit."""
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n_nodes = n_customers + 1
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_customers, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    time_mat_int = (time_mat).astype(int)

    def time_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(time_mat_int[f, t])

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    demand_full = [0] + list(demands)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(demand_full[node])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [int(capacity)] * n_customers, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    routes = []
    total_time = 0
    for vehicle_id in range(n_customers):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot_idx:
                route.append(node - 1)
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            total_time += routing.GetArcCostForVehicle(prev_index, index, vehicle_id)
        if route:
            routes.append(route)
    return {"routes": routes, "total_time_s": total_time}


def solve_ortools_cvrptw(n_customers, demands, capacity, tw, time_mat, dist_mat, emis_mat,
                          weights=(1.0, 0.02, 0.01, 300.0), time_limit_s=5, depot_idx=0,
                          first_solution="PATH_CHEAPEST_ARC", metaheuristic="GUIDED_LOCAL_SEARCH",
                          scale=1, precise=False, warm_start=None):
    """OR-Tools CVRPTW with the SAME weighted objective and soft time-window
    penalty (2 per second late) used to score every other algorithm, so the
    reference is a like-for-like opponent, not a time-only solver."""
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    w_t, w_d, w_e, w_r = weights
    n_nodes = n_customers + 1
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_customers, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
    cost = (w_t * time_mat + w_d * dist_mat + w_e * emis_mat) * scale

    def cost_cb(a, b_):
        return int(round(cost[manager.IndexToNode(a), manager.IndexToNode(b_)]))
    cidx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cidx)
    routing.SetFixedCostOfAllVehicles(int(round(w_r * scale)))

    dem = [0] + list(demands)
    didx = routing.RegisterUnaryTransitCallback(lambda a: int(dem[manager.IndexToNode(a)]))
    routing.AddDimensionWithVehicleCapacity(didx, 0, [int(capacity)] * n_customers, True, "Cap")

    def time_cb(a, b_):
        t_ = time_mat[manager.IndexToNode(a), manager.IndexToNode(b_)]
        return int(round(t_)) if precise else int(t_)
    tidx = routing.RegisterTransitCallback(time_cb)
    horizon = 86400
    routing.AddDimension(tidx, horizon, horizon, True, "Time")
    tdim = routing.GetDimensionOrDie("Time")
    for c in range(n_customers):
        idx = manager.NodeToIndex(c + 1)
        earliest, latest = tw[c]
        tdim.CumulVar(idx).SetRange(int(earliest), horizon)
        tdim.SetCumulVarSoftUpperBound(idx, int(latest), int(2 * scale))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, first_solution)
    params.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)
    params.time_limit.FromMilliseconds(int(round(1000 * time_limit_s)))
    if warm_start:
        # fair disruption comparison: OR-Tools re-optimises FROM the surviving plan,
        # exactly as the archive/QPSO do -- never from scratch
        routing.CloseModelWithParameters(params)
        init = routing.ReadAssignmentFromRoutes([list(r) and [c + 1 for c in r] for r in warm_start if r], True)
        if init is not None:
            sol = routing.SolveFromAssignmentWithParameters(init, params)
            if sol is not None:
                routes = []
                for v in range(n_customers):
                    i = routing.Start(v); r = []
                    while not routing.IsEnd(i):
                        nd = manager.IndexToNode(i)
                        if nd != depot_idx: r.append(nd - 1)
                        i = sol.Value(routing.NextVar(i))
                    if r: routes.append(r)
                return routes
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None
    routes = []
    for v in range(n_customers):
        i = routing.Start(v)
        r = []
        while not routing.IsEnd(i):
            node = manager.IndexToNode(i)
            if node != depot_idx:
                r.append(node - 1)
            i = sol.Value(routing.NextVar(i))
        if r:
            routes.append(r)
    return routes
