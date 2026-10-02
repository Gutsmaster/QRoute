"""
Quantum Particle Swarm Optimization (delta-potential-well variant, Sun et al.)
applied to VRP via random-key encoding.

Encoding: a particle is a real vector in [0,1]^n_customers ("priority keys").
Decoding (split): sort customers by key ascending -> giant tour -> greedily
slice into vehicle routes respecting capacity (classic random-key VRP split).

Update rule (delta-potential well):
  mbest      = mean of personal-best positions across the swarm
  p          = phi * pbest + (1-phi) * gbest              (phi ~ U(0,1))
  x_new      = p +/- beta * |mbest - x| * ln(1/u)          (u ~ U(0,1))
  beta (contraction-expansion coefficient) decays linearly across iterations,
  giving the exploration->exploitation balance QPSO is chosen for.
"""
import numpy as np
from local_search import split_optimal, local_search_pass, clarke_wright_savings, routes_to_keys, make_neighbors, inter_route_pass


def decode_random_key(keys, customers, demands, capacity, depot=0):
    """Random-key decode: sort by key, greedily split into capacity-feasible
    routes. Returns list of routes, each a list of customer indices
    (0-based into `customers`)."""
    order = np.argsort(keys)
    routes = []
    current = []
    load = 0
    for idx in order:
        d = demands[idx]
        if load + d > capacity and current:
            routes.append(current)
            current = []
            load = 0
        current.append(idx)
        load += d
    if current:
        routes.append(current)
    return routes


def evaluate_routes(routes, time_mat, dist_mat, emis_mat, time_windows_s,
                     demands, capacity, depot_idx=0):
    """
    Computes total time, distance, emissions, capacity/time-window penalty
    for a full route plan. Indices in `routes` are 1-based positions into
    the instance's node list where 0 is the depot; here we pass customer
    indices already offset by +1 by the caller for matrix lookups.
    """
    total_time = total_dist = total_emis = 0.0
    penalty = 0.0
    n_routes = len(routes)
    route_times = []
    for route in routes:
        load = sum(demands[c] for c in route)
        if load > capacity:
            penalty += (load - capacity) * 500.0
        t = 0.0
        d = 0.0
        e = 0.0
        prev = depot_idx
        clock = 0.0
        for c in route:
            node_idx = c + 1  # +1 because index 0 in matrices is the depot
            t += time_mat[prev, node_idx]
            d += dist_mat[prev, node_idx]
            e += emis_mat[prev, node_idx]
            clock += time_mat[prev, node_idx]
            earliest, latest = time_windows_s[c]
            if clock < earliest:
                clock = earliest  # wait
            if clock > latest:
                penalty += (clock - latest) * 2.0
            prev = node_idx
        # return to depot
        t += time_mat[prev, depot_idx]
        d += dist_mat[prev, depot_idx]
        e += emis_mat[prev, depot_idx]
        total_time += t
        total_dist += d
        total_emis += e
        route_times.append(t)
    return {
        "total_time_s": total_time,
        "total_dist_m": total_dist,
        "total_emis_g": total_emis,
        "n_routes": n_routes,
        "penalty": penalty,
        "route_times": route_times,
    }


def fitness_weighted(metrics, weights=(1.0, 0.02, 0.01, 300.0)):
    """Scalarized objective: w_time*time + w_dist*dist + w_emis*emis
    + w_route*n_routes, plus constraint penalty (always full weight)."""
    w_t, w_d, w_e, w_r = weights
    return (
        w_t * metrics["total_time_s"]
        + w_d * metrics["total_dist_m"]
        + w_e * metrics["total_emis_g"]
        + w_r * metrics["n_routes"]
        + metrics["penalty"]
    )


def _two_opt_keys(keys, n, demands, capacity, time_mat, dist_mat, emis_mat,
                   time_windows_s, weights, max_passes=1):
    """Lightweight 2-opt refinement applied to the GIANT TOUR implied by the
    keys (sort order), then re-encoded back into keys. Used to hybridize
    QPSO into a memetic algorithm -- pure discrete QPSO via random-key
    encoding is known in the literature to often underperform local-search-
    augmented heuristics; this closes that gap honestly rather than
    reporting only the weaker pure-QPSO number."""
    order = list(np.argsort(keys))

    def cost_of_order(order):
        routes = decode_random_key(np.argsort(np.argsort(order)).astype(float) / n,
                                    list(range(n)), demands, capacity)
        m = evaluate_routes(routes, time_mat, dist_mat, emis_mat, time_windows_s, demands, capacity)
        return fitness_weighted(m, weights)

    best_cost = cost_of_order(order)
    for _ in range(max_passes):
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, min(i + 6, len(order))):  # neighborhood-limited for speed
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = cost_of_order(new_order)
                if c < best_cost:
                    order, best_cost = new_order, c
                    improved = True
        if not improved:
            break
    new_keys = np.argsort(np.argsort(order)).astype(float) / n
    return new_keys, best_cost



def _kendall_norm(order_a, order_b):
    """Normalised Kendall-tau distance (0..1) between two visiting orders."""
    n = len(order_a)
    pa = np.empty(n); pa[np.asarray(order_a)] = np.arange(n)
    pb = np.empty(n); pb[np.asarray(order_b)] = np.arange(n)
    da = pa[:, None] - pa[None, :]; db = pb[:, None] - pb[None, :]
    return float(((da * db) < 0).sum() / 2) / (n * (n - 1) / 2)


class QPSO:
    """
    Swarm optimizer with a switchable position-update rule, so QPSO and
    classical PSO can be compared with IDENTICAL infrastructure (same DP
    split, same per-particle local search, same savings seeding, same
    stagnation handling). Only `update_rule` differs:

      update_rule="quantum"   -> delta-potential-well update (QPSO)
      update_rule="classical" -> velocity/inertia update (standard PSO)

    QPSO-side options (all documented QPSO literature variants):
      weighted_mbest  : mean-best weighted by pbest fitness rank (WQPSO)
      stall_boost     : if gbest stagnates for `stall_iters` iterations,
                        ramp beta back up for `boost_len` iterations
                        (reverse-annealing-style re-exploration, now part
                        of the core loop, not only the disruption demo)
      tunnel_prob     : per-particle probability of a 'tunneling' jump --
                        a random pair swap in the particle's visiting order
                        (lets particles hop between basins the smooth
                        delta-well distribution rarely reaches)
    """
    def __init__(self, n_customers, demands, capacity, time_windows_s,
                 time_mat, dist_mat, emis_mat, n_particles=30, n_iter=120,
                 weights=(1.0, 0.02, 0.01, 300.0), seed=0,
                 beta_schedule=None, init_bias=None, memetic=False,
                 memetic_every=10, use_optimal_split=True,
                 local_search=True, ls_window=6, ls_every=1, ls_elite_frac=1.0, savings_seed_frac=0.1,
                 update_rule="quantum", rank_jump=False, attractor_mode="global",
                 attractor_k_frac=0.2, beta_start=0.45, beta_end=0.1,
                 weighted_mbest=True, elite_frac=1.0, scalar_u=False, stall_boost=False, stall_iters=8,
                 boost_len=6, tunnel_prob=0.5, tunnel_swaps=3,
                 diversity_boost=False, diversity_floor=0.08, diversity_boost_beta=0.85,
                 crossover_mode=None, crossover_frac=0.0, crossover_elite_frac=0.3,
                 writeback_mode=None, mbest_mode="mean", renormalize=False, diag=False,
                 ls_extended=False, ls_ext_within=0.02, ls_ext_k=6,
                 jump_gap_scale=False, gap_clip=(0.5, 3.0), tunnel_mode="swap", tunnel_t=1.0, tunnel_knn=4,
                 lamarck_prob=0.0, restart_std=0.0, disruption_boost=True, update_frac=1.0,
                 w_inertia=0.4, c1=1.5, c2=1.5):
        _sd = QPSO.STACK_DEFAULTS            # driver-level switch; empty by default => behaviour identical to the frozen version
        if writeback_mode is None:
            writeback_mode = _sd.get("writeback_mode")
        if not ls_extended:
            ls_extended = _sd.get("ls_extended", False)
        self.n = n_customers
        self.demands = demands
        self.capacity = capacity
        self.tw = time_windows_s
        self.time_mat = time_mat
        self.dist_mat = dist_mat
        self.emis_mat = emis_mat
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.weights = weights
        self.rng = np.random.default_rng(seed)
        self.beta_start, self.beta_end = beta_start, beta_end
        self.beta_schedule = beta_schedule or (
            lambda it, nit: self.beta_start - (self.beta_start - self.beta_end) * it / max(1, nit))
        self.init_bias = init_bias
        self.memetic = memetic
        self.memetic_every = memetic_every
        self.use_optimal_split = use_optimal_split
        self.local_search = local_search
        self.ls_window = ls_window
        self.ls_every = ls_every
        self.ls_elite_frac = ls_elite_frac
        self.savings_seed_frac = savings_seed_frac
        self.update_rule = update_rule
        self.weighted_mbest = weighted_mbest
        self.elite_frac = elite_frac
        self.scalar_u = scalar_u
        self.rank_jump = rank_jump
        self.attractor_mode = attractor_mode
        self.attractor_k_frac = attractor_k_frac
        self.stall_boost, self.stall_iters, self.boost_len = stall_boost, stall_iters, boost_len
        self.tunnel_prob = tunnel_prob
        self.tunnel_swaps = tunnel_swaps
        self.diversity_boost = diversity_boost
        self.diversity_floor = diversity_floor
        self.diversity_boost_beta = diversity_boost_beta
        self.crossover_mode = crossover_mode
        self.writeback_mode = writeback_mode
        self.ls_extended, self.ls_ext_within = ls_extended, ls_ext_within
        self._nn = make_neighbors(time_mat, ls_ext_k) if ls_extended else None
        self.ls_ext_k = ls_ext_k
        self.mbest_mode = mbest_mode
        self.renormalize = renormalize
        self.diag = {} if diag else None
        self.crossover_frac = crossover_frac
        self.crossover_elite_frac = crossover_elite_frac
        self.lamarck_prob = lamarck_prob
        self.restart_std = restart_std
        self.disruption_boost = disruption_boost
        self.update_frac = update_frac
        self.w_inertia, self.c1, self.c2 = w_inertia, c1, c2
        self.archive = []
        self._cache = {}
        self.jump_gap_scale, self.gap_clip = jump_gap_scale, gap_clip
        self.tunnel_mode, self.tunnel_t, self.tunnel_knn = tunnel_mode, tunnel_t, tunnel_knn
        self._tunnel_P = self._build_tunnel_kernel() if tunnel_mode in ("reloc_ctqw", "reloc_heat") else None

    def _init_positions(self):
        X = self.rng.random((self.n_particles, self.n))
        if self.init_bias is not None:
            X = 0.5 * X + 0.5 * (1 - self.init_bias)[None, :]
        if self.savings_seed_frac > 0:
            try:
                savings_routes = clarke_wright_savings(self.demands, self.capacity, self.time_mat)
                savings_keys = routes_to_keys(savings_routes, self.n)
                n_seed = max(1, int(self.n_particles * self.savings_seed_frac))
                for i in range(n_seed):
                    noise = self.rng.normal(0, 0.03, self.n)
                    X[i] = np.clip(savings_keys + noise, 0.0, 1.0)
            except Exception:
                pass
        return X

    def _eval(self, keys):
        if self.use_optimal_split:
            order = list(np.argsort(keys))
            routes = split_optimal(order, self.demands, self.capacity, self.time_mat,
                                    self.dist_mat, self.emis_mat, self.tw, self.weights,
                                    cache=self._cache)
        else:
            routes = decode_random_key(keys, list(range(self.n)), self.demands, self.capacity)
        metrics = evaluate_routes(routes, self.time_mat, self.dist_mat, self.emis_mat,
                                   self.tw, self.demands, self.capacity)
        fit = fitness_weighted(metrics, self.weights)
        return fit, metrics, routes

    def _dpush(self, key, val):
        self.diag.setdefault(key, []).append(float(val))

    def _renorm(self, keys):
        """Re-space a key vector uniformly while preserving its permutation
        (decoded solution and fitness are unchanged)."""
        r = np.argsort(np.argsort(keys, kind="stable"), kind="stable")
        return (r + 0.5) / self.n

    def _build_tunnel_kernel(self):
        """Row-stochastic customer->customer relocation kernel from a kNN graph on travel time.
        reloc_ctqw: |U_ij|^2 of a continuous-time quantum walk U=exp(-iAt) (t=tunnel_t, fixed a priori).
        reloc_heat: classical diffusion exp(-tL) with t chosen to MATCH the CTQW kernel's mean row entropy (control)."""
        from scipy.linalg import expm
        n = self.n
        Tm = np.array(self.time_mat, dtype=float)[1:, 1:]
        A = np.zeros((n, n))
        for c in range(n):
            d = Tm[c].copy(); d[c] = np.inf
            for j in np.argsort(d)[: self.tunnel_knn]:
                A[c, j] = A[j, c] = 1.0
        def offdiag(P):
            P = np.array(P, dtype=float); np.fill_diagonal(P, 0.0)
            return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        def mean_entropy(P):
            return float(np.mean(-np.sum(np.where(P > 0, P * np.log(P + 1e-300), 0.0), axis=1)))
        Pq = offdiag(np.abs(expm(-1j * A * self.tunnel_t)) ** 2)
        if self.tunnel_mode == "reloc_ctqw":
            return Pq
        L = np.diag(A.sum(axis=1)) - A
        target = mean_entropy(Pq)
        best = min((offdiag(expm(-t * L)) for t in np.geomspace(0.02, 20, 80)), key=lambda P: abs(mean_entropy(P) - target))
        return best

    def _tunnel_relocate(self, keys):
        """Relocate one customer next to a partner customer; keeps the particle's own key VALUES (spacing)."""
        n = self.n
        c = int(self.rng.integers(n))
        if self.tunnel_mode == "reloc_uniform":
            j = int(self.rng.integers(n - 1)); j = j + 1 if j >= c else j
        else:
            j = int(self.rng.choice(n, p=self._tunnel_P[c]))
        after = self.rng.random() < 0.5
        order = list(np.argsort(keys)); order.remove(c)
        order.insert(order.index(j) + (1 if after else 0), c)
        vals = np.sort(keys) + np.arange(n) * 1e-9
        new = np.empty(n); new[order] = vals
        return new

    def _gap_weights(self, X):
        """Per-customer jump scale = distance to the nearest other key (relative to the particle's mean gap), clipped."""
        lo, hi = self.gap_clip
        W = np.empty_like(X)
        for i in range(X.shape[0]):
            r = np.argsort(X[i]); sr = X[i][r]
            dp = np.r_[np.inf, np.diff(sr)]; dn = np.r_[np.diff(sr), np.inf]
            g = np.empty(X.shape[1]); g[r] = np.minimum(dp, dn)
            W[i] = np.clip(g / max(g.mean(), 1e-12), lo, hi)
        return W

    def _apply_rank_jump(self, p, jump, sign):
        """Applies the delta-well jump as a rank displacement rather than a raw
        key-value shift. `p` (attractor mix of pbest/gbest) sets the BASE
        permutation; `jump` (same magnitude formula as vector/scalar-u QPSO)
        sets how many rank-positions each customer moves, in `sign` direction."""
        n_particles, n = p.shape
        shift = np.clip(np.round(jump * n), 0, n - 1) * sign
        X_new = np.empty_like(p)
        for i in range(n_particles):
            rank = np.argsort(np.argsort(p[i]))  # rank[j] = current position of customer j
            new_rank = rank + shift[i]
            tie = rank / (n * 10.0)  # stable tie-break preserves relative order on collision
            new_order = np.argsort(new_rank + tie)
            keys = np.empty(n)
            keys[new_order] = np.arange(n) / n
            X_new[i] = keys
        return X_new

    def _per_particle_attractor(self, pbest, pbest_fit, gbest):
        """Global mode: every particle attracts toward the single shared gbest
        (current default). Stochastic-elite mode: each particle independently
        draws ITS OWN attractor from the top-k elite pbest set each iteration
        (rank-weighted), so the swarm pulls toward several distinct good
        solutions at once instead of collapsing onto one point -- closer to
        how a GA's population preserves multiple basins, but each particle
        still moves on its own via the quantum/classical update (no
        crossover between particles' solutions)."""
        if self.attractor_mode == "global":
            return np.broadcast_to(gbest, pbest.shape)
        finite = np.where(np.isfinite(pbest_fit), pbest_fit, np.nanmax(pbest_fit[np.isfinite(pbest_fit)]))
        order = np.argsort(finite)
        k = max(2, int(round(self.n_particles * self.attractor_k_frac)))
        elite_idx = order[:k]
        w = (k - np.arange(k)).astype(float); w /= w.sum()
        choice = self.rng.choice(k, size=self.n_particles, p=w)
        return pbest[elite_idx[choice]]

    def _order_crossover_keys(self, keys_a, keys_b):
        """Standard GA-style order crossover (OX), operating on the DECODED
        giant tour (not the swarm dynamics). Honestly a GA operator -- used
        here only as a sparse, occasional supplement to QPSO's own swarm
        search, not as the primary mechanism (see crossover_frac)."""
        n = self.n
        order_a, order_b = list(np.argsort(keys_a)), list(np.argsort(keys_b))
        a, b = sorted(self.rng.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b] = order_a[a:b]
        fill = [g for g in order_b if g not in child[a:b]]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[j]; j += 1
        keys = np.empty(n); keys[child] = np.arange(n) / n
        return keys

    def _quantum_blend_crossover(self, keys_a, keys_b):
        """Quantum-inspired recombination: a per-dimension qubit-style
        rotation angle theta ~ U(0, pi/2) blends the two parents' continuous
        KEY values as cos^2(theta)*a + sin^2(theta)*b (cos^2+sin^2=1, echoing
        a qubit measurement-probability split rather than GA's discrete
        permutation-splicing). Stays in continuous key space throughout, so
        it is mechanistically distinct from the order-crossover GA operator,
        not just a relabeling of it."""
        theta = self.rng.uniform(0, np.pi / 2, self.n)
        c2, s2 = np.cos(theta) ** 2, np.sin(theta) ** 2
        return c2 * keys_a + s2 * keys_b

    def _apply_crossover(self, X, pbest, pbest_fit):
        """Sparse steady-state supplement: pick `crossover_frac` fraction of
        the swarm's WORST particles each iteration and replace them with
        offspring bred from randomly paired ELITE pbest particles. The rest
        of the swarm (the majority) continues under pure QPSO/classical
        swarm dynamics untouched -- this is why it's a hybrid, not a GA:
        the swarm's own search remains primary, crossover only backfills
        the weakest slots."""
        if self.crossover_mode is None or self.crossover_frac <= 0:
            return X
        n_replace = max(1, int(round(self.n_particles * self.crossover_frac)))
        k_elite = max(2, int(round(self.n_particles * self.crossover_elite_frac)))
        elite_idx = np.argsort(pbest_fit)[:k_elite]
        worst_idx = np.argsort(pbest_fit)[::-1][:n_replace]
        for w in worst_idx:
            pa, pb = self.rng.choice(elite_idx, 2, replace=False)
            if self.crossover_mode == "order":
                child = self._order_crossover_keys(pbest[pa], pbest[pb])
            elif self.crossover_mode == "quantum_blend":
                child = self._quantum_blend_crossover(pbest[pa], pbest[pb])
            else:
                raise ValueError(self.crossover_mode)
            X[w] = child
        return X

    def run(self, disruption_iter=None, disruption_time_mats=None, verbose=False):
        X = self._init_positions()
        V = self.rng.uniform(-0.1, 0.1, X.shape)
        pbest = X.copy()
        pbest_fit = np.full(self.n_particles, np.inf)
        gbest, gbest_fit, gbest_metrics, gbest_routes = None, np.inf, None, None
        history = []
        stall, boost_left = 0, 0

        for it in range(self.n_iter):
            if disruption_iter is not None and it == disruption_iter and disruption_time_mats is not None:
                self.time_mat = disruption_time_mats
                self._cache.clear()
                if self.ls_extended: self._nn = make_neighbors(self.time_mat, self.ls_ext_k)
                pbest_fit[:] = np.inf
                gbest_fit = np.inf
                boost_left = self.boost_len * 3 if self.disruption_boost else 0

            prev_best = gbest_fit
            do_ls_this_iter = self.local_search and (it % self.ls_every == 0)
            n_ls = self.n_particles if self.ls_elite_frac >= 1.0 else max(1, int(round(self.n_particles * self.ls_elite_frac)))
            # rank particles by LAST iteration's pbest_fit so "elite" LS targeting is causal, not clairvoyant
            ls_target = set(np.argsort(pbest_fit)[:n_ls].tolist()) if do_ls_this_iter else set()
            for i in range(self.n_particles):
                order_pre = np.argsort(X[i]) if self.diag is not None else None
                fit, metrics, routes = self._eval(X[i])
                if self.diag is not None:
                    self._dpush("ls_improved", 0)  # overwritten below if LS improves
                if do_ls_this_iter and i in ls_target:
                    imp_routes = local_search_pass(
                        routes, self.demands, self.capacity, self.time_mat,
                        self.dist_mat, self.emis_mat, self.tw, self.weights,
                        window=self.ls_window, cache=self._cache)
                    imp_metrics = evaluate_routes(
                        imp_routes, self.time_mat, self.dist_mat, self.emis_mat,
                        self.tw, self.demands, self.capacity)
                    imp_fit = fitness_weighted(imp_metrics, self.weights)
                    if self.ls_extended and np.isfinite(gbest_fit) and min(fit, imp_fit) <= gbest_fit * (1.0 + self.ls_ext_within):
                        base_r = imp_routes if imp_fit < fit else routes
                        ext_r = inter_route_pass(base_r, self.demands, self.capacity, self.time_mat, self.dist_mat,
                                                 self.emis_mat, self.tw, self.weights, self._nn, cache=self._cache)
                        ext_m = evaluate_routes(ext_r, self.time_mat, self.dist_mat, self.emis_mat, self.tw, self.demands, self.capacity)
                        ext_f = fitness_weighted(ext_m, self.weights)
                        if ext_f < min(fit, imp_fit):
                            imp_routes, imp_metrics, imp_fit = ext_r, ext_m, ext_f
                    if imp_fit < fit:
                        fit, metrics, routes = imp_fit, imp_metrics, imp_routes
                        if self.diag is not None:
                            self.diag["ls_improved"][-1] = 1.0
                            self._dpush("desync_kendall", _kendall_norm(order_pre, [c for r in routes for c in r]))
                        if self.rng.random() < self.lamarck_prob:
                            X[i] = routes_to_keys(routes, self.n)
                        if self.writeback_mode == "sorted_reassign":
                            # minimal-disturbance write-back: keep this particle's own
                            # key VALUES (and hence their spacing), only re-assign them
                            # to customers so that argsort(keys) == improved visiting order
                            order_post = [c for r in routes for c in r]
                            vals = np.sort(X[i]) + np.arange(self.n) * 1e-9
                            newk = np.empty(self.n); newk[order_post] = vals
                            X[i] = np.clip(newk, 0.0, 1.0 + 1e-6)
                if self.renormalize:
                    X[i] = self._renorm(X[i])
                self.archive.append((fit, X[i].copy(), metrics))
                if fit < pbest_fit[i]:
                    pbest_fit[i], pbest[i] = fit, X[i].copy()
                if fit < gbest_fit:
                    gbest_fit, gbest, gbest_metrics, gbest_routes = fit, X[i].copy(), metrics, routes

            # stagnation tracking (drives the beta boost)
            if gbest_fit < prev_best - 1e-9:
                stall = 0
            else:
                stall += 1
            if self.stall_boost and stall >= self.stall_iters and boost_left == 0:
                boost_left, stall = self.boost_len, 0

            div_boost = False
            if self.diversity_boost:
                samp = X[:min(10, self.n_particles)]
                ranks = np.argsort(samp, axis=1)
                # mean fraction of positions differing between the first particle's rank order and the rest (cheap proxy)
                base = ranks[0]
                frac_diff = np.mean([(ranks[k] != base).mean() for k in range(1, len(ranks))]) if len(ranks) > 1 else 1.0
                div_boost = frac_diff < self.diversity_floor

            X_old = X.copy()
            if self.update_rule == "quantum":
                finite = np.where(np.isfinite(pbest_fit), pbest_fit, np.nanmax(pbest_fit[np.isfinite(pbest_fit)]))
                order_idx = np.argsort(finite)  # best-first
                if self.elite_frac < 1.0:
                    k = max(2, int(round(self.n_particles * self.elite_frac)))
                    elite_idx = order_idx[:k]
                else:
                    elite_idx = order_idx
                if self.weighted_mbest:
                    ranks = np.empty(len(elite_idx)); ranks[np.argsort(finite[elite_idx])] = np.arange(len(elite_idx))
                    w = (len(elite_idx) - ranks).astype(float); w /= w.sum()
                else:
                    w = np.full(len(elite_idx), 1.0 / len(elite_idx))
                if self.mbest_mode == "mean":
                    if self.weighted_mbest:
                        mbest = (w[:, None] * pbest[elite_idx]).sum(axis=0)
                    else:
                        mbest = pbest[elite_idx].mean(axis=0)
                else:
                    # Borda consensus: average the customers' RANKS across the pbest
                    # set (keys of different permutations are not meaningfully averageable)
                    rk = np.argsort(np.argsort(pbest[elite_idx], axis=1), axis=1).astype(float)
                    avg_rank = (w[:, None] * rk).sum(axis=0)
                    if self.mbest_mode == "borda_sorted":
                        cons = np.argsort(avg_rank, kind="stable")
                        mbest = np.empty(self.n); mbest[cons] = (np.arange(self.n) + 0.5) / self.n
                    else:  # "borda_mean"
                        mbest = (avg_rank + 0.5) / self.n
                if self.diag is not None:
                    self._dpush("mbest_std", np.std(mbest))
                    self._dpush("mbest_absdev", np.mean(np.abs(mbest[None, :] - X)))
                beta = self.beta_schedule(it, self.n_iter)
                if boost_left > 0:
                    beta = max(beta, 0.85)
                    boost_left -= 1
                if div_boost:
                    beta = max(beta, self.diversity_boost_beta)
                attractor = self._per_particle_attractor(pbest, pbest_fit, gbest)
                phi = self.rng.random((self.n_particles, self.n))
                p = phi * pbest + (1 - phi) * attractor
                if self.scalar_u:
                    u = np.clip(self.rng.random((self.n_particles, 1)), 1e-6, 1 - 1e-6)
                    u = np.broadcast_to(u, (self.n_particles, self.n))
                else:
                    u = np.clip(self.rng.random((self.n_particles, self.n)), 1e-6, 1 - 1e-6)
                sign = np.where(self.rng.random((self.n_particles, self.n)) > 0.5, 1.0, -1.0)
                jump = beta * np.abs(mbest[None, :] - X) * np.log(1.0 / u)
                if self.jump_gap_scale:
                    jump = jump * self._gap_weights(X)
                if self.rank_jump:
                    # Diagnosed problem: a raw-key jump of a given magnitude has an
                    # unpredictable, often tiny effect on the actual PERMUTATION
                    # (argsort is discontinuous), so the "quantum jump size" the
                    # formula computes doesn't reliably reach the solution space.
                    # Fix: use the identical delta-well magnitude/direction formula,
                    # but apply it as a RANK displacement (move customer j's rank by
                    # round(jump_j * n) positions in the direction of `sign`) instead
                    # of adding it to the raw key value. Still per-particle, still
                    # driven by the same pbest/gbest/mbest attractor -- no
                    # cross-particle crossover, so this is not a GA operator.
                    X = self._apply_rank_jump(p, jump, sign)
                else:
                    X = np.clip(p + sign * jump, 0.0, 1.0)
            else:
                beta = 0.0
                if boost_left > 0:
                    boost_left -= 1
                attractor = self._per_particle_attractor(pbest, pbest_fit, gbest)
                r1 = self.rng.random((self.n_particles, self.n))
                r2 = self.rng.random((self.n_particles, self.n))
                V = self.w_inertia * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (attractor - X)
                X = np.clip(X + V, 0.0, 1.0)

            if self.diag is not None:
                for i in range(self.n_particles):
                    o_old, o_new = np.argsort(X_old[i]), np.argsort(X[i])
                    self._dpush("native_kendall", _kendall_norm(o_old, o_new))
                    self._dpush("native_dist", np.linalg.norm(X[i] - X_old[i]))
                    self._dpush("native_changed", float(not np.array_equal(o_old, o_new)))
            if self.update_frac < 1.0:   # dimension-wise (sparse) update: only a fraction of customers move
                keep = self.rng.random(X.shape) >= self.update_frac
                X = np.where(keep, X_old, X)

            # shared (both update rules): tunneling jumps + collapse restart
            if self.tunnel_prob > 0:
                for i in range(self.n_particles):
                    if self.rng.random() < self.tunnel_prob:
                        for _ in range(int(self.rng.integers(1, self.tunnel_swaps + 1))):
                            if self.tunnel_mode == "swap":
                                a, b = self.rng.choice(self.n, 2, replace=False)
                                X[i, a], X[i, b] = X[i, b], X[i, a]
                            else:
                                X[i] = self._tunnel_relocate(X[i])
            if self.restart_std > 0 and X.std(axis=0).mean() < self.restart_std:
                worst = np.argsort(pbest_fit)[self.n_particles // 2:]
                X[worst] = self.rng.random((len(worst), self.n))
                pbest_fit[worst] = np.inf

            if self.diag is not None:
                for i in range(self.n_particles):
                    self._dpush("total_kendall", _kendall_norm(np.argsort(X_old[i]), np.argsort(X[i])))
            X = self._apply_crossover(X, pbest, pbest_fit)

            history.append(gbest_fit)
            if verbose and it % 20 == 0:
                print(f"iter {it}: gbest_fit={gbest_fit:.1f} beta={beta:.3f}")

        return {
            "gbest_keys": gbest,
            "gbest_routes": gbest_routes,
            "gbest_fit": gbest_fit,
            "gbest_metrics": gbest_metrics,
            "history": history,
        }


QPSO.STACK_DEFAULTS = {}


if __name__ == "__main__":
    from data_loader import load_day
    from calibration import build_edge_table
    from graph_builder import build_graph, largest_component_subgraph
    from vrp_instance import make_synthetic_instance

    df = load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson")
    df = build_edge_table(df)
    G = build_graph(df)
    Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645))
    inst = make_synthetic_instance(Gs, n_customers=15, capacity=100)

    demands = [inst.demands[c] for c in inst.customers]
    tw = [inst.time_windows[c] for c in inst.customers]

    qpso = QPSO(
        n_customers=len(inst.customers), demands=demands, capacity=inst.capacity,
        time_windows_s=tw, time_mat=inst.time_mat[9], dist_mat=inst.dist_mat[9],
        emis_mat=inst.emis_mat[9], n_particles=30, n_iter=100, seed=1,
    )
    result = qpso.run(verbose=True)
    print("FINAL:", result["gbest_fit"], result["gbest_metrics"])
