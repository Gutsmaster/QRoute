"""
The four real fixes identified as missing from the first pass:

1. split_optimal        - DP split of a giant tour into capacity-feasible
                           routes, minimizing total cost (replaces the dumb
                           greedy "fill until capacity" split).
2. local_search_pass     - windowed 2-opt (within-route) + Or-opt (relocate
                           a single customer, possibly to a different route)
                           - applied every iteration to every particle, not
                           just occasionally to gbest.
3. clarke_wright_savings - classic savings-algorithm construction, used to
                           seed part of the initial swarm instead of pure
                           random noise.
4. routes_to_keys        - re-encodes an improved route set back into
                           random-key form so PSO position updates keep
                           working after local search / DP-split changes it.
"""
import numpy as np


# ---------------------------------------------------------------- 1. optimal split
def route_cost(route, time_mat, dist_mat, emis_mat, time_windows_s, demands,
                weights, depot_idx=0, cache=None):
    if cache is not None:
        key = tuple(route)
        hit = cache.get(key)
        if hit is not None:
            return hit
    w_t, w_d, w_e, _ = weights
    load = sum(demands[c] for c in route)
    penalty = max(0, load - 1e18)  # capacity checked by caller; keep 0 here
    t = d = e = 0.0
    prev = depot_idx
    clock = 0.0
    for c in route:
        node = c + 1
        t += time_mat[prev, node]; d += dist_mat[prev, node]; e += emis_mat[prev, node]
        clock += time_mat[prev, node]
        earliest, latest = time_windows_s[c]
        if clock < earliest:
            clock = earliest
        if clock > latest:
            penalty += (clock - latest) * 2.0
        prev = node
    t += time_mat[prev, depot_idx]; d += dist_mat[prev, depot_idx]; e += emis_mat[prev, depot_idx]
    result = w_t * t + w_d * d + w_e * e + penalty
    if cache is not None:
        cache[key] = result
    return result


def split_optimal(order, demands, capacity, time_mat, dist_mat, emis_mat,
                   time_windows_s, weights, depot_idx=0, max_route_len=None, cache=None):
    """
    Prins-style optimal split: given a fixed customer visiting ORDER (the
    giant tour), find the cost-minimizing way to cut it into capacity-
    feasible routes via DP. O(n^2) edges, each edge cost O(segment length).
    This is what a real random-key VRP solver uses -- greedy "fill until
    capacity" (the original implementation) ignores cost entirely when
    deciding where to cut.
    """
    n = len(order)
    if max_route_len is None:
        max_route_len = n
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[0] = 0.0
    parent = [-1] * (n + 1)

    for i in range(n):
        if dp[i] == INF:
            continue
        load = 0
        seg = []
        for j in range(i, min(n, i + max_route_len)):
            c = order[j]
            load += demands[c]
            if load > capacity:
                break
            seg.append(c)
            w_route = weights[3]  # per-route fixed cost
            cost = dp[i] + route_cost(seg, time_mat, dist_mat, emis_mat,
                                       time_windows_s, demands, weights, depot_idx, cache) + w_route
            if cost < dp[j + 1]:
                dp[j + 1] = cost
                parent[j + 1] = i

    if dp[n] == INF:
        # shouldn't happen (single-customer routes always feasible unless
        # demand[c] > capacity), but fall back to greedy split for safety
        from qpso_core import decode_random_key
        keys = np.argsort(np.argsort(order)).astype(float) / n
        return decode_random_key(keys, list(range(n)), demands, capacity)

    routes = []
    j = n
    while j > 0:
        i = parent[j]
        routes.append(order[i:j])
        j = i
    routes.reverse()
    return routes


# ---------------------------------------------------------------- 2. local search
def local_search_pass(routes, demands, capacity, time_mat, dist_mat, emis_mat,
                       time_windows_s, weights, depot_idx=0, window=6, cache=None):
    """
    One pass of windowed 2-opt (within route) + Or-opt (relocate one customer
    within its route or into the adjacent route). Route costs are memoized
    via `cache` (a dict owned by the caller) since converging swarms
    re-evaluate the same routes constantly -- this is a pure speed-up, it
    does not change which moves are accepted.
    """
    routes = [list(r) for r in routes]
    rc = lambda r: route_cost(r, time_mat, dist_mat, emis_mat, time_windows_s,
                              demands, weights, depot_idx, cache)

    for ridx, route in enumerate(routes):
        improved, passes = True, 0
        cur = rc(route)
        while improved and passes < 2:
            improved = False
            passes += 1
            for i in range(len(route) - 1):
                for j in range(i + 1, min(i + window, len(route))):
                    cand = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    c = rc(cand)
                    if c < cur - 1e-9:
                        route, cur, improved = cand, c, True
            routes[ridx] = route

    for ridx in range(len(routes)):
        i = 0
        while i < len(routes[ridx]):
            route = routes[ridx]
            c = route[i]
            base_cost = rc(route)
            trial = route[:i] + route[i + 1:]
            trial_cost = rc(trial) if trial else 0.0
            best_delta, best_move = -1e-9, None
            for pos in range(max(0, i - window), min(len(trial) + 1, i + window)):
                cand = trial[:pos] + [c] + trial[pos:]
                delta = rc(cand) - base_cost
                if delta < best_delta:
                    best_delta, best_move = delta, ("same", cand)
            for oidx in (ridx - 1, ridx + 1):
                if 0 <= oidx < len(routes):
                    other = routes[oidx]
                    if sum(demands[x] for x in other) + demands[c] > capacity:
                        continue
                    other_base = rc(other)
                    for pos in range(0, min(len(other) + 1, window + 1)):
                        cand_o = other[:pos] + [c] + other[pos:]
                        delta = (trial_cost + rc(cand_o)) - (base_cost + other_base)
                        if delta < best_delta:
                            best_delta, best_move = delta, ("other", oidx, cand_o)
            if best_move is None:
                i += 1
            elif best_move[0] == "same":
                routes[ridx] = best_move[1]
                i += 1
            else:
                routes[ridx] = trial
                routes[best_move[1]] = best_move[2]
                # customer moved away; index i now points at the next customer
    return [r for r in routes if r]


# ---------------------------------------------------------------- 3. Clarke-Wright savings construction
def clarke_wright_savings(demands, capacity, time_mat, depot_idx=0):
    """
    Classic Clarke-Wright savings algorithm: start with one route per
    customer, greedily merge the pair of routes with the highest "savings"
    (cost saved by joining them into one route via the depot) as long as
    capacity allows. Produces a strong, non-random initial solution -- used
    to SEED part of the swarm instead of every particle starting from pure
    noise.
    """
    n = len(demands)
    customers = list(range(n))
    routes = {c: [c] for c in customers}
    route_of = {c: c for c in customers}

    savings = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = (time_mat[depot_idx, i + 1] + time_mat[depot_idx, j + 1]
                 - time_mat[i + 1, j + 1])
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    for s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        route_i, route_j = routes[ri], routes[rj]
        if route_i[-1] != i or route_j[0] != j:
            continue  # only merge at the endpoints (classic C-W rule)
        load = sum(demands[c] for c in route_i) + sum(demands[c] for c in route_j)
        if load > capacity:
            continue
        merged = route_i + route_j
        new_id = ri
        routes[new_id] = merged
        del routes[rj]
        for c in merged:
            route_of[c] = new_id

    return list(routes.values())


# ---------------------------------------------------------------- 4. re-encode routes -> keys
def routes_to_keys(routes, n):
    """Flattens routes back into a giant-tour order, then converts that
    order into rank-based keys in [0,1] so PSO's continuous position update
    can keep operating on it next iteration."""
    order = [c for route in routes for c in route]
    # any customer somehow dropped (shouldn't happen) gets appended
    missing = [c for c in range(n) if c not in order]
    order.extend(missing)
    ranks = np.argsort(np.argsort(order))
    keys = np.zeros(n)
    for pos, c in zip(ranks, order):
        keys[c] = pos / n
    return keys
