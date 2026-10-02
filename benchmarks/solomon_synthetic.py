"""
HONESTY NOTE: this sandbox has no network access to fetch the official
Solomon (1987) / CVRPLIB instance files. Instead we GENERATE synthetic
instances that follow the same well-documented Solomon structure:
  - C-type: customers in tight geographic CLUSTERS
  - R-type: customers uniformly RANDOM
  - RC-type: a MIX of clustered and random customers
Coordinates are synthetic (unit square, distance = Euclidean * scale), not
tied to the real Delhi graph -- this is a standard-structure synthetic
sweep for cross-checking QPSO's general VRP behaviour, separate from the
real-Delhi-data benchmark. Both are reported separately and never
conflated.
"""
import numpy as np


def generate_instance(kind, n_customers=25, seed=0, capacity=100,
                       coord_scale=100.0, speed_kmh=40.0, tw_width=180):
    """kind in {'C', 'R', 'RC'}. Returns dict with xy coords, demands,
    time_windows_s, and time/dist matrices (Euclidean, single static
    'hour' -- these are not time-dependent, matching the classic Solomon
    format)."""
    rng = np.random.default_rng(seed)
    n = n_customers

    if kind == "C":
        n_clusters = max(2, n // 6)
        centers = rng.uniform(10, coord_scale - 10, size=(n_clusters, 2))
        xy = np.zeros((n, 2))
        for i in range(n):
            c = centers[rng.integers(0, n_clusters)]
            xy[i] = c + rng.normal(0, 4, size=2)
    elif kind == "R":
        xy = rng.uniform(0, coord_scale, size=(n, 2))
    elif kind == "RC":
        half = n // 2
        n_clusters = max(2, half // 6)
        centers = rng.uniform(10, coord_scale - 10, size=(n_clusters, 2))
        xy = np.zeros((n, 2))
        for i in range(half):
            c = centers[rng.integers(0, n_clusters)]
            xy[i] = c + rng.normal(0, 4, size=2)
        xy[half:] = rng.uniform(0, coord_scale, size=(n - half, 2))
    else:
        raise ValueError(kind)

    depot = np.array([[coord_scale / 2, coord_scale / 2]])
    all_xy = np.vstack([depot, xy])  # index 0 = depot

    diff = all_xy[:, None, :] - all_xy[None, :, :]
    dist_km = np.sqrt((diff ** 2).sum(-1))  # treat units as km directly
    time_s = dist_km / speed_kmh * 3600.0

    demands = rng.integers(5, 25, size=n).tolist()
    time_windows_s = []
    for i in range(n):
        start = rng.integers(0, 480) * 60
        time_windows_s.append((int(start), int(start + tw_width * 60)))

    emis_g = dist_km * 130.0  # flat emissions factor for this synthetic set

    return {
        "kind": kind, "n": n, "xy": all_xy, "demands": demands,
        "capacity": capacity, "time_windows_s": time_windows_s,
        "time_mat": time_s, "dist_mat": dist_km * 1000.0, "emis_mat": emis_g,
    }


def generate_sweep(n_per_kind=3, n_customers=25, seed_base=100):
    """Generates the C/R/RC sweep set."""
    instances = []
    for kind in ("C", "R", "RC"):
        for i in range(n_per_kind):
            inst = generate_instance(kind, n_customers=n_customers, seed=seed_base + i + {'C': 0, 'R': 10, 'RC': 20}[kind])
            inst["name"] = f"{kind}{i+1}-synthetic"
            instances.append(inst)
    return instances
