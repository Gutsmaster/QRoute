"""
Builds a concrete VRP instance from the road graph:
  - pick 1 depot + N customer nodes
  - assign demand + time windows (synthetic, since the dataset has no
    delivery-order data -- clearly labelled as such)
  - precompute, for each hour of day, the shortest-path TIME, DISTANCE,
    CONGESTION-exposure and EMISSIONS between every pair of instance nodes,
    using that hour's edge weights (time-dependent VRP, hour-snapshot
    simplification -- standard practice, documented here explicitly)
"""
import networkx as nx
import numpy as np


class VRPInstance:
    def __init__(self, G, depot, customers, demands, capacity, time_windows,
                 hours=range(24)):
        self.G = G
        self.depot = depot
        self.customers = customers
        self.nodes = [depot] + customers
        self.n = len(self.nodes)
        self.demands = demands  # dict node -> demand (depot=0)
        self.capacity = capacity
        self.time_windows = time_windows  # dict node -> (earliest_s, latest_s)
        self.hours = list(hours)
        self.time_mat = {}   # hour -> n x n matrix (seconds)
        self.dist_mat = {}   # hour -> n x n matrix (meters)
        self.cong_mat = {}   # hour -> n x n matrix (avg congestion % along path)
        self.emis_mat = {}   # hour -> n x n matrix (grams CO2)
        self._build_matrices()

    def _build_matrices(self):
        idx = {node: i for i, node in enumerate(self.nodes)}
        for hour in self.hours:
            n = self.n
            T = np.zeros((n, n))
            D = np.zeros((n, n))
            C = np.zeros((n, n))
            E = np.zeros((n, n))

            def weight(u, v, data):
                return float(data["time_by_hour"][hour])

            for src in self.nodes:
                lengths, paths = nx.single_source_dijkstra(self.G, src, weight=weight)
                for dst in self.nodes:
                    i, j = idx[src], idx[dst]
                    if src == dst:
                        continue
                    if dst not in paths:
                        T[i, j] = D[i, j] = E[i, j] = 1e9
                        C[i, j] = 100.0
                        continue
                    path = paths[dst]
                    t = d = e = 0.0
                    cvals = []
                    for a, b in zip(path[:-1], path[1:]):
                        ed = self.G[a][b]
                        t += float(ed["time_by_hour"][hour])
                        d += float(ed["distance_m"])
                        e += float(ed["emissions_by_hour"][hour])
                        cvals.append(float(ed["congestion_by_hour"][hour]))
                    T[i, j] = t
                    D[i, j] = d
                    E[i, j] = e
                    C[i, j] = float(np.mean(cvals)) if cvals else 0.0
            self.time_mat[hour] = T
            self.dist_mat[hour] = D
            self.cong_mat[hour] = C
            self.emis_mat[hour] = E


def make_synthetic_instance(G, n_customers=20, capacity=100, seed=0,
                             depart_hour=9, tw_width_s=3600 * 3):
    """
    Picks a depot + n_customers from the largest-degree nodes of G (so they
    are well-connected, avoiding dead-end stubs), and assigns SYNTHETIC
    demand + time windows. The road network and its congestion/time/
    emissions are REAL (from the dataset); the delivery demand and time
    windows are simulated, since the dataset contains no order-level data.
    """
    rng = np.random.default_rng(seed)
    nodes_by_degree = sorted(G.nodes(), key=lambda n: -G.degree(n))
    candidates = nodes_by_degree[: max(60, n_customers * 3)]
    rng.shuffle(candidates)
    depot = candidates[0]
    customers = candidates[1: 1 + n_customers]

    demands = {depot: 0}
    time_windows = {depot: (0, 24 * 3600)}
    base = depart_hour * 3600
    for c in customers:
        demands[c] = int(rng.integers(5, 25))
        start = base + int(rng.integers(-1800, 1800))
        time_windows[c] = (max(0, start), start + tw_width_s)

    return VRPInstance(
        G, depot, customers, demands, capacity, time_windows,
        hours=range(24),
    )


if __name__ == "__main__":
    from data_loader import load_day
    from calibration import build_edge_table
    from graph_builder import build_graph, largest_component_subgraph
    import time

    df = load_day(
        "data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"
    )
    df = build_edge_table(df)
    G = build_graph(df)
    Gs = largest_component_subgraph(G, (77.205, 28.625, 77.225, 28.645))
    print("subgraph:", Gs.number_of_nodes(), Gs.number_of_edges())

    t0 = time.time()
    inst = make_synthetic_instance(Gs, n_customers=15, capacity=100)
    print("instance build time:", time.time() - t0, "s")
    print("depot:", inst.depot, "customers:", len(inst.customers))
    print("time matrix @9am (s):\n", inst.time_mat[9].round(1))
