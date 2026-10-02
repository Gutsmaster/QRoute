"""
Builds outputs/map.html: an interactive Folium map showing
  - the real road subgraph (central Delhi bbox)
  - the depot and customer nodes
  - the best QPSO route(s)
  - the best Tuned-LS baseline route(s), for visual comparison
"""
import sys
import os
import folium
sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks")

from data_loader import load_day
from calibration import build_edge_table
from graph_builder import build_graph, largest_component_subgraph
from vrp_instance import make_synthetic_instance
from qpso_core import QPSO, decode_random_key
from baselines import solve_ortools_cvrptw

BBOX = (77.205, 28.625, 77.225, 28.645)
N_CUSTOMERS = 24

df = load_day("data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson")
df = build_edge_table(df)
G = build_graph(df)
Gs = largest_component_subgraph(G, BBOX, strongly_connected=True)
inst = make_synthetic_instance(Gs, n_customers=N_CUSTOMERS, capacity=100, seed=0)
demands = [inst.demands[c] for c in inst.customers]
tw = [inst.time_windows[c] for c in inst.customers]
T, D, E = inst.time_mat[9], inst.dist_mat[9], inst.emis_mat[9]

qpso_res = QPSO(N_CUSTOMERS, demands, 100, tw, T, D, E, n_particles=24, n_iter=80, seed=1).run()
qpso_routes = qpso_res["gbest_routes"]

ls_routes = solve_ortools_cvrptw(N_CUSTOMERS, demands, 100, tw, T, D, E, time_limit_s=5)

center_lat = Gs.nodes[inst.depot]["lat"]
center_lon = Gs.nodes[inst.depot]["lon"]
m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

# draw the underlying real road network lightly
for u, v, data in Gs.edges(data=True):
    folium.PolyLine(
        [[Gs.nodes[u]["lat"], Gs.nodes[u]["lon"]], [Gs.nodes[v]["lat"], Gs.nodes[v]["lon"]]],
        color="lightgray", weight=1.5, opacity=0.5,
    ).add_to(m)

# depot marker
folium.Marker(
    [center_lat, center_lon], popup="DEPOT",
    icon=folium.Icon(color="black", icon="home"),
).add_to(m)

# customer markers
for i, c in enumerate(inst.customers):
    folium.CircleMarker(
        [Gs.nodes[c]["lat"], Gs.nodes[c]["lon"]], radius=5,
        popup=f"Customer {i} (demand={demands[i]})",
        color="blue", fill=True, fill_opacity=0.7,
    ).add_to(m)


def node_path_between(Gs, src, dst, hour=9):
    import networkx as nx
    def weight(u, v, data):
        return float(data["time_by_hour"][hour])
    try:
        return nx.shortest_path(Gs, src, dst, weight=weight)
    except Exception:
        return [src, dst]


def draw_routes(routes, customers, depot, color, label_prefix):
    fg = folium.FeatureGroup(name=label_prefix)
    for r_idx, route in enumerate(routes):
        seq = [depot] + [customers[c] for c in route] + [depot]
        for a, b in zip(seq[:-1], seq[1:]):
            path = node_path_between(Gs, a, b)
            latlon = [[Gs.nodes[n]["lat"], Gs.nodes[n]["lon"]] for n in path]
            folium.PolyLine(latlon, color=color, weight=4, opacity=0.8,
                             tooltip=f"{label_prefix} route {r_idx}").add_to(fg)
    fg.add_to(m)


draw_routes(qpso_routes, inst.customers, inst.depot, "red", "QPSO route")
draw_routes(ls_routes, inst.customers, inst.depot, "green", "OR-Tools route")

folium.LayerControl().add_to(m)
os.makedirs("outputs", exist_ok=True)
m.save("outputs/map.html")
print("Saved outputs/map.html")
print("QPSO routes:", len(qpso_routes), "fit:", qpso_res["gbest_fit"])
print("OR-Tools routes:", len(ls_routes))
