"""
Builds a NetworkX directed graph from calibrated segment/hour data.
Nodes = road-segment endpoints (snapped to ~1m precision so shared
intersections merge into one node). Edges = road segments, carrying a
24-length array of (time_s, congestion_pct, emissions_g) -- one per hour
of day -- so the graph is genuinely time-dependent, matching the PS's
"dynamic weight update mechanism" requirement.
"""
import networkx as nx
import numpy as np


def _snap(lon, lat, precision=5):
    return (round(lon, precision), round(lat, precision))


def build_graph(edge_df):
    """
    edge_df: output of calibration.build_edge_table (long: segment/hour rows)
    Returns a networkx.DiGraph where each edge holds:
      time_by_hour       : np.array[24]  (seconds)
      congestion_by_hour  : np.array[24]  (percent)
      emissions_by_hour   : np.array[24]  (grams CO2)
      distance_m, street_name, frc
    """
    G = nx.DiGraph()
    grouped = edge_df.groupby("segment_id")
    for seg_id, g in grouped:
        g = g.sort_values("hour")
        if len(g) < 24:
            continue  # incomplete day, skip for consistency
        first = g.iloc[0]
        u = _snap(first.lon0, first.lat0)
        v = _snap(first.lon1, first.lat1)
        if u == v:
            continue
        time_arr = g["time_s"].to_numpy()
        cong_arr = g["congestion_pct"].to_numpy()
        emis_arr = g["emissions_g"].to_numpy()
        G.add_node(u, lon=u[0], lat=u[1])
        G.add_node(v, lon=v[0], lat=v[1])
        G.add_edge(
            u, v,
            segment_id=int(seg_id),
            street_name=str(first.street_name),
            distance_m=float(first.distance_m),
            frc=int(first.frc),
            time_by_hour=time_arr,
            congestion_by_hour=cong_arr,
            emissions_by_hour=emis_arr,
        )
    return G


def largest_component_subgraph(G, bbox=None, strongly_connected=True):
    """
    Optionally restrict to a bounding box, then take the largest connected
    component -- gives a routable demo network.

    HONESTY NOTE: road segments are directed (many Delhi roads are one-way
    in this dataset). Using only WEAK connectivity can leave some node
    pairs with no directed path between them (real one-way-street
    behaviour) -- at larger customer counts this surfaced as "unreachable"
    sentinel costs in the distance matrix. For any instance where every
    customer must be reachable from and back to the depot, we take the
    largest STRONGLY connected component instead, which guarantees a
    directed path exists in both directions between every pair of nodes.
    """
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        keep = [
            n for n, d in G.nodes(data=True)
            if min_lon <= d["lon"] <= max_lon and min_lat <= d["lat"] <= max_lat
        ]
        G = G.subgraph(keep).copy()
    if strongly_connected:
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.weakly_connected_components(G))
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def edge_time(G, u, v, hour):
    return float(G[u][v]["time_by_hour"][hour % 24])


if __name__ == "__main__":
    from data_loader import load_day
    from calibration import build_edge_table

    df = load_day(
        "data/new_delhi_traffic_dataset/probe_counts/geojson/new_delhi__2024-08-12_to_2024-08-12_.geojson"
    )
    df = build_edge_table(df)
    G = build_graph(df)
    print("full graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    # central Delhi bbox (Connaught Place area, roughly)
    bbox = (77.19, 28.60, 77.26, 28.66)
    Gs = largest_component_subgraph(G, bbox)
    print("demo subgraph:", Gs.number_of_nodes(), "nodes,", Gs.number_of_edges(), "edges")
