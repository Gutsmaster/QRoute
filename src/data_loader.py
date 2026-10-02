"""
Parses the raw Delhi NCR probe-count GeoJSON files (one per day) into a
clean per-segment, per-hour table.

Raw format per day file:
  feature[0]  -> metadata (timeSets definition, dateRanges) -- SKIPPED
  feature[1:] -> one feature per road segment:
      geometry.coordinates : LineString [[lon,lat], ...]
      properties.segmentId, streetName, speedLimit, frc, distance (m)
      properties.segmentProbeCounts : [{timeSet, dateRange, probeCount}, ...]

timeSet ids 2..25 map to the 24 hourly bins 00:00-01:00 ... 23:00-00:00
(offset by +2 because timeSet id 1 is unused / reserved in this export).
"""
import ijson
import json
from decimal import Decimal
import numpy as np
import pandas as pd


def _dec_to_float(obj):
    """Recursively convert Decimal -> float so json/numpy don't choke."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [_dec_to_float(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dec_to_float(v) for k, v in obj.items()}
    return obj


def load_day(geojson_path, max_segments=None):
    """
    Stream-parses one day's GeoJSON file.
    Returns a DataFrame: one row per (segment, hour) with columns
      segment_id, street_name, speed_limit_kmh, frc, distance_m,
      lon0, lat0, lon1, lat1, hour, probe_count
    """
    rows = []
    with open(geojson_path, "rb") as f:
        feature_iter = ijson.items(f, "features.item")
        seg_count = 0
        for feat in feature_iter:
            geom = feat.get("geometry")
            if geom is None:
                continue  # metadata feature
            props = feat["properties"]
            if "segmentProbeCounts" not in props:
                continue
            coords = _dec_to_float(geom["coordinates"])
            if len(coords) < 2:
                continue
            lon0, lat0 = coords[0]
            lon1, lat1 = coords[-1]
            seg_id = props.get("segmentId")
            street = props.get("streetName") or "unnamed"
            speed_limit = float(props.get("speedLimit") or 30)
            frc = int(props.get("frc") if props.get("frc") is not None else 6)
            distance = float(_dec_to_float(props.get("distance") or 0.0))

            for pc in props["segmentProbeCounts"]:
                hour = int(pc["timeSet"]) - 2  # timeSet 2 -> hour 0
                if hour < 0 or hour > 23:
                    continue
                rows.append((
                    seg_id, street, speed_limit, frc, distance,
                    lon0, lat0, lon1, lat1, hour, int(pc["probeCount"]),
                ))
            seg_count += 1
            if max_segments is not None and seg_count >= max_segments:
                break

    df = pd.DataFrame(rows, columns=[
        "segment_id", "street_name", "speed_limit_kmh", "frc", "distance_m",
        "lon0", "lat0", "lon1", "lat1", "hour", "probe_count",
    ])
    return df


def bbox_filter(df, min_lon, min_lat, max_lon, max_lat):
    """Keep only segments whose midpoint lies inside the given bounding box."""
    mid_lon = (df["lon0"] + df["lon1"]) / 2
    mid_lat = (df["lat0"] + df["lat1"]) / 2
    mask = (
        (mid_lon >= min_lon) & (mid_lon <= max_lon)
        & (mid_lat >= min_lat) & (mid_lat <= max_lat)
    )
    return df[mask].copy()


if __name__ == "__main__":
    import sys
    df = load_day(sys.argv[1], max_segments=500)
    print(df.head())
    print(df.shape)
