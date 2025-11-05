from __future__ import annotations
from typing import Dict, Tuple
import numpy as np, pandas as pd, geopandas as gpd, osmnx as ox
from shapely.geometry import Point
from .geo_utils import pick_utm_crs, to_projected, make_buffer


# =============================================================================
# Helper utilities
# =============================================================================

def _bbox_from_gdf(gdf: gpd.GeoDataFrame, margin_deg: float = 0.005) -> Tuple[float,float,float,float]:
    """Return (north, south, east, west) with a small margin in degrees."""
    g = gdf.to_crs(4326)
    minx, miny, maxx, maxy = g.total_bounds
    west, south, east, north = minx - margin_deg, miny - margin_deg, maxx + margin_deg, maxy + margin_deg
    return north, south, east, west


def _project_both(line_wgs84: gpd.GeoDataFrame, gdf_osm: gpd.GeoDataFrame):
    """Project both GeoDataFrames to the same UTM CRS."""
    utm = pick_utm_crs(line_wgs84)
    return to_projected(line_wgs84, utm), to_projected(gdf_osm, utm)


# =============================================================================
# Bridges and tunnels
# =============================================================================
def bridges_tunnels_counts(line_wgs84: gpd.GeoDataFrame,
                           buffer_m: float = 30,
                           debug_plot: bool = True) -> Dict[str, float]:
    """
    Count and measure bridges/tunnels crossed by the ride.

    Uses distance-based matching and optional debug plotting to verify overlap.
    """
    import matplotlib.pyplot as plt

    n, s, e, w = _bbox_from_gdf(line_wgs84, margin_deg=0.03)
    bbox = (w, s, e, n)

    g_bridge = ox.features.features_from_bbox(bbox=bbox, tags={"bridge": True})
    g_tunnel = ox.features.features_from_bbox(bbox=bbox, tags={"tunnel": True})

    # --- Clean & CRS fix ---
    for gdf in (g_bridge, g_tunnel):
        gdf = gdf[gdf.geometry.notna()]
        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)

    g_bridge = g_bridge[g_bridge.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    g_tunnel = g_tunnel[g_tunnel.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    line_p, g_bridge_p = _project_both(line_wgs84, g_bridge)
    _, g_tunnel_p = _project_both(line_wgs84, g_tunnel)
    buf = make_buffer(line_p, buffer_m)
    buf_poly = buf.geometry.iloc[0]
    line_geom = line_p.geometry.iloc[0]

    def _stats(gdf, label):
        if gdf.empty:
            return 0, 0.0, 0.0
        ids, lengths = set(), []
        for osmid, row in gdf.iterrows():
            if row.geometry.distance(line_geom) <= buffer_m:
                inter = row.geometry.intersection(buf_poly)
                if not inter.is_empty:
                    L = inter.length
                    if L > 0:
                        ids.add(osmid)
                        lengths.append(L)
        print(f"{label}: {len(ids)} detected within {buffer_m} m")
        return len(ids), float(sum(lengths)), float(max(lengths)) if lengths else 0.0

    cb, mb, lb = _stats(g_bridge_p, "Bridges")
    ct, mt, lt = _stats(g_tunnel_p, "Tunnels")

    result = {
        "count_bridges": int(cb),
        "meters_on_bridges": round(mb, 1),
        "longest_bridge_m": round(lb, 1),
        "count_tunnels": int(ct),
        "meters_in_tunnels": round(mt, 1),
        "longest_tunnel_m": round(lt, 1),
    }

    # --- Optional visual debug ---
    if debug_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        g_bridge_p.plot(ax=ax, color="blue", alpha=0.4, label="Bridges")
        g_tunnel_p.plot(ax=ax, color="green", alpha=0.4, label="Tunnels")
        buf.plot(ax=ax, color="orange", alpha=0.2, label=f"{buffer_m} m buffer")
        line_p.plot(ax=ax, color="red", linewidth=2, label="Ride")
        ax.legend()
        ax.set_title("Bridge/Tunnel Overlap Debug")
        plt.show()

    print("📊 Bridge/Tunnel summary:", result)
    return result


# =============================================================================
# Intersections
# =============================================================================
def intersections_gdf(line_wgs84: gpd.GeoDataFrame, node_buffer_m: float = 10) -> gpd.GeoDataFrame:
    """Return GeoDataFrame of intersection nodes near the ride (degree ≥ 3)."""
    n, s, e, w = _bbox_from_gdf(line_wgs84)
    bbox = (w, s, e, n)

    G = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type="all",
        simplify=True,
        retain_all=False,
        truncate_by_edge=False
    )

    # ✅ In OSMnx 2.x, graph_to_gdfs returns only the requested layer
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)

    # Keep intersections (street_count >= 3) or degree >= 3 fallback
    if "street_count" in nodes.columns:
        nodes = nodes[nodes["street_count"] >= 3].copy()
    else:
        import networkx as nx
        deg = pd.Series(dict(nx.degree(G))).rename("degree")
        nodes = nodes.join(deg).query("degree >= 3")

    if nodes.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=4326)

    utm = pick_utm_crs(line_wgs84)
    line_p = to_projected(line_wgs84, utm)
    nodes_p = nodes.to_crs(utm)
    buf = make_buffer(line_p, node_buffer_m)
    nodes_p = nodes_p[nodes_p.intersects(buf.geometry.iloc[0])]
    return nodes_p.to_crs(4326)


# =============================================================================
# Meaningful turns (heading change + intersection proximity)
# =============================================================================
def _bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    br = np.degrees(np.arctan2(x, y))
    return (br + 360) % 360


def turns_at_intersections(df: pd.DataFrame, line_wgs84: gpd.GeoDataFrame,
                           smooth_window=9, buffer_m=15.0,
                           left_deg=30.0, right_deg=-30.0, straight_deg=20.0) -> Dict[str, int]:
    """Detect only turns that occur near OSM intersections."""
    if len(df) < 3:
        return {"turns_left_intersection": 0, "turns_right_intersection": 0, "turns_straight_intersection": 0}

    lat, lon = df["lat"].astype(float).to_numpy(), df["lon"].astype(float).to_numpy()
    br = np.array([_bearing_deg(lat[i], lon[i], lat[i + 1], lon[i + 1]) for i in range(len(df) - 1)])
    s = pd.Series(br).rolling(window=smooth_window, center=True, min_periods=1).median().to_numpy()
    d = np.diff(s)
    d = (d + 180) % 360 - 180  # normalize -180..180

    nodes = intersections_gdf(line_wgs84, node_buffer_m=buffer_m)
    if nodes.empty:
        return {"turns_left_intersection": 0, "turns_right_intersection": 0, "turns_straight_intersection": 0}

    points = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs=4326)
    join_idx = nodes.sindex.nearest(points.geometry, return_all=False)[1]
    idxs = np.unique(join_idx[join_idx < len(d)])

    lefts = (d[idxs] >= left_deg).sum()
    rights = (d[idxs] <= right_deg).sum()
    straights = (np.abs(d[idxs]) <= straight_deg).sum()

    return {
        "turns_left_intersection": int(lefts),
        "turns_right_intersection": int(rights),
        "turns_straight_intersection": int(straights),
    }
