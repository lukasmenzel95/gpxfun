from __future__ import annotations
import geopandas as gpd
from shapely.geometry import LineString
from pyproj import CRS

def build_line_gdf(df) -> gpd.GeoDataFrame:
    """Create a LineString GeoDataFrame (EPSG:4326) from a ride dataframe with lat/lon."""
    coords = list(zip(df["lon"].astype(float), df["lat"].astype(float)))
    return gpd.GeoDataFrame({"ride_id": [0]}, geometry=[LineString(coords)], crs="EPSG:4326")

def pick_utm_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Pick a local UTM CRS so that distance and buffer units are in meters."""
    cen = gdf.to_crs(4326).geometry.iloc[0].centroid
    lon, lat = cen.x, cen.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def to_projected(gdf: gpd.GeoDataFrame, crs: CRS) -> gpd.GeoDataFrame:
    """Reproject to a given CRS (used for meters-based ops)."""
    return gdf.to_crs(crs)

def make_buffer(gdf_line: gpd.GeoDataFrame, meters: float) -> gpd.GeoDataFrame:
    """Buffer the line by N meters (requires projected CRS)."""
    assert gdf_line.crs and not gdf_line.crs.is_geographic, "Need projected CRS for buffer."
    out = gdf_line.copy()
    out["geometry"] = out.buffer(meters)
    return out
