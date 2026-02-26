                      
"""Compute transportation accessibility indicators on the 250 m grid."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
TRANSPORT_DIR = PROCESSED_DIR / "Transportatio"
GRID_PATH = PROCESSED_DIR / "Grids" / "xinwu_grid_250m.gpkg"
GRID_LAYER = "grid_250m"
ROAD_PATH = PROCESSED_DIR / "Road" / "xinwu_road_centerlines.gpkg"
ROAD_LAYER = "road_centerlines"
BUS_PATH = PROCESSED_DIR / "Bus" / "xinwu_bus_network.gpkg"
BUS_STOP_LAYER = "stops"
BOUNDARY_PATH = PROCESSED_DIR / "Boundary" / "xinwu_boundary_32650.geojson"

ROAD_HALF_WIDTH_M = 8.0                                                         
SUBWAY_BUFFER_RADIUS_M = 250.0

OUTPUT_GPKG = TRANSPORT_DIR / "xinwu_transportatio_250m.gpkg"
OUTPUT_LAYER = "transportatio_250m"
OUTPUT_CSV = TRANSPORT_DIR / "xinwu_transportatio_250m.csv"
SUBWAY_CACHE = TRANSPORT_DIR / "xinwu_subway_stations.gpkg"
SUBWAY_CACHE_LAYER = "subway_stations"


def ensure_directories() -> None:
    TRANSPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_grid() -> gpd.GeoDataFrame:
    grid = gpd.read_file(GRID_PATH, layer=GRID_LAYER)
    if grid.crs is None:
        raise ValueError("Grid dataset lacks CRS definition")
    return grid


def load_boundary(to_crs: Optional[str]) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(BOUNDARY_PATH)
    if to_crs and boundary.crs != to_crs:
        boundary = boundary.to_crs(to_crs)
    return boundary


def compute_road_area(grid: gpd.GeoDataFrame) -> pd.Series:
    roads = gpd.read_file(ROAD_PATH, layer=ROAD_LAYER)
    if roads.crs != grid.crs:
        roads = roads.to_crs(grid.crs)
    roads = roads[~roads.geometry.is_empty].copy()
    if roads.empty:
        return pd.Series(dtype=float)
    buffered = roads.buffer(ROAD_HALF_WIDTH_M, cap_style=2)
    buffered = gpd.GeoSeries(buffered, crs=grid.crs)
    buffered = buffered[~buffered.is_empty].buffer(0)
    if buffered.empty:
        return pd.Series(dtype=float)
    union_geom = buffered.unary_union
    if union_geom.is_empty:
        return pd.Series(dtype=float)
    road_poly = gpd.GeoDataFrame(geometry=[union_geom], crs=grid.crs)
    overlay = gpd.overlay(grid[["grid_id_main", "geometry"]], road_poly, how="intersection")
    if overlay.empty:
        return pd.Series(dtype=float)
    overlay["area_m2"] = overlay.geometry.area
    return overlay.groupby("grid_id_main")["area_m2"].sum()


def fetch_subway_from_osm(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    place = "Xinwu District, Wuxi, China"
    tags = {"railway": "station"}
    ox.settings.use_cache = True
    ox.settings.log_console = False
    raw = ox.features_from_place(place, tags)
    raw = raw[~raw.geometry.is_empty].copy()
    raw = raw.to_crs(boundary.crs)
    raw["geometry"] = raw.geometry.centroid
    mask = (
        (raw.get("station") == "subway")
        | (raw.get("public_transport") == "subway")
        | (raw.get("public_transport") == "station")
        | (raw.get("subway") == "yes")
    )
    filtered = raw[mask].copy()
    if filtered.empty:
        return gpd.GeoDataFrame(columns=["name", "geometry"], geometry="geometry", crs=boundary.crs)
    filtered = gpd.clip(filtered, boundary)
    filtered = filtered.dropna(subset=["geometry"])
    filtered = filtered.drop_duplicates(subset=["name"])
    keep_cols = [col for col in ("name", "geometry") if col in filtered.columns]
    return filtered[keep_cols].reset_index(drop=True)


def load_or_fetch_subway(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if SUBWAY_CACHE.exists():
        stations = gpd.read_file(SUBWAY_CACHE, layer=SUBWAY_CACHE_LAYER)
        if stations.crs != boundary.crs:
            stations = stations.to_crs(boundary.crs)
    else:
        stations = fetch_subway_from_osm(boundary)
        if not stations.empty:
            stations.to_file(SUBWAY_CACHE, layer=SUBWAY_CACHE_LAYER, driver="GPKG")
    return stations


def compute_subway_influence(
    grid: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> tuple[pd.Series, gpd.GeoDataFrame]:
    stations = load_or_fetch_subway(boundary)
    if stations.empty:
        empty_series = pd.Series(dtype=float)
        return empty_series, stations
    buffers = stations.copy()
    buffers["geometry"] = buffers.geometry.buffer(SUBWAY_BUFFER_RADIUS_M)
    buffers = buffers[~buffers.geometry.is_empty]
    if buffers.empty:
        empty_series = pd.Series(dtype=float)
        return empty_series, stations
    union_geom = buffers.unary_union
    if isinstance(union_geom, MultiPolygon):
        buffer_gdf = gpd.GeoDataFrame(geometry=list(union_geom.geoms), crs=grid.crs)
    else:
        buffer_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs=grid.crs)
    overlay = gpd.overlay(grid[["grid_id_main", "geometry"]], buffer_gdf, how="intersection")
    if overlay.empty:
        return pd.Series(dtype=float), stations
    overlay["area_m2"] = overlay.geometry.area
    return overlay.groupby("grid_id_main")["area_m2"].sum(), stations


def compute_bus_routes(grid: gpd.GeoDataFrame) -> pd.Series:
    stops = gpd.read_file(BUS_PATH, layer=BUS_STOP_LAYER)
    if stops.crs != grid.crs:
        stops = stops.to_crs(grid.crs)
    stops = stops.dropna(subset=["geometry", "lineName"]).copy()
    if stops.empty:
        return pd.Series(dtype=float)
    joined = gpd.sjoin(stops[["lineName", "geometry"]], grid[["grid_id_main", "geometry"]], how="inner", predicate="within")
    if joined.empty:
        return pd.Series(dtype=float)
    grouped = joined.groupby("grid_id_main")["lineName"].nunique()
    return grouped


def main() -> None:
    ensure_directories()
    grid = load_grid()
    boundary = load_boundary(grid.crs)

    road_area = compute_road_area(grid)
    subway_area, stations = compute_subway_influence(grid, boundary)
    bus_routes = compute_bus_routes(grid)

    df = pd.DataFrame({"grid_id_main": grid["grid_id_main"]})
    df = df.merge(road_area.rename("road_area_m2"), on="grid_id_main", how="left")
    df = df.merge(subway_area.rename("subway_influence_m2"), on="grid_id_main", how="left")
    df = df.merge(bus_routes.rename("bus_routes_cnt"), on="grid_id_main", how="left")
    df = df.fillna({"road_area_m2": 0.0, "subway_influence_m2": 0.0, "bus_routes_cnt": 0})
    df["road_area_ha"] = df["road_area_m2"] / 10_000.0
    df["subway_influence_ha"] = df["subway_influence_m2"] / 10_000.0
    df["bus_routes_cnt"] = df["bus_routes_cnt"].round().astype(int)

    merged = grid.merge(df, on="grid_id_main", how="left")
    merged[["road_area_m2", "subway_influence_m2", "road_area_ha", "subway_influence_ha"]] = merged[
        ["road_area_m2", "subway_influence_m2", "road_area_ha", "subway_influence_ha"]
    ].clip(lower=0)

    merged.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver="GPKG")
    merged.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print(f"Saved grid metrics to {OUTPUT_GPKG}")
    print(f"Saved CSV summary to {OUTPUT_CSV}")
    if not stations.empty:
        print(f"Subway stations cached at {SUBWAY_CACHE}")


if __name__ == "__main__":
    main()
