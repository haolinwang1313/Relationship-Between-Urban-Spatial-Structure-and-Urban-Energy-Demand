                      
"""
Preprocess spatial datasets for Xinwu District, Wuxi.

Steps performed:
1. Fetch (or reuse) the Xinwu administrative boundary from OpenStreetMap.
2. Clip and reproject building footprints, land-use polygons,
   road centerlines, and bus network data to the Xinwu boundary.
3. Save processed layers to ``Data/Processed`` and emit a quick summary JSON.

The script expects the raw datasets to be placed under ``Data/`` as provided
by the user:
    - Boundary/CHN_ADM2
    - Building/China_3/jiangsu.shp (province-wide building footprints)
    - Land_Use/EULUC_China_20.gpkg (national land-use polygons)
    - load_centerline/{wuxi,xinwu}.geojson (OpenStreetMap exports)
    - Bus/Comprehensive_Vector_2024/.../Wuxi City/*.shp (city-wide bus network)

Run inside the project root with the dedicated geo virtualenv activated:
```
source .venv_geo/bin/activate
python scripts/preprocess_xinwu.py
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = RAW_DIR / "Processed"
TARGET_CRS = "EPSG:32650"                               
BOUNDARY_PLACE = "Xinwu District, Wuxi, Jiangsu, China"
BOUNDARY_RAW_PATH = PROCESSED_DIR / "Boundary" / "xinwu_boundary_wgs84.geojson"
BOUNDARY_TARGET_PATH = PROCESSED_DIR / "Boundary" / "xinwu_boundary_32650.geojson"
SUMMARY_PATH = PROCESSED_DIR / "summary_xinwu.json"


def ensure_directories() -> None:
    """Create processed sub-directories if missing."""
    for sub in ("Boundary", "Building", "Land_Use", "Road", "Bus"):
        (PROCESSED_DIR / sub).mkdir(parents=True, exist_ok=True)


def fetch_boundary() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fetch Xinwu District boundary from OSM or reuse cached GeoJSON.

    Returns
    -------
    tuple
        - boundary_wgs84: GeoDataFrame with EPSG:4326 geometry
        - clip_geom: GeoDataFrame representing the exact district polygon
        - search_geom: GeoDataFrame buffered by 2 km for bbox queries
    """
    if BOUNDARY_RAW_PATH.exists():
        boundary_wgs84 = gpd.read_file(BOUNDARY_RAW_PATH)
    else:
        ox.settings.nominatim_timeout = 45
        boundary_wgs84 = ox.geocode_to_gdf(BOUNDARY_PLACE)[["geometry"]]
        boundary_wgs84 = boundary_wgs84.to_crs("EPSG:4326")
        boundary_wgs84.to_file(BOUNDARY_RAW_PATH, driver="GeoJSON")

    boundary_utm = boundary_wgs84.to_crs(TARGET_CRS)
                                                         
    search_geom = boundary_utm.buffer(2000).to_frame(name="geometry").set_crs(TARGET_CRS)
    search_geom = search_geom.to_crs("EPSG:4326")
    boundary_utm.to_file(BOUNDARY_TARGET_PATH, driver="GeoJSON")

    clip_geom = boundary_wgs84[["geometry"]].copy()
    return boundary_wgs84, clip_geom, search_geom


def _bbox_from_geom(geom_gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return total bounds tuple for convenience."""
    minx, miny, maxx, maxy = geom_gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def process_buildings(boundary: gpd.GeoDataFrame, search_geom: gpd.GeoDataFrame) -> Dict:
    """Clip Jiangsu building footprints to Xinwu and compute footprint/volume metrics."""
    source_path = RAW_DIR / "Building" / "China_3" / "jiangsu.shp"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing building file: {source_path}")

    bbox = _bbox_from_geom(search_geom)
    buildings = gpd.read_file(source_path, bbox=bbox)
    buildings = gpd.clip(buildings, search_geom)
    buildings = buildings.loc[~buildings.geometry.is_empty].copy()

                                                                         
    buildings = gpd.clip(buildings, boundary)
    buildings = buildings.to_crs(TARGET_CRS)
    buildings["footprint_area_m2"] = buildings.geometry.area
    buildings["height_m"] = pd.to_numeric(buildings["Height"], errors="coerce")
    buildings["approx_volume_m3"] = buildings["footprint_area_m2"] * buildings["height_m"]

    out_path = PROCESSED_DIR / "Building" / "xinwu_buildings.gpkg"
    buildings.to_file(out_path, layer="buildings", driver="GPKG")

    summary = {
        "raw_features": int(len(buildings)),
        "total_footprint_area_m2": float(buildings["footprint_area_m2"].sum()),
        "valid_height_share": float(buildings["height_m"].notna().mean()),
        "type_counts": buildings["type"].fillna(-1).astype(int).value_counts().to_dict(),
    }
    return summary


def _normalise_line_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert polygonal features to their boundaries so mixed road geometries
    become usable line work for network analysis.
    """
    def to_line(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            return geom.boundary
        return geom

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(to_line)
    gdf = gdf.dropna(subset=["geometry"])
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf


def process_roads(boundary: gpd.GeoDataFrame, search_geom: gpd.GeoDataFrame) -> Dict:
    """Clip and clean Xinwu road centerlines."""
    xinwu_path = RAW_DIR / "load_centerline" / "xinwu.geojson"
    wuxi_path = RAW_DIR / "load_centerline" / "wuxi.geojson"
    source_path = xinwu_path if xinwu_path.exists() else wuxi_path
    if not source_path.exists():
        raise FileNotFoundError("Road centerline source not found (expected xinwu or wuxi export).")

    roads = gpd.read_file(source_path)
    if source_path == wuxi_path:
        roads = gpd.clip(roads, search_geom)
    roads = gpd.clip(roads, boundary)
    roads_line = _normalise_line_geometries(roads)
    roads_line = roads_line.to_crs(TARGET_CRS)
    roads_line["length_m"] = roads_line.geometry.length

    out_raw = PROCESSED_DIR / "Road" / "xinwu_roads_raw.geojson"
    out_line = PROCESSED_DIR / "Road" / "xinwu_road_centerlines.gpkg"
    roads.to_file(out_raw, driver="GeoJSON")
    roads_line.to_file(out_line, layer="road_centerlines", driver="GPKG")

    summary = {
        "raw_features": int(len(roads)),
        "line_features": int(len(roads_line)),
        "total_length_m": float(roads_line["length_m"].sum()),
    }
    return summary


def process_land_use(boundary: gpd.GeoDataFrame, search_geom: gpd.GeoDataFrame) -> Dict:
    """Clip national land-use data to Xinwu."""
    source_path = RAW_DIR / "Land_Use" / "EULUC_China_20.gpkg"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing land-use file: {source_path}")

    bbox = _bbox_from_geom(search_geom)
    land_use = gpd.read_file(source_path, bbox=bbox)
    land_use = gpd.clip(land_use, boundary)
    land_use = land_use.to_crs(TARGET_CRS)
    land_use["area_m2"] = land_use.geometry.area

    out_path = PROCESSED_DIR / "Land_Use" / "xinwu_land_use.gpkg"
    land_use.to_file(out_path, layer="land_use", driver="GPKG")

    class_counts = land_use["Class"].value_counts().to_dict()
    summary = {
        "features": int(len(land_use)),
        "total_area_m2": float(land_use["area_m2"].sum()),
        "class_counts": class_counts,
    }
    return summary


def process_bus(boundary: gpd.GeoDataFrame) -> Dict:
    """Clip Wuxi bus network datasets to Xinwu."""
    base = RAW_DIR / "Bus" / "Comprehensive_Vector_2024" / "dataset"
    stop_path = base / "simplified net file" / "stop" / "Wuxi City" / "Wuxi City.shp"
    directed_path = base / "simplified net file" / "directed route" / "Wuxi City" / "Wuxi City.shp"
    undirected_path = base / "simplified net file" / "undirected route" / "Wuxi City" / "Wuxi City.shp"
    edge_path = base / "topological net file" / "edge" / "Wuxi City" / "Wuxi City.shp"
    point_path = base / "topological net file" / "point" / "Wuxi City" / "Wuxi City.shp"

    required = [stop_path, directed_path, undirected_path, edge_path, point_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing bus network files: {missing}")

    def _clip_layer(path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        gdf = gpd.clip(gdf, boundary)
        return gdf.to_crs(TARGET_CRS)

    stops = _clip_layer(stop_path)
    directed = _clip_layer(directed_path)
    undirected = _clip_layer(undirected_path)
    edges = _clip_layer(edge_path)
    points = _clip_layer(point_path)

    for gdf in (directed, undirected, edges):
        gdf["length_m"] = gdf.geometry.length

    out_path = PROCESSED_DIR / "Bus" / "xinwu_bus_network.gpkg"
    if out_path.exists():
        out_path.unlink()
    stops.to_file(out_path, layer="stops", driver="GPKG")
    directed.to_file(out_path, layer="directed_routes", driver="GPKG", mode="a")
    undirected.to_file(out_path, layer="undirected_routes", driver="GPKG", mode="a")
    edges.to_file(out_path, layer="topo_edges", driver="GPKG", mode="a")
    points.to_file(out_path, layer="topo_nodes", driver="GPKG", mode="a")

    summary = {
        "stops": int(len(stops)),
        "directed_routes": int(len(directed)),
        "undirected_routes": int(len(undirected)),
        "topology_edges": int(len(edges)),
        "topology_nodes": int(len(points)),
        "directed_total_length_m": float(directed["length_m"].sum()),
    }
    return summary


def main() -> None:
    ensure_directories()
    boundary_wgs84, clip_geom, search_geom = fetch_boundary()

    summaries: Dict[str, Dict] = {}
    summaries["boundary_bbox"] = dict(zip(
        ["min_lon", "min_lat", "max_lon", "max_lat"],
        _bbox_from_geom(boundary_wgs84),
    ))

    summaries["building"] = process_buildings(clip_geom, search_geom)
    summaries["road"] = process_roads(clip_geom, search_geom)
    summaries["land_use"] = process_land_use(clip_geom, search_geom)
    summaries["bus"] = process_bus(clip_geom)

    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)

    print("Processing complete. Summary written to", SUMMARY_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                                                  
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
