                      
"""Recompute built-environment metrics (total floor area) on 250 m grid."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
BUILDING_PATH = PROCESSED_DIR / "Building" / "xinwu_buildings.gpkg"
GRID_PATH = PROCESSED_DIR / "Grids" / "xinwu_urban_form_250m.gpkg"
GRID_LAYER = "urban_form_250m"
LAND_USE_PATH = PROCESSED_DIR / "Land_Use" / "xinwu_land_use.gpkg"
WATER_PATH = PROCESSED_DIR / "Land_Use" / "xinwu_water_osm.gpkg"
BOUNDARY_PATH = PROCESSED_DIR / "Boundary" / "xinwu_boundary_32650.geojson"
OUTPUT_DIR = PROCESSED_DIR / "BuiltEnvironment"
OUTPUT_CSV = OUTPUT_DIR / "xinwu_built_env_250m.csv"
OUTPUT_GPKG = OUTPUT_DIR / "xinwu_built_env_250m.gpkg"

AVG_FLOOR_HEIGHT = 3.3     
PARK_CLASSES = [10]

BUILDING_TYPE_RULES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "single_family": lambda df: df["type"].round().eq(1),
    "multi_family": lambda df: df["type"].round().eq(2),
}

FACILITY_RULES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "facility_sales": lambda df: df["land_class"].isin([2]),
    "facility_office": lambda df: df["land_class"].isin([1]),
    "facility_neighborhood": lambda df: df["land_class"].isin([6, 8, 9]),
    "facility_education": lambda df: df["land_class"].isin([7]),
    "facility_industrial": lambda df: df["land_class"].isin([3]),
}


def load_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    grid = gpd.read_file(GRID_PATH, layer=GRID_LAYER)
    land_use = gpd.read_file(LAND_USE_PATH)
    boundary = gpd.read_file(BOUNDARY_PATH)
    water = gpd.read_file(WATER_PATH) if WATER_PATH.exists() else None
    if land_use.crs != grid.crs:
        land_use = land_use.to_crs(grid.crs)
    if boundary.crs != grid.crs:
        boundary = boundary.to_crs(grid.crs)
    if water is not None and water.crs != grid.crs:
        water = water.to_crs(grid.crs)
    return grid, land_use, boundary, water


def tag_buildings(land_use: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bld = gpd.read_file(BUILDING_PATH)
    tagged = bld.to_crs(land_use.crs)
    tagged["geometry"] = tagged.geometry.centroid
    tagged = gpd.sjoin(tagged, land_use[["Class", "geometry"]], how="left", predicate="within")
    tagged = tagged.rename(columns={"Class": "land_class"}).drop(columns=["index_right"])
    tagged["floor_area_m2"] = tagged["footprint_area_m2"] * np.maximum(
        tagged["height_m"].fillna(AVG_FLOOR_HEIGHT) / AVG_FLOOR_HEIGHT,
        1.0,
    )
    return tagged


def aggregate_floor_area(grid: gpd.GeoDataFrame, tagged: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(tagged, grid[["grid_id_main", "geometry"]], how="inner", predicate="within")
    joined = joined.drop(columns=["index_right"])
    summaries = [joined.groupby("grid_id_main")["floor_area_m2"].sum().rename("floor_area_total_m2")]
    for name, mask in BUILDING_TYPE_RULES.items():
        subset = joined[mask(joined)]
        grp = subset.groupby("grid_id_main")["floor_area_m2"].sum().rename(f"{name}_m2")
        summaries.append(grp)
    for name, mask in FACILITY_RULES.items():
        subset = joined[mask(joined)]
        grp = subset.groupby("grid_id_main")["floor_area_m2"].sum().rename(f"{name}_m2")
        summaries.append(grp)
    summary_df = pd.concat(summaries, axis=1).reset_index()
    return summary_df


def aggregate_land_cover(grid: gpd.GeoDataFrame, land_use: gpd.GeoDataFrame, classes: list[int]) -> pd.Series:
    subset = land_use[land_use["Class"].isin(classes)]
    if subset.empty:
        return pd.Series(dtype=float)
    overlay = gpd.overlay(grid[["grid_id_main", "geometry"]], subset[["geometry"]], how="intersection")
    overlay["area_m2"] = overlay.geometry.area
    return overlay.groupby("grid_id_main")["area_m2"].sum()


def water_area(grid: gpd.GeoDataFrame, water: Optional[gpd.GeoDataFrame], boundary: gpd.GeoDataFrame) -> pd.Series:
    if water is None or water.empty:
        return pd.Series(dtype=float)
    overlay = gpd.overlay(grid[["grid_id_main", "geometry"]], water[["geometry"]], how="intersection")
    overlay["area_m2"] = overlay.geometry.area
    return overlay.groupby("grid_id_main")["area_m2"].sum()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, land_use, boundary, water = load_layers()
    tagged = tag_buildings(land_use)
    summary = aggregate_floor_area(grid, tagged)
    merged = grid.merge(summary, on="grid_id_main", how="left").fillna(0.0)

    parks = aggregate_land_cover(grid, land_use, PARK_CLASSES)
    merged["parks_green_m2"] = merged["grid_id_main"].map(parks).fillna(0.0)
    water_series = water_area(grid, water, boundary)
    merged["water_area_m2"] = merged["grid_id_main"].map(water_series).fillna(0.0)

    for key in BUILDING_TYPE_RULES.keys():
        col = f"{key}_m2"
        if col in merged:
            merged[f"{key}_ha"] = merged[col] / 10000
        else:
            merged[f"{key}_ha"] = 0.0
    for key in FACILITY_RULES.keys():
        col = f"{key}_m2"
        if col in merged:
            merged[f"{key}_ha"] = merged[col] / 10000
        else:
            merged[f"{key}_ha"] = 0.0
    merged["parks_green_ha"] = merged["parks_green_m2"] / 10000
    merged["water_area_ha"] = merged["water_area_m2"] / 10000

    ha_cols = [col for col in merged.columns if col.endswith("_ha")]
    for col in ha_cols:
        merged[col] = merged[col].clip(lower=0)

    merged.to_csv(OUTPUT_CSV, index=False)
    merged.to_file(OUTPUT_GPKG, layer="built_env_250m", driver="GPKG")
    print(f"Saved built environment table (250 m) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
