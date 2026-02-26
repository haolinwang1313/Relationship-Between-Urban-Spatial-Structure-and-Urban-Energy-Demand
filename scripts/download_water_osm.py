                      
"""Fetch OSM water polygons for Xinwu boundary and save to processed data."""

from pathlib import Path

import geopandas as gpd
import osmnx as ox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_PATH = PROJECT_ROOT / "Data" / "Processed" / "Boundary" / "xinwu_boundary_32650.geojson"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "Processed" / "Land_Use" / "xinwu_water_osm.gpkg"


def main() -> None:
    boundary = gpd.read_file(BOUNDARY_PATH)
    boundary = boundary.to_crs("EPSG:4326")
    polygon = boundary.geometry.unary_union
    tags = {
        "natural": ["water", "wetland"],
        "water": ["lake", "reservoir", "river", "pond"],
        "waterway": ["river", "riverbank", "canal", "stream"],
    }
    water = ox.features_from_polygon(polygon, tags=tags)
    water = water.to_crs("EPSG:32650")
    water = water.loc[water.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    boundary_utm = boundary.to_crs("EPSG:32650")
    water = gpd.clip(water, boundary_utm)
    water = water.reset_index(drop=True)
    water.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Saved water polygons to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
