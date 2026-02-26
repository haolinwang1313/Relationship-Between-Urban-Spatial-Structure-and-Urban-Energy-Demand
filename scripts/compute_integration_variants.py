                      
"""
Compute alternative GI/LI variants using expanded road networks.

Usage examples
--------------
source .venv_geo/bin/activate
python scripts/compute_integration_variants.py \
    --network Data/Processed/Road/xinwu_buffer15km_road_centerlines.gpkg \
    --tag buffer15km
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
BOUNDARY_PATH = PROCESSED_DIR / "Boundary" / "xinwu_boundary_32650.geojson"
GRID_PATH = PROCESSED_DIR / "Grids" / "xinwu_grid_250m.gpkg"
GRID_LAYER = "grid_250m"
GRID_ID_FIELD = "grid_id_main"
GRID_OUTPUT_PATH = PROCESSED_DIR / "Grids" / "xinwu_urban_form_250m.gpkg"
GRID_OUTPUT_LAYER = "urban_form_250m"
CSV_PATH = PROCESSED_DIR / "Indicators" / "xinwu_urban_form_250m.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=Path,
        required=True,
        help="Path to the road GeoPackage containing 'road_centerlines' layer.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Suffix tag to append to output columns/files (e.g., buffer15km).",
    )
    parser.add_argument(
        "--local-radius",
        type=float,
        default=500.0,
        help="Radius (meters) for local integration (default: 500 m).",
    )
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.05,
        help="Lower quantile for normalisation (default 0.05).",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.95,
        help="Upper quantile for normalisation (default 0.95).",
    )
    parser.add_argument(
        "--target-boundary",
        type=Path,
        default=BOUNDARY_PATH,
        help="Boundary used to select nodes/grid cells for aggregation (default: Xinwu).",
    )
    parser.add_argument(
        "--target-buffer",
        type=float,
        default=1000.0,
        help="Buffer (meters) applied to target boundary when selecting nodes (default 1000).",
    )
    return parser.parse_args()


def normalise_series(
    series: pd.Series, lower_quantile: float = 0.05, upper_quantile: float = 0.95
) -> pd.Series:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) == 0:
        return pd.Series(np.zeros_like(series), index=series.index)
    lo = np.nanquantile(valid, lower_quantile)
    hi = np.nanquantile(valid, upper_quantile)
    if hi - lo == 0:
        hi = lo + 1e-9
    scaled = (series - lo) / (hi - lo)
    return scaled.clip(0.0, 1.0)


def compute_closeness(
    G: nx.Graph,
    nodes_subset: Iterable[object],
    weight: str,
    local_radius: float,
) -> Tuple[Dict[object, float], Dict[object, float]]:
    if not nodes_subset:
        return {}, {}
    nodelist = list(G.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodelist)}
    target_idx = [int(node_to_idx[node]) for node in nodes_subset]
    matrix = nx.to_scipy_sparse_array(
        G, nodelist=nodelist, weight=weight, dtype=np.float64
    ).tocsr()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)

    print("Running global Dijkstra via SciPy ...")
    dist_global = sparse_dijkstra(matrix, directed=False, indices=target_idx)
    print("Running local Dijkstra via SciPy ...")
    dist_local = sparse_dijkstra(
        matrix,
        directed=False,
        indices=target_idx,
        limit=local_radius if local_radius is not None else np.inf,
    )

    global_cc: Dict[object, float] = {}
    local_cc: Dict[object, float] = {}
    for node, idx_row in zip(nodes_subset, range(len(target_idx))):
        idx_self = node_to_idx[node]

                          
        row = dist_global[idx_row]
        finite = np.isfinite(row)
        finite[idx_self] = False
        reachable = finite.sum()
        if reachable <= 0:
            global_cc[node] = 0.0
        else:
            total = row[finite].sum()
            global_cc[node] = (reachable) / total if total > 0 else 0.0

                         
        row_local = dist_local[idx_row]
        finite_local = np.isfinite(row_local)
        finite_local[idx_self] = False
        reachable_local = finite_local.sum()
        if reachable_local <= 0:
            local_cc[node] = 0.0
        else:
            total_local = row_local[finite_local].sum()
            local_cc[node] = (reachable_local) / total_local if total_local > 0 else 0.0

    return global_cc, local_cc


def build_graph(network_path: Path) -> Tuple[nx.Graph, gpd.GeoDataFrame]:
    roads = gpd.read_file(network_path, layer="road_centerlines")
    roads = roads[roads.geometry.is_valid & (~roads.geometry.is_empty)].copy()
    roads = roads.explode(ignore_index=True)
    roads = roads[roads.geometry.type == "LineString"].copy()
    if "length_m" not in roads.columns:
        roads["length_m"] = roads.geometry.length
    graph = momepy.gdf_to_nx(roads, approach="primal", length="length_m")
    return graph, roads


def aggregate_to_grid(
    nodes: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    joined = gpd.sjoin(
        nodes,
        grid[[GRID_ID_FIELD, "geometry"]],
        how="left",
        predicate="within",
    )
    grouped = (
        joined.dropna(subset=[GRID_ID_FIELD])
        .groupby(GRID_ID_FIELD)
        .agg(
            gi_mean=("global_integration", "mean"),
            li_mean=("local_integration", "mean"),
        )
        .reset_index()
    )
    grouped["gi_norm"] = normalise_series(
        grouped["gi_mean"], lower_quantile=lower_q, upper_quantile=upper_q
    )
    grouped["li_norm"] = normalise_series(
        grouped["li_mean"], lower_quantile=lower_q, upper_quantile=upper_q
    )
    return grouped


def compute_integration(
    network_path: Path,
    tag: str,
    local_radius: float,
    lower_q: float,
    upper_q: float,
    target_boundary_path: Path,
    target_buffer_m: float,
) -> None:
    boundary = gpd.read_file(BOUNDARY_PATH)
    grid = gpd.read_file(GRID_PATH, layer=GRID_LAYER)[[GRID_ID_FIELD, "geometry"]]
    target_boundary = gpd.read_file(target_boundary_path)
    target_boundary = target_boundary.to_crs(grid.crs)
    if target_buffer_m > 0:
        target_geom = target_boundary.geometry.buffer(target_buffer_m).union_all()
    else:
        target_geom = target_boundary.union_all()

    print(f"Building graph from {network_path} ...")
    graph, roads = build_graph(network_path)
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

                                                                       
    target_nodes = [
        node for node in graph.nodes if Point(node).within(target_geom)
    ]
    print(f"Nodes within target boundary: {len(target_nodes)}")

    global_cc, local_cc = compute_closeness(
        graph,
        target_nodes,
        weight="length_m",
        local_radius=local_radius,
    )

    node_records = [
        {
            "geometry": Point(node_id),
            "global_integration": global_cc.get(node_id, 0.0),
            "local_integration": local_cc.get(node_id, 0.0),
        }
        for node_id in graph.nodes
    ]
    nodes = gpd.GeoDataFrame(node_records, geometry="geometry", crs=roads.crs)

    edge_records = []
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([Point(u), Point(v)])
        edge_records.append(
            {
                "geometry": geom,
                "gi_line": (global_cc.get(u, 0.0) + global_cc.get(v, 0.0)) / 2,
                "li_line": (local_cc.get(u, 0.0) + local_cc.get(v, 0.0)) / 2,
            }
        )
    edges = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=roads.crs)
    edges = gpd.clip(edges, boundary)
    edges[f"gi_norm_line_{tag}"] = normalise_series(
        edges["gi_line"], lower_quantile=lower_q, upper_quantile=upper_q
    )
    edges[f"li_norm_line_{tag}"] = normalise_series(
        edges["li_line"], lower_quantile=lower_q, upper_quantile=upper_q
    )

    variant_metrics = aggregate_to_grid(nodes, grid, lower_q, upper_q)
    rename_map = {
        "gi_mean": f"gi_mean_{tag}",
        "li_mean": f"li_mean_{tag}",
        "gi_norm": f"gi_norm_{tag}",
        "li_norm": f"li_norm_{tag}",
    }
    variant_metrics = variant_metrics.rename(columns=rename_map)

                   
    out_line = PROCESSED_DIR / "Road" / f"xinwu_road_integration_{tag}.gpkg"
    if out_line.exists():
        out_line.unlink()
    edges.to_file(out_line, layer="integration_lines", driver="GPKG")
    print(f"Saved integration lines to {out_line}")

                           
    csv_df = pd.read_csv(CSV_PATH)
    for col in rename_map.values():
        if col in csv_df.columns:
            csv_df = csv_df.drop(columns=[col])
    csv_df = csv_df.merge(variant_metrics, on=GRID_ID_FIELD, how="left")
    for col in rename_map.values():
        csv_df[col] = csv_df[col].fillna(0.0)
    csv_df.to_csv(CSV_PATH, index=False)
    print(f"Updated CSV indicators at {CSV_PATH}")

                             
    grid_gdf = gpd.read_file(GRID_OUTPUT_PATH, layer=GRID_OUTPUT_LAYER)
    for col in rename_map.values():
        if col in grid_gdf.columns:
            grid_gdf = grid_gdf.drop(columns=[col])
    grid_gdf = grid_gdf.merge(variant_metrics, on=GRID_ID_FIELD, how="left")
    for col in rename_map.values():
        grid_gdf[col] = grid_gdf[col].fillna(0.0)
    if GRID_OUTPUT_PATH.exists():
        GRID_OUTPUT_PATH.unlink()
    grid_gdf.to_file(GRID_OUTPUT_PATH, layer=GRID_OUTPUT_LAYER, driver="GPKG")
    print(f"Updated GeoPackage indicators at {GRID_OUTPUT_PATH}")


def main() -> None:
    args = parse_args()
    compute_integration(
        network_path=args.network,
        tag=args.tag,
        local_radius=args.local_radius,
        lower_q=args.lower_quantile,
        upper_q=args.upper_quantile,
        target_boundary_path=args.target_boundary,
        target_buffer_m=args.target_buffer,
    )


if __name__ == "__main__":
    main()
