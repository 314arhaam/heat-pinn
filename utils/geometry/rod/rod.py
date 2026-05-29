import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def generate_conrod_pinn_data(output_path: str, plot: bool) -> None:
    # --- Configuration ---
    center_dist = 6.0
    big_hole_r = 1.2
    big_end_outer_r = 1.8
    small_hole_r = 0.6
    small_end_outer_r = 1.0
    shank_width_big = 1.2
    shank_width_small = 0.8
    n_domain = 15000
    n_boundary = 3000

    # --- 1. Construct Geometry ---
    big_end_outer = Point(0, 0).buffer(big_end_outer_r)
    big_end_inner = Point(0, 0).buffer(big_hole_r)
    small_end_outer = Point(0, center_dist).buffer(small_end_outer_r)
    small_end_inner = Point(0, center_dist).buffer(small_hole_r)

    shank = Polygon(
        [
            [-shank_width_big / 2, 0.5],
            [shank_width_big / 2, 0.5],
            [shank_width_small / 2, center_dist - 0.5],
            [-shank_width_small / 2, center_dist - 0.5],
        ]
    )

    conrod_solid = unary_union([big_end_outer, small_end_outer, shank])
    conrod_final = conrod_solid.difference(big_end_inner).difference(small_end_inner)

    # --- 2. Boundary points with t-values ---
    boundary_points = []
    for i in np.linspace(0, conrod_solid.exterior.length, int(n_boundary * 0.6)):
        p = conrod_solid.exterior.interpolate(i)
        boundary_points.append([p.x, p.y, 0.0])

    for i in np.linspace(0, big_end_inner.exterior.length, int(n_boundary * 0.2)):
        p = big_end_inner.exterior.interpolate(i)
        boundary_points.append([p.x, p.y, 1.0])

    for i in np.linspace(0, small_end_inner.exterior.length, int(n_boundary * 0.2)):
        p = small_end_inner.exterior.interpolate(i)
        boundary_points.append([p.x, p.y, 0.75])

    df_boundary = pd.DataFrame(boundary_points, columns=["x", "y", "t"]).astype(np.float64)

    # --- 3. Domain sampling ---
    domain_points = []
    minx, miny, maxx, maxy = conrod_final.bounds
    while len(domain_points) < n_domain:
        pts = np.random.uniform([minx, miny], [maxx, maxy], size=(n_domain, 2))
        for pt in pts:
            if conrod_final.contains(Point(pt)):
                domain_points.append(pt)
            if len(domain_points) >= n_domain:
                break

    df_domain = pd.DataFrame(domain_points, columns=["x", "y"]).astype(np.float64)

    # --- 4. Save outputs ---
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = out_dir / "boundary_data.parquet"
    domain_path = out_dir / "domain_data.parquet"
    df_boundary.to_parquet(boundary_path, index=False)
    df_domain.to_parquet(domain_path, index=False)

    if plot:
        plt.figure(figsize=(6, 10))
        plt.scatter(df_domain["x"], df_domain["y"], s=1, c="lightgray", alpha=0.5, label="Domain")
        scatter = plt.scatter(df_boundary["x"], df_boundary["y"], c=df_boundary["t"], cmap="jet", s=5)
        plt.colorbar(scatter, label="Boundary Value (t)")
        plt.title("Realistic Connecting Rod PINN Geometry")
        plt.axis("equal")
        plot_path = out_dir / "geometry.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate connecting-rod geometry data.")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory path.")
    parser.add_argument("--plot", action="store_true", help="Save geometry plot.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_conrod_pinn_data(output_path=args.output_path, plot=args.plot)