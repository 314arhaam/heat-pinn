import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def generate_clock_gear_data(output_path: str, plot: bool) -> None:
    # --- Configuration ---
    num_teeth = 13
    outer_radius = 1.0
    inner_radius = 0.25
    tooth_depth = 0.15
    
    n_domain = 10000
    n_boundary_total = 1000 
    
    # --- 1. Clock Angle Calculation (Radians) ---
    # 03:30 -> -15 degrees
    # 07:30 -> -135 degrees
    alpha = np.radians(-15)
    beta = np.radians(-135)
    
    # Ensure beta < alpha for logical comparison if needed, 
    # but we will use a wrap-around check.
    # The "Clock Arc" is between -135 and -15.

    # --- 2. Construct Gear Geometry ---
    angles = np.linspace(0, 2 * np.pi, 500)
    # Gear profile
    r_coords = outer_radius + tooth_depth * np.sign(np.sin(num_teeth * angles))
    
    gear_points = np.column_stack([
        r_coords * np.cos(angles),
        r_coords * np.sin(angles)
    ])
    
    gear_poly = Polygon(gear_points)
    hole_poly = Point(0, 0).buffer(inner_radius)
    domain_poly = gear_poly.difference(hole_poly)

    # --- 3. Generate Boundary Points ---
    boundary_list = []

    # Outer Boundary Sampling
    outer_dist = np.linspace(0, gear_poly.exterior.length, n_boundary_total // 2)
    for d in outer_dist:
        p = gear_poly.exterior.interpolate(d)
        # Calculate angle of the point to determine t
        angle = np.arctan2(p.y, p.x)
        
        # Check if angle is within the [07:30, 03:30] arc (Bottom-Right quadrant mostly)
        # Angle range: -135 degrees to -15 degrees
        if beta <= angle <= alpha:
            t_val = 0.4
        else:
            t_val = 1.0
        boundary_list.append([p.x, p.y, t_val])

    # Inner Boundary Sampling (t = 0)
    inner_dist = np.linspace(0, hole_poly.exterior.length, n_boundary_total // 2)
    for d in inner_dist:
        p = hole_poly.exterior.interpolate(d)
        boundary_list.append([p.x, p.y, 0.0])

    df_boundary = pd.DataFrame(boundary_list, columns=['x', 'y', 't'])

    # --- 4. Generate Domain Points (Rejection Sampling) ---
    domain_points = []
    minx, miny, maxx, maxy = domain_poly.bounds
    while len(domain_points) < n_domain:
        pts = np.random.uniform([minx, miny], [maxx, maxy], size=(n_domain // 2, 2))
        for pt in pts:
            if domain_poly.contains(Point(pt)):
                domain_points.append(pt)
            if len(domain_points) >= n_domain: break

    df_domain = pd.DataFrame(domain_points, columns=['x', 'y'])

    # --- 5. Save and Export ---
    df_boundary = df_boundary.astype(np.float64)
    df_domain = df_domain.astype(np.float64)
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_boundary.to_parquet(out_dir / "boundary_data.parquet", index=False)
    df_domain.to_parquet(out_dir / "domain_data.parquet", index=False)

    # --- 6. Visualization ---
    if plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(df_domain['x'], df_domain['y'], s=0.5, c='gainsboro', alpha=0.5)
        scatter = plt.scatter(df_boundary['x'], df_boundary['y'], c=df_boundary['t'], cmap='jet', s=10, edgecolors='none')
        plt.colorbar(scatter, label='Boundary Value (t)')
        plt.title("Gear PINN: Outer Arc [03:30-07:30] t=0.4, Rest t=1.0, Inner t=0.0")
        plt.axis('equal')
        plt.savefig(out_dir / "geometry.png", dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate asymmetric gear geometry data.")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory path.")
    parser.add_argument("--plot", action="store_true", help="Save geometry plot.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_clock_gear_data(output_path=args.output_path, plot=args.plot)