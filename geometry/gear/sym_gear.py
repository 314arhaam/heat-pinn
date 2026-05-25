import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union

def generate_gear_pinn_data():
    # --- Configuration ---
    num_teeth = 13
    outer_radius = 1.0
    inner_radius = 0.25
    tooth_depth = 0.15
    
    n_domain = 10000
    n_boundary_total = 1000  # 500 inner, 500 outer
    
    # --- 1. Construct Gear Geometry ---
    angles = np.linspace(0, 2 * np.pi, 200)
    # Simple gear profile: oscillate radius based on tooth count
    # We use a square-ish wave approximation for teeth
    r_coords = outer_radius + tooth_depth * np.sign(np.sin(num_teeth * angles))
    
    gear_points = np.column_stack([
        r_coords * np.cos(angles),
        r_coords * np.sin(angles)
    ])
    
    gear_poly = Polygon(gear_points)
    hole_poly = Point(0, 0).buffer(inner_radius)
    
    # Final Domain: Gear minus the Hole
    domain_poly = gear_poly.difference(hole_poly)

    # --- 2. Generate Boundary Points ---
    # Outer boundary (t = 1)
    outer_points_raw = []
    distances = np.linspace(0, gear_poly.exterior.length, n_boundary_total // 2)
    for d in distances:
        p = gear_poly.exterior.interpolate(d)
        outer_points_raw.append([p.x, p.y, 1.0])
        
    # Inner boundary (t = 0)
    inner_points_raw = []
    distances = np.linspace(0, hole_poly.exterior.length, n_boundary_total // 2)
    for d in distances:
        p = hole_poly.exterior.interpolate(d)
        inner_points_raw.append([p.x, p.y, 0.0])
        
    df_boundary = pd.DataFrame(
        np.vstack([outer_points_raw, inner_points_raw]), 
        columns=['x', 'y', 't']
    )

    # --- 3. Generate Domain Points (Rejection Sampling) ---
    domain_points = []
    minx, miny, maxx, maxy = domain_poly.bounds
    
    while len(domain_points) < n_domain:
        # Batch generation for speed
        batch_size = (n_domain - len(domain_points)) * 2
        pts = np.random.uniform([minx, miny], [maxx, maxy], size=(batch_size, 2))
        
        for pt in pts:
            p = Point(pt)
            if domain_poly.contains(p):
                domain_points.append(pt)
            if len(domain_points) >= n_domain:
                break
                
    df_domain = pd.DataFrame(domain_points, columns=['x', 'y'])

    # --- 4. Save to Parquet ---
    # Using float64 as requested
    df_boundary = df_boundary.astype(np.float64)
    df_domain = df_domain.astype(np.float64)
    
    df_boundary.to_parquet("boundary_data.parquet")
    df_domain.to_parquet("domain_data.parquet")
    print("Files saved: boundary_data.parquet, domain_data.parquet")

    # --- 5. Plotting ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df_domain['x'], df_domain['y'], s=1, c='lightgray', label='Domain (10k)')
    plt.scatter(df_boundary[df_boundary['t']==1]['x'], 
                df_boundary[df_boundary['t']==1]['y'], s=5, c='red', label='Outer (t=1)')
    plt.scatter(df_boundary[df_boundary['t']==0]['x'], 
                df_boundary[df_boundary['t']==0]['y'], s=5, c='blue', label='Inner (t=0)')
    
    plt.axis('equal')
    plt.title(f"{num_teeth}-Teeth Gear PINN Geometry")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_gear_pinn_data()