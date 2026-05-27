import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

def generate_conrod_pinn_data():
    # --- Configuration ---
    center_dist = 6.0      # Distance between big and small hole centers
    big_hole_r = 1.2       # Inner radius of big hole
    big_end_outer_r = 1.8  # Outer radius of big end
    
    small_hole_r = 0.6     # Inner radius of small hole
    small_end_outer_r = 1.0 # Outer radius of small end
    
    shank_width_big = 1.2   # Shank width near big end
    shank_width_small = 0.8 # Shank width near small end
    
    n_domain = 15000
    n_boundary = 3000

    # --- 1. Construct Geometry ---
    # Circles for the ends
    big_end_outer = Point(0, 0).buffer(big_end_outer_r)
    big_end_inner = Point(0, 0).buffer(big_hole_r)
    
    small_end_outer = Point(0, center_dist).buffer(small_end_outer_r)
    small_end_inner = Point(0, center_dist).buffer(small_hole_r)
    
    # Shank (The I-beam bridge) - Trapezoid connecting the two ends
    shank_coords = [
        [-shank_width_big/2, 0.5],
        [shank_width_big/2, 0.5],
        [shank_width_small/2, center_dist - 0.5],
        [-shank_width_small/2, center_dist - 0.5]
    ]
    shank = Polygon(shank_coords)
    
    # Combine outer shapes and subtract holes
    conrod_solid = unary_union([big_end_outer, small_end_outer, shank])
    conrod_final = conrod_solid.difference(big_end_inner).difference(small_end_inner)

    # --- 2. Sample Boundaries & Assign 't' ---
    boundary_points = []
    
    # We sample from each individual boundary ring to ensure clean 't' assignment
    # Outer perimeter (t = 0)
    outer_coords = conrod_solid.exterior.coords
    for i in np.linspace(0, conrod_solid.exterior.length, int(n_boundary * 0.6)):
        p = conrod_solid.exterior.interpolate(i)
        boundary_points.append([p.x, p.y, 0.0])
        
    # Big hole (t = 1.0)
    for i in np.linspace(0, big_end_inner.exterior.length, int(n_boundary * 0.2)):
        p = big_end_inner.exterior.interpolate(i)
        boundary_points.append([p.x, p.y, 1.0])
        
    # Small hole (t = 0.75)
    for i in np.linspace(0, small_end_inner.exterior.length, int(n_boundary * 0.2)):
        p = small_end_inner.exterior.interpolate(i)
        boundary_points.append([p.x, p.y, 0.75])

    df_boundary = pd.DataFrame(boundary_points, columns=['x', 'y', 't'])

    # --- 3. Sample Domain (Internal) ---
    domain_points = []
    minx, miny, maxx, maxy = conrod_final.bounds
    while len(domain_points) < n_domain:
        pts = np.random.uniform([minx, miny], [maxx, maxy], size=(n_domain, 2))
        for pt in pts:
            p_obj = Point(pt)
            if conrod_final.contains(p_obj):
                domain_points.append(pt)
            if len(domain_points) >= n_domain: break

    df_domain = pd.DataFrame(domain_points, columns=['x', 'y'])

    # --- 4. Enforce float64 and Save ---
    df_boundary = df_boundary.astype(np.float64)
    df_domain = df_domain.astype(np.float64)
    
    df_boundary.to_parquet("conrod_boundary.parquet")
    df_domain.to_parquet("conrod_domain.parquet")

    # --- 5. Visualization ---
    plt.figure(figsize=(6, 10))
    plt.scatter(df_domain['x'], df_domain['y'], s=1, c='lightgray', alpha=0.5, label='Domain')
    scatter = plt.scatter(df_boundary['x'], df_boundary['y'], 
                         c=df_boundary['t'], cmap='jet', s=5)
    
    plt.colorbar(scatter, label='Boundary Value (t)')
    plt.title("Realistic Connecting Rod PINN Geometry")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    generate_conrod_pinn_data()