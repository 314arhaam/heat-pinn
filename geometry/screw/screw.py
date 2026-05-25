import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

def generate_realistic_screw():
    # --- Dimensions ---
    head_radius = 0.6
    head_height = 0.3
    shank_radius = 0.25
    shank_length = 2.0
    tip_length = 0.5
    
    thread_pitch = 0.15
    thread_depth = 0.08
    
    n_domain = 12000
    n_boundary = 2500

    # --- 1. Geometry Construction ---
    # Create the Head (Rounded Top)
    theta = np.linspace(0, np.pi, 50)
    head_top_x = head_radius * np.cos(theta)
    head_top_y = head_height * np.sin(theta)
    
    # Create the Shank with Threads
    y_coords = np.arange(0, -shank_length, -0.01)
    right_side = []
    left_side = []
    
    for y in y_coords:
        # Oscillating thread profile using a triangle wave approximation
        # We shift the phase on one side to simulate the helical inclination
        thread_offset = (y % thread_pitch) / thread_pitch
        wave = thread_depth * (1 - 2 * abs(thread_offset - 0.5))
        
        right_side.append([shank_radius + wave, y - head_height])
        left_side.append([-shank_radius - wave, y - head_height])

    # Create the Tapered Tip
    tip_y = np.linspace(-shank_length - head_height, -shank_length - head_height - tip_length, 20)
    right_tip = []
    left_tip = []
    for i, y in enumerate(tip_y):
        shrink_factor = 1 - (i / len(tip_y))
        right_tip.append([shank_radius * shrink_factor, y])
        left_tip.append([-shank_radius * shrink_factor, y])

    # Combine all points into a single clockwise path
    # 1. Head top (Left to Right)
    # 2. Head right side
    # 3. Shank right (Threads)
    # 4. Tip
    # 5. Shank left (Threads)
    # 6. Head left side
    
    full_profile = (
        np.column_stack([head_top_x[::-1], head_top_y]).tolist() +
        [[head_radius, -head_height/2], [shank_radius + thread_depth, -head_height]] +
        right_side +
        right_tip +
        left_tip[::-1] +
        left_side[::-1] +
        [[-shank_radius - thread_depth, -head_height], [-head_radius, -head_height/2]]
    )
    
    screw_poly = Polygon(full_profile)

    # --- 2. Boundary Sampling & Logical T-Values ---
    boundary_data = []
    distances = np.linspace(0, screw_poly.exterior.length, n_boundary)
    
    # Boundary logic thresholds
    y_tip_threshold = -shank_length - head_height + 0.1
    y_top_threshold = head_height * 0.8
    
    for d in distances:
        p = screw_poly.exterior.interpolate(d)
        x, y = p.x, p.y
        
        if y > y_top_threshold:
            t = 1.0  # Top of head
        elif y < y_tip_threshold:
            t = 0.1  # The Tip
        elif x > 0:
            t = 0.8  # Right Side
        else:
            t = 0.3  # Left Side
            
        boundary_data.append([x, y, t])

    # --- 3. Domain Sampling (Interior) ---
    domain_points = []
    minx, miny, maxx, maxy = screw_poly.bounds
    while len(domain_points) < n_domain:
        batch = np.random.uniform([minx, miny], [maxx, maxy], size=(n_domain, 2))
        for pt in batch:
            if screw_poly.contains(Point(pt)):
                domain_points.append(pt)
            if len(domain_points) >= n_domain: break

    # --- 4. Export to Parquet (Float64) ---
    df_boundary = pd.DataFrame(boundary_data, columns=['x', 'y', 't']).astype(np.float64)
    df_domain = pd.DataFrame(domain_points, columns=['x', 'y']).astype(np.float64)
    
    df_boundary.to_parquet("realistic_screw_boundary.parquet")
    df_domain.to_parquet("realistic_screw_domain.parquet")

    # --- 5. Visualization ---
    plt.figure(figsize=(6, 12))
    plt.scatter(df_domain['x'], df_domain['y'], s=0.5, c='silver', alpha=0.3, label='Internal Domain')
    
    scatter = plt.scatter(df_boundary['x'], df_boundary['y'], 
                         c=df_boundary['t'], cmap='coolwarm', s=4, edgecolors='none')
    
    plt.colorbar(scatter, label='Boundary Temperature / Value (t)')
    plt.title("Realistic Screw Geometry for PINN")
    plt.axis('equal')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    generate_realistic_screw()
