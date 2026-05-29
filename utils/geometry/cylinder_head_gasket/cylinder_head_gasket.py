import argparse
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Function to create the Cylinder Head Gasket geometry
def create_cylinder_head_gasket(output_path, plot=False):
    # Define the outer boundary
    outer_boundary = Point(0, 0).buffer(1.0)  # Outer circle with radius 1.0

    # Define the inner holes
    holes = [
        Point(0.5, 0.5).buffer(0.1),
        Point(-0.5, 0.5).buffer(0.1),
        Point(0.5, -0.5).buffer(0.1),
        Point(-0.5, -0.5).buffer(0.1)
    ]

    # Add smaller screw holes
    screw_holes = [
        Point(x, y).buffer(0.01)
        for x, y in np.random.uniform(-0.9, 0.9, size=(10, 2))
    ]

    # Combine all holes
    all_holes = unary_union(holes + screw_holes)

    # Subtract holes from the outer boundary
    gasket = outer_boundary.difference(all_holes)

    # Sample boundary points
    boundary_points = np.array(gasket.exterior.coords)

    # Sample domain points
    minx, miny, maxx, maxy = gasket.bounds
    domain_points = []
    while len(domain_points) < 20000:
        x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
        if gasket.contains(Point(x, y)):
            domain_points.append((x, y))
    domain_points = np.array(domain_points)

    # Save to Parquet
    boundary_df = pd.DataFrame(boundary_points, columns=['x', 'y'])
    domain_df = pd.DataFrame(domain_points, columns=['x', 'y'])
    boundary_df.to_parquet(f"{output_path}_boundary.parquet")
    domain_df.to_parquet(f"{output_path}_domain.parquet")

    # Plot if required
    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'r-', label='Boundary')
        plt.scatter(domain_points[:, 0], domain_points[:, 1], s=1, label='Domain')
        plt.legend()
        plt.axis('equal')
        plt.show()

# CLI setup
def main():
    parser = argparse.ArgumentParser(description="Generate Cylinder Head Gasket geometry.")
    parser.add_argument("--output", type=str, required=True, help="Output path for the Parquet files.")
    parser.add_argument("--plot", action="store_true", help="Plot the geometry.")
    args = parser.parse_args()

    create_cylinder_head_gasket(args.output, args.plot)

if __name__ == "__main__":
    main()