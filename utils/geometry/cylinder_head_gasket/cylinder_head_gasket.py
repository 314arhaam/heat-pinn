import argparse
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# Cylinder Head Gasket Geometry Generator

def create_gasket_geometry(plate_width, plate_height, hole_radius, screw_radius, n_holes, n_screws, n_domain, n_boundary, output_path, plot):
    # Plate rectangle
    plate = Polygon([
        (0, 0), (plate_width, 0), (plate_width, plate_height), (0, plate_height)
    ])

    # Main holes (evenly spaced along the plate)
    holes = []
    x_spacing = plate_width / (n_holes + 1)
    y_center = plate_height / 2
    for i in range(1, n_holes + 1):
        x = i * x_spacing
        holes.append(Point(x, y_center).buffer(hole_radius, resolution=64))

    # Screw holes (randomly placed, not overlapping main holes or outside plate)
    screws = []
    rng = np.random.default_rng(42)
    screw_count = 0
    while screw_count < n_screws:
        x = rng.uniform(screw_radius, plate_width - screw_radius)
        y = rng.uniform(screw_radius, plate_height - screw_radius)
        pt = Point(x, y)
        if plate.contains(pt) and all(hole.distance(pt) > hole_radius + screw_radius for hole in holes):
            screws.append(pt.buffer(screw_radius, resolution=32))
            screw_count += 1

    # Random small holes (smaller than main holes)
    random_holes = []
    for _ in range(3):
        r = rng.uniform(screw_radius * 1.5, hole_radius * 0.7)
        x = rng.uniform(r, plate_width - r)
        y = rng.uniform(r, plate_height - r)
        pt = Point(x, y)
        if plate.contains(pt) and all(hole.distance(pt) > hole_radius + r for hole in holes):
            random_holes.append(pt.buffer(r, resolution=32))

    # Subtract all holes from plate
    all_holes = unary_union(holes + screws + random_holes)
    domain = plate.difference(all_holes)

    # Boundary sampling (outer + holes)
    boundary_points = []
    # Outer boundary
    outer = np.array(domain.exterior.coords)
    idx = np.round(np.linspace(0, len(outer) - 1, n_boundary // (1 + len(holes) + len(screws) + len(random_holes)))).astype(int)
    boundary_points.extend(outer[idx])
    # Holes boundaries
    for interior in domain.interiors:
        coords = np.array(interior.coords)
        idx = np.round(np.linspace(0, len(coords) - 1, max(2, n_boundary // (1 + len(holes) + len(screws) + len(random_holes))))).astype(int)
        boundary_points.extend(coords[idx])
    boundary_points = np.array(boundary_points, dtype=np.float64)

    # Domain sampling (rejection sampling)
    minx, miny, maxx, maxy = domain.bounds
    domain_points = []
    while len(domain_points) < n_domain:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt = Point(x, y)
        if domain.contains(pt):
            domain_points.append((x, y))
    domain_points = np.array(domain_points, dtype=np.float64)

    # Save to Parquet
    pd.DataFrame(boundary_points, columns=["x", "y"]).to_parquet(f"{output_path}_boundary.parquet")
    pd.DataFrame(domain_points, columns=["x", "y"]).to_parquet(f"{output_path}_domain.parquet")

    # Plot
    if plot:
        plt.figure(figsize=(10, 3))
        xs, ys = zip(*plate.exterior.coords)
        plt.plot(xs, ys, 'k-', lw=2, label='Plate')
        for h in holes:
            xs, ys = zip(*h.exterior.coords)
            plt.plot(xs, ys, 'b-', lw=1)
        for s in screws:
            xs, ys = zip(*s.exterior.coords)
            plt.plot(xs, ys, 'g-', lw=1)
        for r in random_holes:
            xs, ys = zip(*r.exterior.coords)
            plt.plot(xs, ys, 'm-', lw=1)
        plt.scatter(domain_points[:,0], domain_points[:,1], s=0.2, c='orange', label='Domain')
        plt.scatter(boundary_points[:,0], boundary_points[:,1], s=2, c='red', label='Boundary')
        plt.legend()
        plt.axis('equal')
        plt.title('Cylinder Head Gasket Geometry')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cylinder Head Gasket geometry and dataset.")
    parser.add_argument('--output', type=str, required=True, help='Output path prefix for Parquet files')
    parser.add_argument('--plot', action='store_true', help='Plot the geometry and points')
    parser.add_argument('--plate_width', type=float, default=1.0, help='Plate width (m)')
    parser.add_argument('--plate_height', type=float, default=0.25, help='Plate height (m)')
    parser.add_argument('--hole_radius', type=float, default=0.1, help='Main hole radius (m)')
    parser.add_argument('--screw_radius', type=float, default=0.01, help='Screw hole max radius (m)')
    parser.add_argument('--n_holes', type=int, default=4, help='Number of main holes')
    parser.add_argument('--n_screws', type=int, default=12, help='Number of screw holes')
    parser.add_argument('--n_domain', type=int, default=20000, help='Number of domain points')
    parser.add_argument('--n_boundary', type=int, default=2000, help='Number of boundary points')
    args = parser.parse_args()

    create_gasket_geometry(
        plate_width=args.plate_width,
        plate_height=args.plate_height,
        hole_radius=args.hole_radius,
        screw_radius=args.screw_radius,
        n_holes=args.n_holes,
        n_screws=args.n_screws,
        n_domain=args.n_domain,
        n_boundary=args.n_boundary,
        output_path=args.output,
        plot=args.plot
    )