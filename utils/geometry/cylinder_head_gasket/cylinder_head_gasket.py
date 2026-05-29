import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Cylinder Head Gasket Geometry Generator

def create_gasket_geometry(plate_width, plate_height, hole_radius, screw_radius, n_cylinders, n_screws, n_domain, n_boundary, space, output_path, plot):
    # Increase space between cylinder holes for better separation
    space = max(space, 0.03)  # Set a larger minimum space (e.g., 3 cm)

    # Plate rectangle
    plate = Polygon([
        (0, 0), (plate_width, 0), (plate_width, plate_height), (0, plate_height)
    ])

    # Cylinder holes: distribute horizontally, centered vertically, with at least 'space' between
    total_cyl_width = n_cylinders * 2 * hole_radius + (n_cylinders - 1) * space
    x_start = (plate_width - total_cyl_width) / 2 + hole_radius
    y_center = plate_height / 2
    cylinders = []
    for i in range(n_cylinders):
        x = x_start + i * (2 * hole_radius + space)
        cylinders.append(Point(x, y_center).buffer(hole_radius, resolution=64))

    # Screw holes: randomly placed, not overlapping cylinders, random holes, or each other
    screws = []
    rng = np.random.default_rng(42)
    screw_count = 0
    max_attempts = 10000
    attempts = 0
    while screw_count < n_screws and attempts < max_attempts:
        x = rng.uniform(screw_radius, plate_width - screw_radius)
        y = rng.uniform(screw_radius, plate_height - screw_radius)
        pt = Point(x, y)
        # Ensure no intersection with cylinders or other screws
        if (
            plate.contains(pt)
            and all(cyl.distance(pt) > hole_radius + screw_radius + space/2 for cyl in cylinders)
            and all(screw.distance(pt) > 2 * screw_radius + 1e-6 for screw in screws)  # Strict non-overlap
        ):
            screws.append(pt.buffer(screw_radius, resolution=32))
            screw_count += 1
        attempts += 1

    # Random small holes (smaller than main holes)
    random_holes = []
    for _ in range(3):
        r = rng.uniform(screw_radius * 1.5, hole_radius * 0.7)
        x = rng.uniform(r, plate_width - r)
        y = rng.uniform(r, plate_height - r)
        pt = Point(x, y)
        if (
            plate.contains(pt)
            and all(cyl.distance(pt) > hole_radius + r + space/2 for cyl in cylinders)
            and all(screw.distance(pt) > screw_radius + r + space/2 for screw in screws)
            and all(hole.distance(pt) > r * 2 + 1e-6 for hole in random_holes)
        ):
            random_holes.append(pt.buffer(r, resolution=32))

    # Subtract all holes from plate
    all_holes = unary_union(cylinders + screws + random_holes)
    domain = plate.difference(all_holes)

    # Boundary sampling (outer + holes)
    boundary_points = []
    # Outer boundary
    outer = np.array(domain.exterior.coords)
    idx = np.round(np.linspace(0, len(outer) - 1, n_boundary // (1 + len(cylinders) + len(screws) + len(random_holes)))).astype(int)
    boundary_points.extend(outer[idx])
    # Holes boundaries
    for interior in domain.interiors:
        coords = np.array(interior.coords)
        idx = np.round(np.linspace(0, len(coords) - 1, max(2, n_boundary // (1 + len(cylinders) + len(screws) + len(random_holes))))).astype(int)
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
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(boundary_points, columns=["x", "y"]).to_parquet(out_dir / "boundary_data.parquet", index=False)
    pd.DataFrame(domain_points, columns=["x", "y"]).to_parquet(out_dir / "domain_data.parquet", index=False)

    # Plot
    if plot:
        plt.figure(figsize=(10, 3))
        xs, ys = zip(*plate.exterior.coords)
        plt.plot(xs, ys, 'k-', lw=2, label='Plate')
        for c in cylinders:
            xs, ys = zip(*c.exterior.coords)
            plt.plot(xs, ys, 'b-', lw=1, label='Cylinder' if c is cylinders[0] else None)
        for s in screws:
            xs, ys = zip(*s.exterior.coords)
            plt.plot(xs, ys, 'g-', lw=1, label='Screw' if s is screws[0] else None)
        for r in random_holes:
            xs, ys = zip(*r.exterior.coords)
            plt.plot(xs, ys, 'm-', lw=1, label='Random Hole' if r is random_holes[0] else None)
        plt.scatter(domain_points[:,0], domain_points[:,1], s=0.2, c='orange', label='Domain')
        plt.scatter(boundary_points[:,0], boundary_points[:,1], s=2, c='red', label='Boundary')
        plt.legend()
        plt.axis('equal')
        plt.title('Cylinder Head Gasket Geometry')
        plt.savefig(out_dir / "geometry.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cylinder Head Gasket geometry and dataset.")
    parser.add_argument('--output-path', type=str, required=True, help='Output directory path for Parquet files')
    parser.add_argument('--plot', action='store_true', help='Plot the geometry and points and save figure')
    parser.add_argument('--plate_width', type=float, default=1.0, help='Plate width (m)')
    parser.add_argument('--plate_height', type=float, default=0.25, help='Plate height (m)')
    parser.add_argument('--hole_radius', type=float, default=0.1, help='Cylinder hole radius (m)')
    parser.add_argument('--screw_radius', type=float, default=0.01, help='Screw hole max radius (m)')
    parser.add_argument('--n_cylinders', type=int, default=4, help='Number of cylinder holes')
    parser.add_argument('--n_screws', type=int, default=10, help='Number of screw holes')
    parser.add_argument('--n_domain', type=int, default=20000, help='Number of domain points')
    parser.add_argument('--n_boundary', type=int, default=2000, help='Number of boundary points')
    parser.add_argument('--space', type=float, default=0.03, help='Minimum space between cylinder holes (m)')
    args = parser.parse_args()

    create_gasket_geometry(
        plate_width=args.plate_width,
        plate_height=args.plate_height,
        hole_radius=args.hole_radius,
        screw_radius=args.screw_radius,
        n_cylinders=args.n_cylinders,
        n_screws=args.n_screws,
        n_domain=args.n_domain,
        n_boundary=args.n_boundary,
        space=args.space,
        output_path=args.output_path,
        plot=args.plot
    )