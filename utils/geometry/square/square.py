import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from pathlib import Path
import os

def generate_data(
    data_per_boundary: int,
    Nc: int,
    output_path: str,
    plot: bool,
    bc_values: list[float],
):
    """
    Generate boundary and collocation data and optionally plot them.
    Saves:
      - boundary_data.parquet
      - domain_data.parquet
      - geometry.png (if plot)
    in output_path.
    """

    n_bc = 4
    if len(bc_values) != 4:
        raise ValueError(f"bc_values must have length 4, got {len(bc_values)}")

    # --- Boundary data generation ---
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros((n_bc, data_per_boundary, 3), dtype=np.float64)

    for i, j in zip(range(n_bc), [-1, +1, -1, +1]):
        points = (engine.random(n=data_per_boundary)[:, 0] - 0.5) * 2
        if i < 2:
            # vertical boundaries: x fixed, y varies
            data[i, :, 0] = j
            data[i, :, 1] = points
        else:
            # horizontal boundaries: x varies, y fixed
            data[i, :, 0] = points
            data[i, :, 1] = j

    # assign boundary condition values (t) for each of the 4 boundaries
    for i in range(n_bc):
        data[i, :, 2] = bc_values[i]

    data = data.reshape(data_per_boundary * n_bc, 3)
    boundary_data = pd.DataFrame(data, columns=["x", "y", "t"])

    # --- Collocation (domain) data ---
    engine = qmc.LatinHypercube(d=2)
    colloc = engine.random(n=Nc)
    colloc = 2 * (colloc - 0.5)  # map to [-1, 1]^2
    domain_data = pd.DataFrame(colloc, columns=["x", "y"])

    # --- Ensure output directory exists ---
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save Parquet files ---
    boundary_path = out_dir / "boundary_data.parquet"
    domain_path = out_dir / "domain_data.parquet"
    boundary_data.to_parquet(boundary_path, index=False)
    domain_data.to_parquet(domain_path, index=False)

    print(f"Saved boundary data to:  {boundary_path}")
    print(f"Saved domain data to:    {domain_path}")

    # --- Optional plotting ---
    if plot:
        plt.figure()
        plt.title("Boundary Data points and Collocation points")
        plt.scatter(data[:, 0], data[:, 1], marker="x", c="k", label="Boundary")
        plt.scatter(colloc[:, 0], colloc[:, 1], s=0.2, marker=".", c="r", label="Domain")
        plt.legend()
        plt.axis('equal')
        # Save plot in the same directory as data
        plot_path = out_dir / "geometry.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to: {plot_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="square-data-gen",
        description="Generate boundary and collocation data for a square PINN problem.",
    )
    parser.add_argument(
        "--data-per-boundary",
        type=int,
        default=500,
        help="Number of points per boundary (default: 500).",
    )
    parser.add_argument(
        "--Nc",
        type=int,
        default=20000,
        help="Number of collocation (domain) points (default: 20000).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./output/",
        help='Directory to save Parquet files (default: "./output/").',
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot of boundary and collocation points.",
    )
    parser.add_argument(
        "--bc-values",
        type=float,
        nargs=4,
        metavar=("BC0", "BC1", "BC2", "BC3"),
        default=[50.0, 90.0, 0.0, 0.0],
        help=(
            "Four boundary condition values (t) for the 4 boundaries, "
            "in order [left, right, bottom, top]. "
            'Default: 50.0 90.0 0.0 0.0'
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_data(
        data_per_boundary=args.data_per_boundary,
        Nc=args.Nc,
        output_path=args.output_path,
        plot=args.plot,
        bc_values=args.bc_values,
    )
