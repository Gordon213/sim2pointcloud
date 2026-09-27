#!/usr/bin/env python3
"""Visualize dense point-cloud frames stored by convert_cube_hdf5.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PNG/PLY previews from a pointcloud HDF5 file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pointcloud_preview"))
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--max-plot-points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument(
        "--no-ply", action="store_true", help="Do not write the ASCII PLY file."
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def depth_colors(ranges: np.ndarray) -> np.ndarray:
    minimum = float(np.min(ranges))
    maximum = float(np.max(ranges))
    scale = max(maximum - minimum, 1e-9)
    normalized = np.clip((ranges - minimum) / scale, 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(normalized)
    return np.rint(rgba[:, :3] * 255.0).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("ply\nformat ascii 1.0\n")
        output.write(f"element vertex {len(points)}\n")
        output.write("property float x\nproperty float y\nproperty float z\n")
        output.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        output.write("end_header\n")
        for point, color in zip(points, colors):
            output.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def set_equal_3d_limits(axis, points: np.ndarray) -> None:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * 0.52, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def save_figure(
    path: Path,
    frame_index: int,
    points: np.ndarray,
    ranges: np.ndarray,
    range_image: np.ndarray,
    max_plot_points: int,
    rng: np.random.Generator,
    elevation: float,
    azimuth: float,
) -> None:
    if len(points) > max_plot_points:
        selected = np.sort(rng.choice(len(points), max_plot_points, replace=False))
        plot_points = points[selected]
        plot_ranges = ranges[selected]
    else:
        plot_points = points
        plot_ranges = ranges

    figure = plt.figure(figsize=(11.5, 5.0), dpi=150)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.25))

    range_axis = figure.add_subplot(grid[0, 0])
    color_map = plt.get_cmap("turbo").copy()
    color_map.set_bad("black")
    image = range_axis.imshow(range_image, cmap=color_map)
    range_axis.set_title("Range image (metres)")
    range_axis.set_xlabel("ray column")
    range_axis.set_ylabel("ray row")
    figure.colorbar(image, ax=range_axis, fraction=0.046, pad=0.04)

    cloud_axis = figure.add_subplot(grid[0, 1], projection="3d")
    scatter = cloud_axis.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        c=plot_ranges,
        cmap="turbo",
        s=1.2,
        linewidths=0,
        depthshade=False,
    )
    cloud_axis.set_xlabel("x forward (m)")
    cloud_axis.set_ylabel("y left (m)")
    cloud_axis.set_zlabel("z up (m)")
    cloud_axis.view_init(elev=elevation, azim=azimuth)
    set_equal_3d_limits(cloud_axis, plot_points)
    cloud_axis.set_title(f"Point cloud ({len(points):,} valid returns)")
    figure.colorbar(scatter, ax=cloud_axis, fraction=0.035, pad=0.08, label="range (m)")
    figure.suptitle(f"Simulator-to-point-cloud frame {frame_index:,}")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def export_frame(
    dataset: h5py.Dataset,
    attributes: dict[str, Any],
    frame_index: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    frame_count, ray_count, coordinate_count = dataset.shape
    if coordinate_count != 3:
        raise ValueError(f"expected XYZ points, got shape {dataset.shape}")
    if frame_index < 0:
        frame_index += frame_count
    if not 0 <= frame_index < frame_count:
        raise IndexError(f"frame {frame_index} is outside [0, {frame_count})")

    resolution = int(attributes.get("ray_resolution", round(ray_count**0.5)))
    if resolution * resolution != ray_count:
        raise ValueError(
            f"cannot form a square range image from {ray_count} rays; "
            "ray_resolution metadata is missing or invalid"
        )
    dense_points = np.asarray(dataset[frame_index], dtype=np.float32)
    valid = np.isfinite(dense_points).all(axis=1)
    points = dense_points[valid]
    if len(points) == 0:
        raise RuntimeError(f"frame {frame_index} has no finite XYZ returns")
    ranges = np.linalg.norm(points, axis=1)
    dense_ranges = np.full(ray_count, np.nan, dtype=np.float32)
    dense_ranges[valid] = ranges
    range_image = dense_ranges.reshape(resolution, resolution)

    stem = f"frame_{frame_index:06d}"
    png_path = args.output_dir / f"{stem}.png"
    save_figure(
        png_path,
        frame_index,
        points,
        ranges,
        range_image,
        args.max_plot_points,
        rng,
        args.elevation,
        args.azimuth,
    )
    if not args.no_ply:
        write_ply(args.output_dir / f"{stem}.ply", points, depth_colors(ranges))

    stats = {
        "frame": frame_index,
        "rays": ray_count,
        "valid_returns": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "range_min_metres": float(ranges.min()),
        "range_mean_metres": float(ranges.mean()),
        "range_max_metres": float(ranges.max()),
        "xyz_min": points.min(axis=0).tolist(),
        "xyz_max": points.max(axis=0).tolist(),
        "dataset_attributes": attributes,
    }
    stats_path = args.output_dir / f"{stem}.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"PNG:  {png_path}")
    if not args.no_ply:
        print(f"PLY:  {args.output_dir / f'{stem}.ply'}")
    print(f"JSON: {stats_path}")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.max_plot_points <= 0:
        raise ValueError("max-plot-points must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.input, "r") as input_file:
        if "pointcloud" not in input_file:
            raise KeyError("input HDF5 does not contain 'pointcloud'")
        dataset = input_file["pointcloud"]
        if dataset.ndim != 3:
            raise ValueError(f"pointcloud must be rank 3, got {dataset.shape}")
        attributes = {
            str(key): json_safe(value) for key, value in input_file.attrs.items()
        }
        for frame_index in args.frames:
            export_frame(dataset, attributes, frame_index, args, rng)


if __name__ == "__main__":
    main()

