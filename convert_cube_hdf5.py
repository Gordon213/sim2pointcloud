#!/usr/bin/env python3
"""Convert saved OGBench Cube MuJoCo states to dense ray point clouds.

The source file is always read-only.  Generation is resumable through an
``OUTPUT.partial`` file and the final name appears only after validation.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401 - registers source HDF5 compression filters
import numpy as np

import stable_worldmodel.envs  # noqa: F401 - registers swm/OGBCube-v0

try:
    from .raycast import MujocoRayCamera, RaySensorConfig
except ImportError:  # Direct execution: python convert_cube_hdf5.py ...
    from raycast import MujocoRayCamera, RaySensorConfig


GENERATOR_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-sense OGBench Cube states with a MuJoCo ray camera."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination HDF5. Required unless --benchmark-frames is used.",
    )
    parser.add_argument("--camera", default="front_pixels")
    parser.add_argument("--ray-resolution", type=int, default=100)
    parser.add_argument("--max-range", type=float, default=2.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--state-batch-size", type=int, default=256)
    parser.add_argument("--output-chunk-frames", type=int, default=8)
    parser.add_argument("--checkpoint-frames", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Convert only the first N rows (0 means the complete dataset).",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=0,
        help="Ray-cast N rows and report speed without creating an output file.",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output is None and args.benchmark_frames == 0:
        raise ValueError("--output is required for dataset generation")
    if args.output is not None and args.source.resolve() == args.output.resolve():
        raise ValueError("source and output paths must differ")
    if args.ray_resolution <= 0 or args.max_range <= 0.0:
        raise ValueError("ray-resolution and max-range must be positive")
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise ValueError("alpha-threshold must be in [0, 1]")
    if args.state_batch_size <= 0 or args.output_chunk_frames <= 0:
        raise ValueError("batch and chunk sizes must be positive")
    if args.state_batch_size % args.output_chunk_frames != 0:
        raise ValueError("state-batch-size must be divisible by output-chunk-frames")
    if args.checkpoint_frames <= 0:
        raise ValueError("checkpoint-frames must be positive")
    if args.max_frames < 0 or args.benchmark_frames < 0:
        raise ValueError("frame limits cannot be negative")


def validate_source(source: h5py.File, max_frames: int) -> tuple[int, int]:
    for key in ("qpos", "qvel"):
        if key not in source:
            raise KeyError(f"source dataset is missing {key!r}")
    total_frames = int(source["qpos"].shape[0])
    if int(source["qvel"].shape[0]) != total_frames:
        raise ValueError("qpos and qvel have different row counts")
    selected_frames = total_frames if max_frames == 0 else min(max_frames, total_frames)
    if selected_frames == 0:
        raise ValueError("source dataset is empty")
    return total_frames, selected_frames


def build_cube_environment(seed: int):
    """Create the exact state-observation environment used by le-wm Cube."""

    env = gym.make(
        "swm/OGBCube-v0",
        env_type="single",
        ob_type="states",
        mode="data_collection",
        height=224,
        width=224,
    )
    env.reset(seed=seed)
    return env, env.unwrapped


def write_metadata(
    output: h5py.File,
    args: argparse.Namespace,
    source_total_frames: int,
    selected_frames: int,
    sensor: MujocoRayCamera,
) -> None:
    source_stat = args.source.stat()
    output.attrs["generator_version"] = GENERATOR_VERSION
    output.attrs["source_path"] = str(args.source.resolve())
    output.attrs["source_size_bytes"] = source_stat.st_size
    output.attrs["source_mtime_ns"] = source_stat.st_mtime_ns
    output.attrs["source_total_frames"] = source_total_frames
    output.attrs["source_frames"] = selected_frames
    output.attrs["ray_resolution"] = args.ray_resolution
    output.attrs["num_rays"] = sensor.num_rays
    output.attrs["camera"] = args.camera
    output.attrs["camera_fovy_degrees"] = sensor.fovy_degrees
    output.attrs["max_range_metres"] = args.max_range
    output.attrs["alpha_threshold"] = args.alpha_threshold
    output.attrs["coordinate_frame"] = "sensor: x forward, y left, z up"
    output.attrs["miss_sentinel"] = "NaN"
    output.attrs["first_hit_only"] = True
    output.attrs["dtype"] = "float32"
    output.attrs["compression"] = "none"
    output.attrs["hostname"] = socket.gethostname()
    output.attrs["stable_worldmodel_version"] = package_version("stable-worldmodel")
    output.attrs["ogbench_version"] = package_version("ogbench")
    output.attrs["mujoco_version"] = package_version("mujoco")
    output.attrs["completed_frames"] = 0
    output.attrs["complete"] = False
    output.attrs["created_unix_time"] = time.time()


def validate_resume(
    output: h5py.File,
    args: argparse.Namespace,
    selected_frames: int,
) -> int:
    if set(output.keys()) != {"pointcloud"}:
        raise ValueError("partial output must contain only 'pointcloud'")
    pointcloud = output["pointcloud"]
    expected_shape = (selected_frames, args.ray_resolution**2, 3)
    if pointcloud.shape != expected_shape or pointcloud.dtype != np.dtype("float32"):
        raise ValueError(
            f"partial output is {pointcloud.shape}/{pointcloud.dtype}; "
            f"expected {expected_shape}/float32"
        )
    checks = {
        "source_path": str(args.source.resolve()),
        "source_size_bytes": args.source.stat().st_size,
        "source_frames": selected_frames,
        "ray_resolution": args.ray_resolution,
        "camera": args.camera,
        "max_range_metres": args.max_range,
        "alpha_threshold": args.alpha_threshold,
    }
    for key, expected in checks.items():
        actual = output.attrs.get(key)
        if actual != expected:
            raise ValueError(
                f"cannot resume: attribute {key!r} is {actual!r}, expected {expected!r}"
            )
    completed = int(output.attrs.get("completed_frames", -1))
    if not 0 <= completed <= selected_frames:
        raise ValueError(f"invalid completed_frames={completed}")
    return completed


def open_output(
    args: argparse.Namespace,
    source_total_frames: int,
    selected_frames: int,
    sensor: MujocoRayCamera,
) -> tuple[h5py.File, int, Path]:
    assert args.output is not None
    partial_path = Path(str(args.output) + ".partial")
    if args.output.exists():
        raise FileExistsError(f"final output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        output = h5py.File(partial_path, "r+")
        completed = validate_resume(output, args, selected_frames)
        print(f"Resuming {partial_path} at row {completed:,}", flush=True)
        return output, completed, partial_path

    output = h5py.File(partial_path, "x", libver="latest")
    output.create_dataset(
        "pointcloud",
        shape=(selected_frames, sensor.num_rays, 3),
        dtype=np.float32,
        chunks=(min(args.output_chunk_frames, selected_frames), sensor.num_rays, 3),
        fillvalue=np.nan,
    )
    write_metadata(output, args, source_total_frames, selected_frames, sensor)
    output.flush()
    print(f"Created {partial_path}", flush=True)
    return output, 0, partial_path


def benchmark(
    args: argparse.Namespace,
    source: h5py.File,
    base_env,
    sensor: MujocoRayCamera,
    total_frames: int,
) -> None:
    count = min(args.benchmark_frames, total_frames)
    qpos = np.asarray(source["qpos"][:count])
    qvel = np.asarray(source["qvel"][:count])
    valid_total = 0
    started = time.monotonic()
    for frame_qpos, frame_qvel in zip(qpos, qvel):
        base_env.set_state(frame_qpos, frame_qvel)
        valid_total += int(sensor.scan().valid.sum())
    elapsed = time.monotonic() - started
    rate = count / max(elapsed, 1e-9)
    print(f"Benchmark: {count:,} frames in {elapsed:.3f}s ({rate:.2f} frames/s)")
    print(f"Mean valid returns: {valid_total / max(count, 1):,.1f}")
    print(f"Full-dataset ray-cast estimate: {total_frames / rate / 3600.0:.2f}h")


def generate(
    args: argparse.Namespace,
    source: h5py.File,
    base_env,
    sensor: MujocoRayCamera,
    source_total_frames: int,
    selected_frames: int,
) -> None:
    output, completed, partial_path = open_output(
        args, source_total_frames, selected_frames, sensor
    )
    pointcloud = output["pointcloud"]
    run_start = time.monotonic()
    run_start_frame = completed
    next_checkpoint = (
        (completed // args.checkpoint_frames) + 1
    ) * args.checkpoint_frames

    try:
        while completed < selected_frames:
            stop = min(completed + args.state_batch_size, selected_frames)
            qpos_batch = np.asarray(source["qpos"][completed:stop])
            qvel_batch = np.asarray(source["qvel"][completed:stop])
            points_batch = np.empty(
                (stop - completed, sensor.num_rays, 3), dtype=np.float32
            )
            valid_total = 0
            for index, (frame_qpos, frame_qvel) in enumerate(
                zip(qpos_batch, qvel_batch)
            ):
                base_env.set_state(frame_qpos, frame_qvel)
                frame = sensor.scan()
                points_batch[index] = frame.points
                valid_total += int(frame.valid.sum())
            pointcloud[completed:stop] = points_batch
            completed = stop

            if completed >= next_checkpoint or completed == selected_frames:
                output.attrs.modify("completed_frames", completed)
                output.attrs["last_checkpoint_unix_time"] = time.time()
                output.flush()
                elapsed = time.monotonic() - run_start
                processed = completed - run_start_frame
                rate = processed / max(elapsed, 1e-9)
                remaining = (selected_frames - completed) / max(rate, 1e-9)
                allocated_bytes = partial_path.stat().st_blocks * 512
                print(
                    f"frames={completed:,}/{selected_frames:,} "
                    f"({100.0 * completed / selected_frames:.3f}%) "
                    f"rate={rate:.2f} frames/s eta={remaining / 3600.0:.2f}h "
                    f"allocated={allocated_bytes / 1e9:.2f}GB "
                    f"last_batch_valid_mean={valid_total / len(points_batch):.1f}",
                    flush=True,
                )
                while next_checkpoint <= completed:
                    next_checkpoint += args.checkpoint_frames

        output.attrs.modify("completed_frames", selected_frames)
        output.attrs.modify("complete", True)
        output.attrs["completed_unix_time"] = time.time()
        output.flush()
    finally:
        output.close()

    with h5py.File(partial_path, "r") as check:
        if int(check.attrs["completed_frames"]) != selected_frames:
            raise RuntimeError("completed frame count does not match output shape")
        if not bool(check.attrs["complete"]):
            raise RuntimeError("output was not marked complete")
        dataset = check["pointcloud"]
        for index in sorted({0, selected_frames // 2, selected_frames - 1}):
            frame = np.asarray(dataset[index])
            finite = np.isfinite(frame).all(axis=1)
            if not finite.any():
                raise RuntimeError(f"validation frame {index} has no valid returns")
            complete_rows = finite | np.isnan(frame).all(axis=1)
            if not complete_rows.all():
                raise RuntimeError(f"validation frame {index} contains partial-NaN rows")

    assert args.output is not None
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite final output: {args.output}")
    os.replace(partial_path, args.output)
    print(f"COMPLETE: {args.output}")
    print(f"Logical size: {args.output.stat().st_size:,} bytes")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    env = None
    try:
        with h5py.File(args.source, "r") as source:
            total_frames, selected_frames = validate_source(source, args.max_frames)
            env, base_env = build_cube_environment(args.seed)
            config = RaySensorConfig(
                camera=args.camera,
                resolution=args.ray_resolution,
                max_range=args.max_range,
                alpha_threshold=args.alpha_threshold,
            )
            with MujocoRayCamera(base_env.model, base_env.data, config) as sensor:
                print(
                    f"source_frames={total_frames:,} selected_frames={selected_frames:,} "
                    f"rays_per_frame={sensor.num_rays:,} "
                    f"logical_output={selected_frames * sensor.num_rays * 12 / 1e9:.2f}GB",
                    flush=True,
                )
                if args.benchmark_frames:
                    benchmark(args, source, base_env, sensor, total_frames)
                else:
                    generate(
                        args,
                        source,
                        base_env,
                        sensor,
                        total_frames,
                        selected_frames,
                    )
    finally:
        if env is not None:
            env.close()


def main() -> None:
    args = parse_args()
    if args.benchmark_frames:
        run(args)
        return

    assert args.output is not None
    lock_path = Path(str(args.output) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another generator holds {lock_path}") from error
        run(args)
    finally:
        lock_file.close()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
