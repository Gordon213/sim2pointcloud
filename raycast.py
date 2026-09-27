"""MuJoCo camera-aligned ray casting for fixed-size XYZ point clouds.

The returned array preserves the camera raster order and uses NaN for rays
that do not hit geometry within ``max_range``.  Point coordinates are expressed
in a robotics-friendly sensor frame: +X forward, +Y left, +Z up.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class RaySensorConfig:
    """Configuration for a square pinhole ray sensor."""

    camera: str = "front_pixels"
    resolution: int = 100
    max_range: float = 2.0
    alpha_threshold: float = 0.5

    def validate(self) -> None:
        if self.resolution < 1:
            raise ValueError("resolution must be positive")
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if not 0.0 <= self.alpha_threshold <= 1.0:
            raise ValueError("alpha_threshold must be in [0, 1]")


@dataclass
class PointCloudFrame:
    """One dense ray frame plus diagnostic hit information."""

    points: np.ndarray
    valid: np.ndarray
    ranges: np.ndarray
    geom_ids: np.ndarray

    @property
    def valid_points(self) -> np.ndarray:
        return self.points[self.valid]


class MujocoRayCamera:
    """Turn a MuJoCo camera into a dense first-hit XYZ ray sensor.

    The class owns a temporary geometry-group modification used to exclude
    translucent geometry.  Use it as a context manager, or call ``close()``,
    so the original groups are restored.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: RaySensorConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or RaySensorConfig()
        self.config.validate()
        self._closed = False

        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera
        )
        if self.camera_id < 0:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                for index in range(model.ncam)
            ]
            raise ValueError(
                f"unknown MuJoCo camera {self.config.camera!r}; "
                f"available cameras: {available}"
            )

        self.fovy_degrees = float(model.cam_fovy[self.camera_id])
        fovy = np.deg2rad(self.fovy_degrees)
        resolution = self.config.resolution
        focal = 0.5 * resolution / np.tan(0.5 * fovy)
        center = resolution / 2.0
        u, v = np.meshgrid(
            np.arange(resolution, dtype=np.float64) + 0.5,
            np.arange(resolution, dtype=np.float64) + 0.5,
        )
        rays_camera = np.stack(
            [
                (u - center) / focal,
                -(v - center) / focal,
                -np.ones_like(u),
            ],
            axis=-1,
        )
        rays_camera /= np.linalg.norm(rays_camera, axis=-1, keepdims=True)
        self._rays_camera = np.ascontiguousarray(
            rays_camera.reshape(-1, 3), dtype=np.float64
        )

        # MuJoCo camera local -Z/+X/+Y becomes sensor +X/-Y/+Z.
        self._rays_sensor = np.ascontiguousarray(
            np.stack(
                [-rays_camera[..., 2], -rays_camera[..., 0], rays_camera[..., 1]],
                axis=-1,
            ).reshape(-1, 3),
            dtype=np.float32,
        )
        self._geom_ids = np.empty(self.num_rays, dtype=np.int32)
        self._ranges = np.empty(self.num_rays, dtype=np.float64)
        self._original_geom_groups = model.geom_group.copy()
        self._geom_group_mask = self._prepare_geometry_filter()

    @property
    def num_rays(self) -> int:
        return self.config.resolution**2

    def _prepare_geometry_filter(self) -> np.ndarray:
        mask = np.ones(6, dtype=np.uint8)
        transparent = self.model.geom_rgba[:, 3] < self.config.alpha_threshold
        if not transparent.any():
            return mask

        used_groups = set(int(group) for group in self._original_geom_groups)
        filter_group = next(
            (group for group in range(5, -1, -1) if group not in used_groups),
            None,
        )
        if filter_group is None:
            raise RuntimeError(
                "cannot exclude translucent geometry: all six MuJoCo geom "
                "groups are already in use"
            )
        self.model.geom_group[transparent] = filter_group
        mask[filter_group] = 0
        return mask

    def scan(self) -> PointCloudFrame:
        """Cast one frame at the environment's current forwarded state."""

        if self._closed:
            raise RuntimeError("ray sensor is closed")

        # Refresh the camera transform every scan so body-mounted cameras work.
        rotation_camera_to_world = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        rays_world = np.ascontiguousarray(
            self._rays_camera @ rotation_camera_to_world.T,
            dtype=np.float64,
        )
        origin_world = np.asarray(
            self.data.cam_xpos[self.camera_id], dtype=np.float64
        ).copy()

        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin_world,
            rays_world.reshape(-1),
            self._geom_group_mask,
            True,
            -1,
            self._geom_ids,
            self._ranges,
            None,
            self.num_rays,
            self.config.max_range,
        )
        valid = (
            (self._geom_ids >= 0)
            & np.isfinite(self._ranges)
            & (self._ranges > 0.0)
            & (self._ranges <= self.config.max_range)
        )
        points = np.full((self.num_rays, 3), np.nan, dtype=np.float32)
        points[valid] = (
            self._ranges[valid, None].astype(np.float32) * self._rays_sensor[valid]
        )
        ranges = np.full(self.num_rays, np.nan, dtype=np.float32)
        ranges[valid] = self._ranges[valid].astype(np.float32)
        return PointCloudFrame(
            points=points,
            valid=valid.copy(),
            ranges=ranges,
            geom_ids=self._geom_ids.copy(),
        )

    def close(self) -> None:
        if not self._closed:
            self.model.geom_group[:] = self._original_geom_groups
            self._closed = True

    def __enter__(self) -> "MujocoRayCamera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

