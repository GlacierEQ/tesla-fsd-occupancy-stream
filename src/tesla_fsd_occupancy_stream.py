"""Deterministic occupancy-stream scenario arithmetic.

This independent portfolio model does not process Tesla camera feeds, execute on
HW3/HW4, run CUDA, or measure vehicle latency. All timing values are derived
from explicit caller-visible modeling assumptions.
"""
from __future__ import annotations

import math
from typing import Any

EVIDENCE_STATE = "MODELED_OCCUPANCY_SCENARIO_NOT_TESLA_FSD_MEASUREMENT"


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric=float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


class TeslaFSDOccupancyStream:
    """Model voxel-grid work and deadline arithmetic from explicit assumptions."""

    def __init__(
        self,
        target_fps: float = 36.0,
        max_latency_ms: float = 10.0,
        modeled_voxels_per_ms: float = 349525.3333333333,
        modeled_camera_overhead_ms: float = 0.8,
    ) -> None:
        self.target_fps=_positive_finite("target_fps",target_fps)
        self.max_latency_ms=_positive_finite("max_latency_ms",max_latency_ms)
        self.modeled_voxels_per_ms=_positive_finite(
            "modeled_voxels_per_ms",modeled_voxels_per_ms
        )
        if (
            isinstance(modeled_camera_overhead_ms,bool)
            or not isinstance(modeled_camera_overhead_ms,(int,float))
            or not math.isfinite(float(modeled_camera_overhead_ms))
            or float(modeled_camera_overhead_ms) < 0
        ):
            raise ValueError("modeled_camera_overhead_ms must be finite and non-negative")
        self.modeled_camera_overhead_ms=float(modeled_camera_overhead_ms)

    def process_camera_frame(
        self,
        camera_count: int = 8,
        voxel_grid_dim: tuple[int,int,int] = (128,128,32),
    ) -> dict[str,Any]:
        """Return modeled workload/deadline values; no camera or vehicle is contacted."""

        if isinstance(camera_count,bool) or not isinstance(camera_count,int) or camera_count < 1:
            raise ValueError("camera_count must be a positive integer")
        if (
            not isinstance(voxel_grid_dim,tuple)
            or len(voxel_grid_dim) != 3
            or any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in voxel_grid_dim)
        ):
            raise ValueError("voxel_grid_dim must contain three positive integers")

        voxels_total=voxel_grid_dim[0]*voxel_grid_dim[1]*voxel_grid_dim[2]
        modeled_latency_ms=(
            voxels_total/self.modeled_voxels_per_ms
            + camera_count*self.modeled_camera_overhead_ms
        )
        deadline_met=modeled_latency_ms <= self.max_latency_ms
        modeled_fps_upper_bound=1000.0/modeled_latency_ms

        return {
            "camera_count":camera_count,
            "voxel_grid_dim":voxel_grid_dim,
            "voxels_modeled":voxels_total,
            "latency_ms":round(modeled_latency_ms,6),
            "modeled_fps_upper_bound":round(modeled_fps_upper_bound,3),
            "target_fps":self.target_fps,
            "deadline_met":deadline_met,
            "status":"MODELED_DEADLINE_MET" if deadline_met else "MODELED_DEADLINE_MISSED",
            "evidence_state":EVIDENCE_STATE,
            "hardware_measurement":False,
            "vehicle_authority":False,
            "camera_feed_access":False,
        }
