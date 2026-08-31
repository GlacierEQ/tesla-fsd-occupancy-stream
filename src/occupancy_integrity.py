"""Temporal integrity compiler for modeled occupancy frames.

The compiler checks monotonic time, unique frame identity, camera-view quorum,
and abrupt occupancy-fraction discontinuity. It performs no perception,
planning, actuation, or vehicle control.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

INTEGRITY_SCHEMA="glaciereq.tesla-occupancy.integrity.v1"
INTEGRITY_EVIDENCE_STATE="MODELED_OCCUPANCY_INTEGRITY_NOT_TESLA_DRIVING_AUTHORITY"


@dataclass(frozen=True,slots=True)
class OccupancyFrame:
    frame_id:str
    timestamp_ms:int
    occupied_voxels:int
    total_voxels:int
    camera_count:int

    def validate(self)->None:
        if not isinstance(self.frame_id,str) or not self.frame_id.strip():
            raise ValueError("frame_id must be non-empty text")
        if isinstance(self.timestamp_ms,bool) or not isinstance(self.timestamp_ms,int) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a non-negative integer")
        for name,value in (
            ("occupied_voxels",self.occupied_voxels),
            ("total_voxels",self.total_voxels),
            ("camera_count",self.camera_count),
        ):
            if isinstance(value,bool) or not isinstance(value,int):
                raise ValueError(f"{name} must be an integer")
        if self.total_voxels <= 0:
            raise ValueError("total_voxels must be positive")
        if not 0 <= self.occupied_voxels <= self.total_voxels:
            raise ValueError("occupied_voxels must be within 0..total_voxels")
        if self.camera_count <= 0:
            raise ValueError("camera_count must be positive")

    @property
    def occupancy_fraction(self)->float:
        return self.occupied_voxels/self.total_voxels


def _digest(value:object)->str:
    payload=json.dumps(value,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_stream_integrity(
    frames:list[OccupancyFrame],
    *,
    min_camera_count:int=4,
    max_occupancy_delta:float=0.35,
)->dict[str,object]:
    if not frames:
        raise ValueError("at least one frame is required")
    if isinstance(min_camera_count,bool) or not isinstance(min_camera_count,int) or min_camera_count < 1:
        raise ValueError("min_camera_count must be a positive integer")
    if (
        isinstance(max_occupancy_delta,bool)
        or not isinstance(max_occupancy_delta,(int,float))
        or not math.isfinite(float(max_occupancy_delta))
        or not 0.0 <= float(max_occupancy_delta) <= 1.0
    ):
        raise ValueError("max_occupancy_delta must be finite and in 0..1")

    seen:set[str]=set()
    issues:list[dict[str,object]]=[]
    rows:list[dict[str,object]]=[]
    previous:OccupancyFrame|None=None

    for frame in frames:
        if not isinstance(frame,OccupancyFrame):
            raise ValueError("frames must contain OccupancyFrame instances")
        frame.validate()
        if frame.frame_id in seen:
            raise ValueError("frame_id values must be unique")
        seen.add(frame.frame_id)
        if previous is not None and frame.timestamp_ms <= previous.timestamp_ms:
            raise ValueError("frame timestamps must be strictly increasing")

        fraction=frame.occupancy_fraction
        if frame.camera_count < min_camera_count:
            issues.append({
                "frame_id":frame.frame_id,
                "code":"INSUFFICIENT_CAMERA_QUORUM",
                "camera_count":frame.camera_count,
            })
        if previous is not None:
            delta=abs(fraction-previous.occupancy_fraction)
            if delta > float(max_occupancy_delta):
                issues.append({
                    "frame_id":frame.frame_id,
                    "code":"OCCUPANCY_DISCONTINUITY",
                    "delta":round(delta,6),
                })

        rows.append({
            "frame_id":frame.frame_id,
            "timestamp_ms":frame.timestamp_ms,
            "occupancy_fraction":round(fraction,6),
            "camera_count":frame.camera_count,
        })
        previous=frame

    state="NOMINAL" if not issues else "REVIEW_REQUIRED"
    body:dict[str,object]={
        "schema":INTEGRITY_SCHEMA,
        "state":state,
        "frames":rows,
        "issues":issues,
        "min_camera_count":min_camera_count,
        "max_occupancy_delta":float(max_occupancy_delta),
        "evidence_state":INTEGRITY_EVIDENCE_STATE,
        "vehicle_authority":False,
        "driving_command":False,
    }
    body["receipt_sha256"]=_digest(body)
    return body
