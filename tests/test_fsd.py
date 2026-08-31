"""Tests for deterministic occupancy-stream scenario arithmetic."""
from __future__ import annotations

import math
import unittest

from src.tesla_fsd_occupancy_stream import (
    EVIDENCE_STATE,
    TeslaFSDOccupancyStream,
)


class OccupancyScenarioTests(unittest.TestCase):
    def test_default_modeled_deadline(self) -> None:
        stream=TeslaFSDOccupancyStream()
        result=stream.process_camera_frame(camera_count=8)
        self.assertTrue(result["deadline_met"])
        self.assertEqual(result["status"],"MODELED_DEADLINE_MET")
        self.assertEqual(result["evidence_state"],EVIDENCE_STATE)
        self.assertFalse(result["hardware_measurement"])
        self.assertFalse(result["vehicle_authority"])
        self.assertFalse(result["camera_feed_access"])

    def test_tight_deadline_is_modeled_miss(self) -> None:
        result=TeslaFSDOccupancyStream(max_latency_ms=0.1).process_camera_frame()
        self.assertFalse(result["deadline_met"])
        self.assertEqual(result["status"],"MODELED_DEADLINE_MISSED")

    def test_result_is_deterministic(self) -> None:
        stream=TeslaFSDOccupancyStream()
        self.assertEqual(stream.process_camera_frame(),stream.process_camera_frame())

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            TeslaFSDOccupancyStream(target_fps=math.nan)
        with self.assertRaises(ValueError):
            TeslaFSDOccupancyStream().process_camera_frame(camera_count=0)
        with self.assertRaises(ValueError):
            TeslaFSDOccupancyStream().process_camera_frame(voxel_grid_dim=(128,0,32))


if __name__=="__main__":
    unittest.main()
