"""Occupancy-stream unit tests — deadline, throttle, receipt fields."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tesla_fsd_occupancy_stream import TeslaFSDOccupancyStream


class TestTeslaFSDOccupancyStream(unittest.TestCase):
    def test_occupancy_processing(self):
        stream = TeslaFSDOccupancyStream(target_fps=36.0, max_latency_ms=10.0)
        res = stream.process_camera_frame(camera_count=8)
        self.assertTrue(res["deadline_met"])
        self.assertEqual(res["status"], "HW4_NOMINAL")
        self.assertEqual(res["camera_count"], 8)

    def test_throttle_under_tight_deadline(self):
        stream = TeslaFSDOccupancyStream(target_fps=36.0, max_latency_ms=0.1)
        res = stream.process_camera_frame(camera_count=8)
        self.assertFalse(res["deadline_met"])
        self.assertEqual(res["status"], "HW4_THROTTLED")

    def test_receipt_fields_present(self):
        stream = TeslaFSDOccupancyStream()
        res = stream.process_camera_frame()
        for key in ("latency_ms", "fps_achieved", "voxels_processed", "deadline_met", "status"):
            self.assertIn(key, res)


if __name__ == "__main__":
    unittest.main()
