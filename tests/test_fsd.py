"""Test suite for Tesla FSD Occupancy Stream solution."""
import unittest
from tesla_fsd_occupancy_stream import TeslaFSDOccupancyStream

class TestTeslaFSDOccupancyStream(unittest.TestCase):

    def test_occupancy_processing(self):
        stream = TeslaFSDOccupancyStream(target_fps=36.0, max_latency_ms=10.0)
        res = stream.process_camera_frame(camera_count=8)
        
        self.assertTrue(res["deadline_met"])
        self.assertEqual(res["status"], "HW4_NOMINAL")

if __name__ == "__main__":
    unittest.main()
