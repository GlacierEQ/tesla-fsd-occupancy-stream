from __future__ import annotations

import unittest

from src.occupancy_integrity import (
    INTEGRITY_EVIDENCE_STATE,
    OccupancyFrame,
    compile_stream_integrity,
)


class OccupancyIntegrityTests(unittest.TestCase):
    def test_nominal_sequence_is_deterministic_and_non_operational(self)->None:
        frames=[
            OccupancyFrame("f1",100,10,100,8),
            OccupancyFrame("f2",110,12,100,8),
            OccupancyFrame("f3",120,13,100,8),
        ]
        first=compile_stream_integrity(frames)
        second=compile_stream_integrity(frames)
        self.assertEqual(first,second)
        self.assertEqual(first["state"],"NOMINAL")
        self.assertEqual(first["evidence_state"],INTEGRITY_EVIDENCE_STATE)
        self.assertFalse(first["vehicle_authority"])
        self.assertFalse(first["driving_command"])
        self.assertEqual(len(first["receipt_sha256"]),64)

    def test_camera_quorum_loss_and_occupancy_jump_require_review(self)->None:
        result=compile_stream_integrity([
            OccupancyFrame("f1",100,10,100,8),
            OccupancyFrame("f2",110,90,100,2),
        ])
        self.assertEqual(result["state"],"REVIEW_REQUIRED")
        codes=[issue["code"] for issue in result["issues"]]
        self.assertIn("INSUFFICIENT_CAMERA_QUORUM",codes)
        self.assertIn("OCCUPANCY_DISCONTINUITY",codes)

    def test_duplicate_and_nonmonotonic_frames_fail_closed(self)->None:
        with self.assertRaises(ValueError):
            compile_stream_integrity([
                OccupancyFrame("f",100,10,100,8),
                OccupancyFrame("f",110,10,100,8),
            ])
        with self.assertRaises(ValueError):
            compile_stream_integrity([
                OccupancyFrame("f1",100,10,100,8),
                OccupancyFrame("f2",100,10,100,8),
            ])

    def test_invalid_frame_values_fail_closed(self)->None:
        with self.assertRaises(ValueError):
            compile_stream_integrity([OccupancyFrame("f1",100,101,100,8)])
        with self.assertRaises(ValueError):
            compile_stream_integrity([],min_camera_count=4)


if __name__=="__main__":
    unittest.main()
