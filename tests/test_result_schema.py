from __future__ import annotations

import unittest

from src.detection.coordinate_utils import local_to_frame_bbox
from src.detection.result_schema import canonical_detection_record, validate_detection


class DetectionSchemaTest(unittest.TestCase):
    def test_slice_local_bbox_stays_local_and_frame_bbox_adds_slice_offset(self) -> None:
        slice_xyxy = [100.0, 200.0, 1124.0, 1224.0]
        local_bbox = [10.0, 20.0, 60.0, 90.0]
        frame_bbox = local_to_frame_bbox(local_bbox, slice_xyxy)

        row = canonical_detection_record(
            frame_id="frame_001.jpg",
            slice_id="frame_001.jpg::x100_y200_w1024_h1024",
            bbox_xyxy=local_bbox,
            class_id=1,
            score=0.9,
            model_name="schema_test_detector",
            backend="unit_test",
            input_width=1024,
            input_height=1024,
            inference_time_ms=1.25,
            bbox_xyxy_frame=frame_bbox,
            slice_x1=slice_xyxy[0],
            slice_y1=slice_xyxy[1],
            slice_x2=slice_xyxy[2],
            slice_y2=slice_xyxy[3],
        )

        validate_detection(row)
        self.assertEqual(row["bbox_xyxy"], local_bbox)
        self.assertEqual(row["bbox_xyxy_frame"], [110.0, 220.0, 160.0, 290.0])


if __name__ == "__main__":
    unittest.main()
