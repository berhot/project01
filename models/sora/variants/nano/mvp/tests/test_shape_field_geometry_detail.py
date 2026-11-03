"""Zero-training numerical regression tests for shape-field geometric evidence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.shape_field_geometry_detail import (
    geometry_stats, compare_masks, save_comparison_grid
)


class GeometryDetailTests(unittest.TestCase):
    def test_bbox_centroid_and_edge(self):
        mask = torch.zeros(1, 1, 32, 32)
        mask[..., 4:10, 9:13] = 1
        stats = geometry_stats(mask)
        self.assertEqual(stats["foreground_pixels"], 24)
        self.assertEqual(stats["bbox"], {
            "left": 9, "right": 12, "top": 4, "bottom": 9,
            "width": 4, "height": 6
        })
        self.assertEqual(stats["centroid"], {"x": 10.5, "y": 6.5})
        self.assertEqual(stats["boundary_pixels"], 16)

    def test_identical_and_shifted_masks(self):
        x = np.zeros((32, 32), dtype=np.float32)
        x[8:16, 8:16] = 1
        self.assertEqual(compare_masks(x, x)["mask_iou"], 1.0)
        self.assertEqual(compare_masks(x, x)["boundary_iou"], 1.0)
        shifted = np.roll(x, 1, axis=1)
        result = compare_masks(shifted, x)
        self.assertLess(result["mask_iou"], 1)
        self.assertEqual(result["centroid_dx"], 1.0)
        self.assertEqual(result["foreground_pixel_error"], 0)

    def test_empty_masks_and_invalid_values(self):
        blank = np.zeros((32, 32))
        self.assertEqual(compare_masks(blank, blank)["mask_iou"], 1.0)
        self.assertIsNone(compare_masks(blank, blank)["centroid_dx"])
        with self.assertRaises(ValueError):
            geometry_stats(np.ones((31, 32)))
        blank[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            geometry_stats(blank)

    def test_visualization_is_generated_without_changing_masks(self):
        blank = np.zeros((32, 32), dtype=np.float32)
        mark = blank.copy()
        mark[4:20, 4:20] = 1
        original = mark.copy()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "grid.png"
            save_comparison_grid(target, mark, blank, mark)
            from PIL import Image
            with Image.open(target) as image:
                self.assertEqual(image.size, (1024, 256))
        np.testing.assert_array_equal(mark, original)


if __name__ == "__main__":
    unittest.main()
