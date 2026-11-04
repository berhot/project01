"""Read-only geometry summary validation and regression tests."""
import unittest
from scripts.shape_field_geometry_review import review


def sample_report():
    items = []
    for shape in ("circle", "square", "triangle"):
        for color in ("red", "green", "blue", "yellow"):
            for language in ("en", "ar"):
                bbox = {"width": 12, "height": 12}
                items.append({
                    "shape": shape, "color": color, "language": language,
                    "heldout_vs_truth": {
                        "mask_iou": 0.8,
                        "boundary_iou": 0.6,
                        "area_error_fraction": -0.1,
                        "centroid_dx": 0.5, "centroid_dy": -0.5,
                        "prediction": {
                            "foreground_pixels": 90, "bbox": bbox},
                        "reference": {
                            "foreground_pixels": 100, "bbox": bbox},
                    },
                    "sampled": {"foreground_pixels": 80},
                })
    return {
        "diagnostic": "shape-field-per-sample-geometry-detail",
        "source_run_dir": "/synthetic", "seed": 42,
        "no_training": True, "release_approved": False,
        "sampled_has_matching_truth": False, "items": items,
    }


class GeometryReviewTests(unittest.TestCase):
    def test_review_aggregates_without_modifying_report(self):
        report = sample_report()
        result = review(report)
        self.assertEqual(result["by_shape"]["square"]["samples"], 8)
        self.assertEqual(result["by_shape"]["triangle"]["heldout_iou_mean"], 0.8)
        self.assertEqual(result["by_shape"]["circle"]["sampled_to_heldout_area_ratio"], round(80/90, 6))
        self.assertEqual(result["by_shape"]["circle"]["heldout_bbox_width_error_mean"], 0)
        self.assertTrue(result["no_training"])
        self.assertFalse(result["release_approved"])
        self.assertEqual(report["items"][0]["heldout_vs_truth"]["mask_iou"], 0.8)

    def test_rejects_unmatched_and_fake_release_reports(self):
        report = sample_report()
        report["release_approved"] = True
        with self.assertRaises(ValueError):
            review(report)
        report["release_approved"] = False
        report["items"].pop()
        with self.assertRaises(ValueError):
            review(report)
        report = sample_report()
        report["items"][1]["language"] = "en"
        with self.assertRaises(ValueError):
            review(report)


if __name__ == "__main__":
    unittest.main()
