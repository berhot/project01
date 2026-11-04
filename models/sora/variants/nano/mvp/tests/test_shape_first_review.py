"""Research gate compares by actual geometry, not unsupported DDPM seed equality."""
import copy
import unittest

from scripts.shape_first_review import compare_architectures


class ShapeFirstReviewTests(unittest.TestCase):
    def report(self,circles=4,squares=7,triangles=0,colors=1.0):
        by_shape={}
        rows=[]
        for shape,count in (("circle",circles),("square",squares),
                            ("triangle",triangles)):
            by_shape[shape]={
                "samples":8,"geometry_pass_count":count,
                "geometry_pass_rate":count/8,
                "color_accuracy":colors,
                "strict_prompt_pass_rate":count/8,
                "mean_ideal_mask_iou":0.9 if shape!="triangle" else 0.5,
            }
            for color in ("red","green","blue","yellow"):
                for lang in ("en","ar"):
                    rows.append({"expected_color":color,
                                 "expected_shape":shape,
                                 "language":lang,"seed":42})
        return {
            "sample_count":24,"by_shape":by_shape,
            "strict_prompt_accuracy":round((circles+squares+triangles)/24,4),
            "items":rows,
        }

    def test_failed_triangles_or_circle_regression_rejected(self):
        base=self.report()
        candidate=self.report(circles=3,squares=7,triangles=0)
        outcome=compare_architectures(base,candidate)
        self.assertFalse(outcome["research_gate_pass"])
        self.assertFalse(outcome["release_approved"])
        self.assertFalse(outcome["checks"]["triangle_geometry_visible"])
        self.assertFalse(outcome["checks"]["circle_geometry_not_regressed"])

    def test_strict_geometry_improvement_only_requires_human_review(self):
        base=self.report()
        candidate=self.report(circles=5,squares=7,triangles=4)
        outcome=compare_architectures(base,candidate)
        self.assertTrue(outcome["research_gate_pass"])
        self.assertTrue(outcome["human_visual_review_required"])
        self.assertFalse(outcome["release_approved"])

    def test_cross_architecture_compares_matched_concepts_not_seed_bytes(self):
        base=self.report()
        candidate=self.report(circles=6,squares=8,triangles=4)
        for item in candidate["items"]:
            item["seed"]=12345  # Different PRNG semantics for mask VAE vs DDPM
        outcome=compare_architectures(base,candidate)
        self.assertTrue(outcome["research_gate_pass"])
        self.assertIn("NOT equivalent",outcome["warning"])
        broken=copy.deepcopy(candidate)
        broken["items"][0]["expected_shape"]="circle"
        broken["items"][1]["expected_shape"]="triangle"
        with self.assertRaises(ValueError):
            compare_architectures(base,broken)
        with self.assertRaises(ValueError):
            compare_architectures(base,candidate,min_triangle_passes=0)
