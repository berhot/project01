import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.pixel_dataset_v2 import CONCEPTS
from scripts.pixel_diagnose import audit_evaluation, analyze_directory


class PixelDiagnoseTests(unittest.TestCase):
    def make_fixture(self, root):
        """Synthetic fake report used ONLY to test the diagnostic parser."""
        run=root/"guidance_7p5_seed_42"
        run.mkdir(parents=True)
        items=[]
        for color,shape in CONCEPTS:
            for language in ("en","ar"):
                name=f"{color}_{shape}_{language}.png"
                Image.new("RGB",(32,32),(255,0,0)).save(run/name)
                items.append({
                    "image":name,"seed":42,"language":language,
                    "prompt":f"{color} {shape}",
                    "heuristic":{"color":"red","shape":"circle"},
                })
        (run/"report.json").write_text(json.dumps({
            "dataset_version":"v2","guidance_scale":7.5,
            "items":items,"metrics":{"prompt_accuracy":0.0833},
        }),encoding="utf-8")
        (root/"summary.json").write_text(json.dumps({
            "task":"sora-pixel-bilingual-guidance-comparison",
            "results":[{
                "seed":42,"guidance_scale":7.5,
                "report":str(run/"report.json"),
            }],
        }),encoding="utf-8")
        return run

    def test_detect_same_wrong_answers_for_both_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=self.make_fixture(root)
            report=audit_evaluation(run/"report.json")
            self.assertEqual(report["total_images"],24)
            self.assertEqual(report["distinct_heuristic_predictions"],1)
            self.assertEqual(report["dominant_prediction_fraction"],1.0)
            self.assertEqual(report["bilingual_pairs"],12)
            self.assertEqual(report["bilingual_agree_but_both_wrong"],11)
            self.assertEqual(report["bilingual_heuristic_agreement"],1.0)
            self.assertEqual(report["mean_bilingual_pixel_difference"],0.0)
            self.assertEqual(len(report["pairs"]),12)
            self.assertEqual(report["shape_accuracy_by_expected_shape"],{
                "circle":1.0,"square":0.0,"triangle":0.0,
            })
            self.assertEqual(report["prompt_accuracy_by_expected_shape"],{
                "circle":0.25,"square":0.0,"triangle":0.0,
            })
            self.assertEqual(report["prompt_accuracy_by_concept"]["red circle"],1.0)
            self.assertEqual(report["prompt_accuracy_by_concept"]["yellow triangle"],0.0)

    def test_root_analysis_rejects_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.make_fixture(root)
            output=root/"diagnosis.json"
            result=analyze_directory(root,output)
            self.assertEqual(len(result["results"]),1)
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                analyze_directory(root,output)

    def test_same_noise_shape_comparison_exposes_identical_squares(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=self.make_fixture(root)
            path=run/"report.json"
            report=json.loads(path.read_text(encoding="utf-8"))
            report["shared_noise_across_concepts"]=True
            path.write_text(json.dumps(report),encoding="utf-8")
            audit=audit_evaluation(path)
            self.assertTrue(audit["shared_noise_across_concepts"])
            self.assertEqual(audit["mean_square_triangle_mask_iou"],1.0)
            self.assertEqual(len(audit["square_triangle_same_noise_comparison"]),8)
            self.assertTrue(all(
                item["square_triangle_pixel_mae"]==0
                for item in audit["square_triangle_same_noise_comparison"]
            ))

    def test_missing_png_is_error_not_silent_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=self.make_fixture(root)
            (run/"red_circle_ar.png").unlink()
            with self.assertRaises(FileNotFoundError):
                audit_evaluation(run/"report.json")
