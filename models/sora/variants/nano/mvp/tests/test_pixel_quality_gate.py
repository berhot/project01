"""Unit tests for conservative score-gating of comparable image reports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pixel_quality_gate import compare_reports


class PixelQualityGateTests(unittest.TestCase):
    def make_reports(self,root,seed=42,candidate_seed=42):
        paths=[]
        for name,s in (("baseline",seed),("candidate",candidate_seed)):
            path=Path(root)/f"{name}.json"
            items=[
                {"image":f"{color}_{shape}_{lang}.png",
                 "language":lang,"seed":s}
                for shape in ("circle","square","triangle")
                for color in ("red","green","blue","yellow")
                for lang in ("en","ar")
            ]
            path.write_text(json.dumps({
                "dataset_version":"v2","sampler":"posterior_mean",
                "guidance_scale":5.0,
                "shared_noise_across_concepts":False,
                "items":items,
                "metrics":{
                    "prompt_accuracy":0.6667 if name=="baseline" else 0.2917,
                    "color_accuracy":1.0 if name=="baseline" else 0.625,
                },
            }),encoding="utf-8")
            paths.append(path)
        return paths

    def test_rejects_regression_despite_strong_shape_probe(self):
        with tempfile.TemporaryDirectory() as d:
            before,after=self.make_reports(d)
            reports=[
                {"shape_accuracy_by_expected_shape":{
                    "circle":1.0,"square":1.0,"triangle":0.0}},
                {"shape_accuracy_by_expected_shape":{
                    "circle":0.0,"square":1.0,"triangle":0.0}},
            ]
            with patch("scripts.pixel_quality_gate.audit_evaluation",
                       side_effect=reports):
                gate=compare_reports(before,after)
            self.assertFalse(gate["accepted_by_toy_gate"])
            self.assertTrue(gate["human_visual_review_required"])
            self.assertFalse(gate["checks"]["circles_preserved"])
            self.assertFalse(gate["checks"]["triangle_recognizable_heuristic"])
            self.assertFalse(gate["checks"]["colors_preserved"])

    def test_accepts_only_when_all_old_classes_and_triangles_hold(self):
        with tempfile.TemporaryDirectory() as d:
            before,after=self.make_reports(d)
            candidate=json.loads(after.read_text())
            candidate["metrics"]["prompt_accuracy"]=0.75
            candidate["metrics"]["color_accuracy"]=1.0
            after.write_text(json.dumps(candidate))
            reports=[
                {"shape_accuracy_by_expected_shape":{
                    "circle":1.0,"square":1.0,"triangle":0.0}},
                {"shape_accuracy_by_expected_shape":{
                    "circle":1.0,"square":1.0,"triangle":0.25}},
            ]
            with patch("scripts.pixel_quality_gate.audit_evaluation",
                       side_effect=reports):
                gate=compare_reports(before,after)
            self.assertTrue(gate["accepted_by_toy_gate"])
            self.assertTrue(gate["human_visual_review_required"])

    def test_legacy_missing_shared_noise_defaults_false(self):
        """Original v2 reports predate --shared-noise and used per-prompt seeds."""
        with tempfile.TemporaryDirectory() as d:
            before,after=self.make_reports(d)
            baseline=json.loads(before.read_text())
            baseline.pop("shared_noise_across_concepts")
            before.write_text(json.dumps(baseline))
            reports=[
                {"shape_accuracy_by_expected_shape":{
                    "circle":1.0,"square":1.0,"triangle":0.0}},
                {"shape_accuracy_by_expected_shape":{
                    "circle":1.0,"square":1.0,"triangle":0.0}},
            ]
            with patch("scripts.pixel_quality_gate.audit_evaluation",
                       side_effect=reports):
                result=compare_reports(before,after)
            self.assertFalse(result["accepted_by_toy_gate"])
            self.assertTrue(result["checks"]["circles_preserved"])
            self.assertTrue(result["checks"]["squares_preserved"])
            self.assertFalse(result["checks"]["triangle_recognizable_heuristic"])

    def test_legacy_missing_shared_noise_rejects_explicit_shared_noise(self):
        with tempfile.TemporaryDirectory() as d:
            before,after=self.make_reports(d)
            baseline=json.loads(before.read_text())
            baseline.pop("shared_noise_across_concepts")
            before.write_text(json.dumps(baseline))
            candidate=json.loads(after.read_text())
            candidate["shared_noise_across_concepts"]=True
            after.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError,"shared_noise_across_concepts"):
                compare_reports(before,after)

    def test_rejects_invalid_shared_noise_field_type(self):
        with tempfile.TemporaryDirectory() as d:
            before,after=self.make_reports(d)
            candidate=json.loads(after.read_text())
            candidate["shared_noise_across_concepts"]="false"
            after.write_text(json.dumps(candidate))
            with self.assertRaisesRegex(ValueError,"shared_noise_across_concepts"):
                compare_reports(before,after)

    def test_refuses_unmatched_seeds_before_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            before,after=self.make_reports(d,candidate_seed=123)
            with self.assertRaises(ValueError):
                compare_reports(before,after)
