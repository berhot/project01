"""Structured V3 language-interface tests. No image model changes."""
import unittest
from src.design_prompt_v3 import parse_design_prompt, UnsupportedDesignPrompt


class DesignPromptV3Tests(unittest.TestCase):
    def test_basic_english_square(self):
        result = parse_design_prompt("Create a red square")
        self.assertEqual(result.shape, "square")
        self.assertEqual(result.color, "red")
        self.assertEqual(result.corner_radius, 0.0)
        self.assertEqual(result.language, "en")

    def test_large_rounded_english(self):
        result = parse_design_prompt(
            "Create a red square with large rounded corners")
        self.assertEqual(result.corner_radius, 0.75)
        self.assertEqual(result.shape, "square")

    def test_large_rounded_arabic(self):
        result = parse_design_prompt("ارسم مربع أحمر بزوايا دائرية كبيرة")
        self.assertEqual(result.corner_radius, 0.75)
        self.assertEqual(result.color, "red")
        self.assertEqual(result.language, "ar")

    def test_circle_and_blue_color(self):
        self.assertEqual(parse_design_prompt("blue circle").shape, "circle")

    def test_reject_unsupported_modifiers_instead_of_ignoring(self):
        for prompt in (
            "red square with shiny metallic texture",
            "two red squares",
            "red square with a golden circle above",
            "red circle with rounded corners",
            "make something pretty",
            "",
        ):
            with self.subTest(prompt=prompt):
                with self.assertRaises(UnsupportedDesignPrompt):
                    parse_design_prompt(prompt)


if __name__ == "__main__":
    unittest.main()
