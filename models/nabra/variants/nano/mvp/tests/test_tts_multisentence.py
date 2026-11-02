import unittest
from src.tts.multisentence import sample_index


class MultiSentenceTests(unittest.TestCase):
    def test_epoch_covers_each_recording_once(self):
        for count in (1, 2, 7):
            for epoch in range(3):
                ids = [sample_index(epoch * count + position + 1, count, 42)
                       for position in range(count)]
                self.assertEqual(sorted(ids), list(range(count)))

    def test_repeatable_after_resume(self):
        first = [sample_index(step, 9, 2026) for step in range(1, 31)]
        later = [sample_index(step, 9, 2026) for step in range(11, 31)]
        self.assertEqual(first[10:], later)

    def test_invalid_args(self):
        for step, count in ((0, 3), (1, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                sample_index(step, count, 42)


if __name__ == "__main__":
    unittest.main()
