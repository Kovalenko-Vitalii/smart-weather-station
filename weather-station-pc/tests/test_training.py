import importlib.util
import unittest
from test_core import reading

@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Training dependencies not installed")
class TrainingTests(unittest.TestCase):
    def test_no_raw_sample_overlap(self):
        from train_model import make_examples, partition
        rows = [reading(i, 20 + i * 0.001, 1700000000 + i * 5) for i in range(1000)]
        train, validation, test = partition(make_examples(rows), len(rows))
        self.assertLess(max(e["target"] for e in train), min(e["start"] for e in validation))
        self.assertLess(max(e["target"] for e in validation), min(e["start"] for e in test))
        self.assertEqual(rows[train[0]["target"]]["observed_epoch"] - rows[train[0]["start"] + 4]["observed_epoch"], 60)

if __name__ == "__main__":
    unittest.main()