"""Regression checks for quick-mode protection and model input/output routing."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import features, models, pipeline


class PipelineSafetyTests(unittest.TestCase):
    def test_quick_reads_reference_tables_without_training_or_writes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = root / "reports" / "tables"
            tables.mkdir(parents=True)
            for filename in pipeline.RESULT_FILES.values():
                (tables / filename).write_text("published reference\n", encoding="utf-8")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir()}
            with patch.object(pipeline, "ROOT", root), \
                 patch.object(models, "main", side_effect=AssertionError("CV in quick mode")), \
                 patch.object(models, "run_final", side_effect=AssertionError("test evaluation in quick mode")):
                paths = pipeline.run_models({}, quick=True)
            self.assertTrue(all(p.parent == tables for p in paths.values()))
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir()}
            self.assertEqual(before, after)

    def test_full_routes_existing_model_api_to_generated_split_and_tables(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "full" / "features"
            outputs = root / "full" / "model_tables"
            previous = models.TABLES_DIR, models.load_split

            def cv(argv):
                self.assertEqual(argv, [])
                self.assertEqual(models.TABLES_DIR, outputs)
                models.load_split()
                return outputs / "cv_results.csv"

            def final(random_state):
                self.assertEqual(random_state, 42)
                self.assertEqual(models.TABLES_DIR, outputs)
                models.load_split()
                return {"test": outputs / "test_results.csv"}

            with patch.object(pipeline, "ROOT", root), \
                 patch.object(features, "load_split", return_value=()), \
                 patch.object(models, "main", side_effect=cv), \
                 patch.object(models, "run_final", side_effect=final):
                result = pipeline.run_models({"train": split / "train.parquet", "test": split / "test.parquet"})
                self.assertEqual(features.load_split.call_count, 2)
                features.load_split.assert_called_with(out_dir=split)
            self.assertEqual(result["cv"], outputs / "cv_results.csv")
            self.assertEqual((models.TABLES_DIR, models.load_split), previous)

    def test_failed_model_run_restores_module_paths(self):
        previous = models.TABLES_DIR, models.load_split
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with pipeline._model_paths(Path("temporary-split"), Path("temporary-tables")):
                raise RuntimeError("deliberate verification failure")
        self.assertEqual((models.TABLES_DIR, models.load_split), previous)


if __name__ == "__main__":
    unittest.main()
