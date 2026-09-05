"""Unit tests for src/config.py -- guards against path constants drifting
out of sync with the directory layout the scripts assume."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


class TestPathConstants(unittest.TestCase):
    def test_data_dirs_live_under_root(self) -> None:
        for path in [config.DATA_RAW_DIR, config.DATA_PROCESSED_DIR, config.RESULTS_DIR]:
            self.assertTrue(path.is_relative_to(config.ROOT_DIR), f"{path} escapes ROOT_DIR")

    def test_artifact_paths_sit_in_processed_dir(self) -> None:
        self.assertEqual(config.FAISS_INDEX_PATH.parent, config.DATA_PROCESSED_DIR)
        self.assertEqual(config.CHUNKS_PATH.parent, config.DATA_PROCESSED_DIR)

    def test_manifest_path_sits_in_raw_dir(self) -> None:
        self.assertEqual(config.MANIFEST_PATH.parent, config.DATA_RAW_DIR)
        self.assertEqual(config.MANIFEST_PATH.name, "manifest.json")

    def test_eval_set_path_exists_and_is_committed(self) -> None:
        # The eval set is curated, committed data (unlike the regenerable
        # index artifacts), so a missing file here is a real repo problem.
        self.assertTrue(config.EVAL_SET_PATH.exists(), f"missing {config.EVAL_SET_PATH}")

    def test_embedding_model_name_is_pinned(self) -> None:
        self.assertEqual(config.EMBEDDING_MODEL_NAME, "BAAI/bge-small-en-v1.5")


if __name__ == "__main__":
    unittest.main()
