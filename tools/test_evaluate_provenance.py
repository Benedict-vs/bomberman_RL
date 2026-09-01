import hashlib
import unittest
from pathlib import Path

from tools.evaluate import REPO_ROOT, collect_agent_provenance


class EvaluationProvenanceTest(unittest.TestCase):
    def test_ben_task2_records_exact_model_and_callback_hashes(self):
        provenance = collect_agent_provenance(["ben_task2"])["ben_task2"]
        model_path = (
            REPO_ROOT
            / "agent_code"
            / "ben_task2"
            / provenance["model_file"]
        )

        self.assertTrue(provenance["model_exists"])
        self.assertEqual(
            provenance["escape_feature_mode"],
            "reachable_safe_tiles",
        )
        self.assertEqual(
            provenance["model_sha256"],
            hashlib.sha256(model_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(provenance["callbacks_sha256"]), 64)

    def test_ben_task3_records_the_actual_fallback_model(self):
        from agent_code.ben_task3 import callbacks

        provenance = collect_agent_provenance(["ben_task3"])["ben_task3"]
        model_path = (
            REPO_ROOT
            / "agent_code"
            / "ben_task3"
            / provenance["model_file"]
        )

        self.assertEqual(
            provenance["model_file"],
            callbacks.INFERENCE_MODEL_FILE,
        )
        if callbacks.INFERENCE_MODEL_FILE != callbacks.MODEL_FILE:
            self.assertEqual(
                provenance["configured_model_file"],
                callbacks.MODEL_FILE,
            )
        else:
            self.assertNotIn("configured_model_file", provenance)
        self.assertTrue(provenance["model_exists"])
        self.assertEqual(
            provenance["model_sha256"],
            hashlib.sha256(model_path.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
