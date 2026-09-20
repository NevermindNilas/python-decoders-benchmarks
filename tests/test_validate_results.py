import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.validate_results import validateResults


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "history").mkdir()
        for resolution, width, height in [("360p", 640, 360), ("720p", 1280, 720)]:
            result = {"videoInfo": {"width": width, "height": height, "frameCount": 300},
                      "runnerInfo": {"runId": "123"}, "config": {"runs": 3},
                      "decoders": {"Nelux": {"fps": 100, "successfulRuns": 3, "frameCount": 300}}}
            (self.root / f"{resolution}_results.json").write_text(json.dumps(result))
            (self.root / "history" / f"{resolution}_history.json").write_text(json.dumps({"runs": [{"runId": "123", "decoders": result["decoders"]}]}))
            (self.root / f"{resolution}_diagram.png").write_bytes(b"chart")
            (self.root / "history" / f"{resolution}_trend.png").write_bytes(b"chart")

    @patch.dict("os.environ", {"GITHUB_RUN_ID": "123"})
    def test_accepts_complete_current_results(self):
        validateResults(self.root)

    @patch.dict("os.environ", {"GITHUB_RUN_ID": "456"})
    def test_rejects_stale_results(self):
        with self.assertRaisesRegex(ValueError, "not from this Actions run"):
            validateResults(self.root)

    @patch.dict("os.environ", {"GITHUB_RUN_ID": "123"})
    def test_rejects_failed_nelux_and_missing_chart(self):
        path = self.root / "360p_results.json"
        result = json.loads(path.read_text())
        result["decoders"]["Nelux"]["error"] = "native runtime unavailable"
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "Nelux must succeed"):
            validateResults(self.root)
        del result["decoders"]["Nelux"]["error"]
        path.write_text(json.dumps(result))
        (self.root / "360p_diagram.png").unlink()
        with self.assertRaisesRegex(ValueError, "Missing chart"):
            validateResults(self.root)
