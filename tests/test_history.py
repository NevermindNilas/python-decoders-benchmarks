import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.history import plotHistoryTrends, appendHistory


class HistoryTests(unittest.TestCase):
    def test_rerun_replaces_same_run_without_removing_other_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.json")
            result = {"runnerInfo": {"runId": "1", "runAttempt": "1"}, "decoders": {"Nelux": {"fps": 100, "version": "0.19.0"}}}
            appendHistory(result, path, "720p")
            result["runnerInfo"]["runId"] = "2"
            appendHistory(result, path, "720p")
            result["runnerInfo"]["runAttempt"] = "2"
            appendHistory(result, path, "720p")
            runs = json.loads(Path(path).read_text())["runs"]
            self.assertEqual([r["runId"] for r in runs], ["1", "2"])
            self.assertEqual(runs[-1]["runAttempt"], "2")
            self.assertEqual(runs[-1]["decoders"]["Nelux"]["version"], "0.19.0")

    def test_failure_is_a_gap_and_cpu_cohorts_remain_separate(self):
        runs = [
            {"timestampUtc": f"2026-09-{day:02d}T00:00:00Z", "runner": "stable",
             "systemInfo": {"cpu": {"model": cpu, "logicalCores": 4}},
             "decoders": {"Nelux": data, "Unavailable": {"error": "missing"}}}
            for day, cpu, data in [(1, "CPU A", {"fps": 100}), (2, "CPU A", {"error": "broken"}),
                                    (3, "CPU A", {"fps": 120}), (4, "CPU B", {"fps": 80})]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps({"runs": runs}))
            with patch("matplotlib.figure.Figure.savefig"), patch("src.history.plt.close"):
                plotHistoryTrends(str(path), str(path.with_suffix(".png")), "720p")
                figure = plt.gcf()
                self.assertEqual(len(figure.axes), 2)
                points = figure.axes[0].lines[0].get_ydata()
                self.assertEqual(points[0], 100)
                self.assertTrue(math.isnan(points[1]))
                self.assertEqual(points[2], 120)
                self.assertIn("Unavailable (unavailable)", figure.axes[0].get_legend_handles_labels()[1])
            plt.close("all")

    def test_hosted_runs_form_one_time_series(self):
        runs = [
            {
                "timestampUtc": f"2026-09-{day:02d}T00:00:00Z",
                "runner": f"GitHub Actions {1000 + day}",
                "isCi": True,
                "os": "Linux 6.17.0-azure",
                "decoders": {"PyAV": {"fpsMedian": fps}},
            }
            for day, fps in [(1, 100), (15, 120)]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps({"runs": runs}))
            with patch("matplotlib.figure.Figure.savefig") as save:
                with patch("src.history.plt.close"):
                    plotHistoryTrends(str(path), str(path.with_suffix(".png")), "720p")
                    figure = plt.gcf()
                    self.assertEqual(len(figure.axes), 1, "Ephemeral runner names must not create duplicate panels")
                    self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [100, 120])
                    save.assert_called_once()
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
