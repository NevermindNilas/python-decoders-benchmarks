import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.history import plotHistoryTrends


class HistoryTests(unittest.TestCase):
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
