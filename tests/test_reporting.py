import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.reporting import createPerformanceDiagram, generate_frame_count_markdown


class ReportingTests(unittest.TestCase):
    def test_chart_and_table_include_failed_nelux_without_fake_fps(self):
        results = {"decoders": {"PyAV": {"fps": 100, "fpsMedian": 110, "fpsMin": 90, "fpsMax": 120},
                                 "Nelux": {"error": "undefined symbol", "fps": 0}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.md"
            generate_frame_count_markdown(results, path)
            self.assertIn("| Nelux | unknown | N/A | 0 | undefined symbol |", path.read_text())
            self.assertIn("110.00", path.read_text())
            with patch("src.reporting.plt.close"):
                createPerformanceDiagram(results, str(path.with_suffix(".png")))
                ax = plt.gcf().axes[0]
                self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["PyAV", "Nelux"])
                self.assertEqual([bar.get_width() for bar in ax.patches], [110])
            plt.close("all")

    def test_chart_failure_propagates_instead_of_publishing_old_image(self):
        with patch("matplotlib.figure.Figure.savefig", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                createPerformanceDiagram({"decoders": {"Nelux": {"fps": 20}}}, "unused.png")

