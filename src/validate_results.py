"""Refuse to publish stale results, missing charts or a failed Nelux benchmark."""

import json
import os
from pathlib import Path


def validateResults(directory="."):
    directory = Path(directory)
    for resolution, dimensions in [("360p", (640, 360)), ("720p", (1280, 720))]:
        result = json.loads((directory / f"{resolution}_results.json").read_text())
        info = result["videoInfo"]
        if (info["width"], info["height"]) != dimensions:
            raise ValueError(f"{resolution}: wrong video dimensions")
        if result["runnerInfo"].get("runId") != os.environ["GITHUB_RUN_ID"]:
            raise ValueError(f"{resolution}: results are not from this Actions run")
        nelux = result["decoders"]["Nelux"]
        if "error" in nelux or nelux.get("successfulRuns") != result["config"]["runs"] or nelux.get("fps", 0) <= 0:
            raise ValueError(f"{resolution}: Nelux must succeed in every timed iteration: {nelux}")
        if not info.get("frameCount") or nelux["frameCount"] != info["frameCount"]:
            raise ValueError(f"{resolution}: incomplete Nelux decode")
        history = json.loads((directory / "history" / f"{resolution}_history.json").read_text())
        record = history["runs"][-1]
        if record.get("runId") != os.environ["GITHUB_RUN_ID"] or record["decoders"]["Nelux"]["fps"] != nelux["fps"]:
            raise ValueError(f"{resolution}: history does not match current measurements")
        for path in [directory / f"{resolution}_diagram.png", directory / "history" / f"{resolution}_trend.png"]:
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing chart: {path}")
    print("Both resolutions have current measurements, complete Nelux results and charts.")


if __name__ == "__main__":
    validateResults()
