"""Persist benchmark runs over time and plot per-library trends.

History layout: one JSON file per resolution under `history/`, each containing
a list of run records:

    {
      "resolution": "1280x720",
      "runs": [
        {
          "timestampUtc": "2026-04-26T12:34:56Z",
          "runner": "ci-fv-az...",
          "commit": "34867a6",
          "isCi": true,
          "config": {"runs": 3, "warmup": 1},
          "decoders": {
            "PyAV": {"fpsMedian": 123.4, "fpsStd": 1.2, "frameCount": 15691, ...},
            ...
          }
        }
      ]
    }

The trend chart slices that file by runner so a single library's evolution
is comparable only against runs from the same hardware.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime


_HISTORY_KEYS_TO_KEEP = (
    "fps",
    "fpsMedian",
    "fpsMean",
    "fpsStd",
    "fpsMin",
    "fpsMax",
    "fpsCv",
    "frameCount",
    "elapsedTime",
    "runs",
    "successfulRuns",
    "error",
)


def _trimDecoderResult(decoderResult: dict[str, Any]) -> dict[str, Any]:
    """Drop per-iteration arrays before persisting — keep history files small."""
    return {k: decoderResult[k] for k in _HISTORY_KEYS_TO_KEEP if k in decoderResult}


def appendHistory(results: dict[str, Any], historyPath: str, resolution: str) -> None:
    os.makedirs(os.path.dirname(historyPath), exist_ok=True)

    if os.path.isfile(historyPath):
        with open(historyPath, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {"resolution": resolution, "runs": []}

    runnerInfo = results.get("runnerInfo", {})
    record = {
        "timestampUtc": runnerInfo.get("timestampUtc"),
        "runner": runnerInfo.get("runner", "unknown"),
        "commit": runnerInfo.get("commit", ""),
        "isCi": runnerInfo.get("isCi", False),
        "os": runnerInfo.get("os", ""),
        "python": runnerInfo.get("python", ""),
        "config": results.get("config", {}),
        "videoInfo": results.get("videoInfo", {}),
        "decoders": {
            name: _trimDecoderResult(data)
            for name, data in results.get("decoders", {}).items()
        },
    }

    history["runs"].append(record)

    with open(historyPath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"History appended to {historyPath} (now {len(history['runs'])} runs)")


def plotHistoryTrends(historyPath: str, outputPath: str, resolution: str) -> None:
    """Plot fps-over-time per (runner, library) for the given resolution."""
    if not os.path.isfile(historyPath):
        print(f"History file {historyPath} missing — skipping trend plot.")
        return

    with open(historyPath, "r", encoding="utf-8") as f:
        history = json.load(f)

    runs = history.get("runs", [])
    if not runs:
        print("No history runs yet — skipping trend plot.")
        return

    # series[(runner, library)] -> list[(datetime, fps)]
    series: dict[tuple[str, str], list[tuple[datetime, float]]] = defaultdict(list)
    for run in runs:
        ts = run.get("timestampUtc")
        if not ts:
            continue
        try:
            when = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        runner = run.get("runner", "unknown")
        for lib, data in run.get("decoders", {}).items():
            fps = data.get("fpsMedian", data.get("fps", 0))
            if "error" in data or not fps:
                continue
            series[(runner, lib)].append((when, fps))

    if not series:
        print("No plottable points in history — skipping trend plot.")
        return

    # One subplot per runner so libraries on different hardware aren't mixed.
    runners = sorted({runner for runner, _ in series.keys()})
    fig, axes = plt.subplots(
        len(runners), 1, figsize=(14, 5 * len(runners)), squeeze=False
    )

    for axIdx, runner in enumerate(runners):
        ax = axes[axIdx][0]
        runnerSeries = {
            lib: pts for (r, lib), pts in series.items() if r == runner
        }
        for lib in sorted(runnerSeries.keys()):
            pts = sorted(runnerSeries[lib])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=lib, linewidth=1.5, markersize=4)

        ax.set_title(f"{resolution} — runner: {runner}")
        ax.set_ylabel("FPS (median)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(f"Decoder performance over time ({resolution})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(outputPath) or ".", exist_ok=True)
    fig.savefig(outputPath, dpi=180)
    plt.close(fig)
    print(f"Trend chart saved to {outputPath}")
