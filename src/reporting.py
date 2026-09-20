"""Charts and release tables built from the same current-run measurements."""

import math
from textwrap import shorten

import matplotlib.pyplot as plt


def validResult(data):
    fps = data.get("fpsMedian", data.get("fps", 0))
    return "error" not in data and isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0


def createPerformanceDiagram(results, outputPath):
    decoders = results["decoders"]
    names = sorted(decoders, key=lambda name: (
        not validResult(decoders[name]),
        -decoders[name].get("fpsMedian", decoders[name].get("fps", 0)) if validResult(decoders[name]) else 0,
        name,
    ))
    if not names:
        raise ValueError("No decoder results to plot")
    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.45 + 2)))
    try:
        for index, name in enumerate(names):
            data = decoders[name]
            if validResult(data):
                fps = data.get("fpsMedian", data.get("fps", 0))
                low, high = data.get("fpsMin", fps), data.get("fpsMax", fps)
                ax.barh(index, fps, color="#3498db", xerr=[[max(0, fps-low)], [max(0, high-fps)]], capsize=3)
                ax.annotate(f"{fps:.1f} FPS", (max(high, fps), index), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
            else:
                reason = shorten(str(data.get("error", "No valid measurements")), width=105, placeholder="…")
                ax.text(0, index, f"Unavailable: {reason}", color="#a33", va="center", fontsize=9)
        ax.set_yticks(range(len(names)), names)
        # Failed rows have text but no bar to extend Matplotlib's data limits.
        ax.set_ylim(len(names) - 0.5, -0.5)
        maxFps = max((max(data.get("fpsMax", 0), data.get("fpsMedian", data.get("fps", 0)))
                      for data in decoders.values() if validResult(data)), default=1)
        ax.set_xlim(0, maxFps * 1.2)
        ax.set_xlabel("Median FPS (whiskers: min–max timed iterations)")
        info = results.get("videoInfo", {})
        runner = results.get("runnerInfo", {})
        ax.set_title(f"Decoder performance — {info.get('width', '?')}×{info.get('height', '?')}\n{runner.get('timestampUtc', '')} · {runner.get('runner', 'unknown runner')}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outputPath, dpi=180)
    finally:
        plt.close(fig)


def generate_frame_count_markdown(results, outputPath):
    """Include every decoder, its FPS/version and failures in published lists."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    expected = results.get("videoInfo", {}).get("frameCount")
    lines = ["| Decoder | Version | Median FPS | Frames | Status |",
             "|---|---|---:|---:|---|"]
    for name, data in results.get("decoders", {}).items():
        fps = f"{data.get('fpsMedian', data.get('fps', 0)):.2f}" if validResult(data) else "N/A"
        status = data.get("error", "OK" if validResult(data) else "No valid measurements")
        if validResult(data) and expected and data.get("frameCount") != expected:
            status = f"Frame count mismatch (expected {expected})"
        elif validResult(data) and data.get("successfulRuns", 1) < data.get("runs", 1):
            status = f"Partial: {data['successfulRuns']}/{data['runs']} timed runs succeeded"
        lines.append(f"| {cell(name)} | {cell(data.get('version', 'unknown'))} | {fps} | {data.get('frameCount', 0)} | {cell(status)} |")
    with open(outputPath, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
