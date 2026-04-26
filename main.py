import os
import json
import signal
import statistics
import subprocess
import time
import urllib.request
import traceback
import platform
import psutil
import cv2
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

# src imports
from src.backends.ffmpeg import decodeWithFFMPEG_RGB24
from src.backends.imageio import decodeWithImageioFFMPEG
from src.backends.opencv import decodeWithOpenCV
from src.backends.pyav import decodeWithPyAV
from src.backends.torchaudio import decodeWithTorchaudio
from src.backends.ffmpegcv import decodeWithFFMPEGCV_Block, decodeWithFFMPEGCV_NoBlock
from src.backends.decord import decodeWithDecord
from src.backends.deffcode import decodeWithDeffcode
from src.backends.maxTheoretical import (
    decodeWithMaxTheoreticalRGB,
    decodeWithMaxTheoreticalYUV420,
)
from src.backends.videoreaderrs import (
    decodeWithVideoReaderRS,
    decodeWithVideoReaderRSFast,
)
from src.backends.torchcodec import decodeWithTorchCodec
from src.backends.ffmpegpython import decodeWithFFmpegPython
from src.backends.colorConversionAnalysis import (
    analyzeColorConversionDifferences,
    analyzeVideoColorProperties,
)
from src.backends.celuxdecoder import decodeWithCeLux
from src.coloredPrints import lightcyan


def absolutePath(path: str) -> str:
    if path is not None and path != "":
        return os.path.abspath(os.path.expanduser(str(path)))
    return path


def downloadVideo(url: str, outputPath: str) -> str:
    """Download a video from the specified URL to the output path."""
    print(f"Downloading video from {url}...")

    if not os.path.exists(os.path.dirname(outputPath)):
        os.makedirs(os.path.dirname(outputPath))

    # googleapis bucket rejects default urllib UA with HTTP 403
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 python-decoders-benchmarks"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, open(
        outputPath, "wb"
    ) as outFile:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            outFile.write(chunk)

    print(f"Video downloaded to {outputPath}")

    return outputPath


def getRunnerInfo() -> dict[str, Any]:
    """Stable identifier of the runner so history per-runner can be tracked."""
    runner = os.environ.get("RUNNER_NAME") or os.environ.get("BENCHMARK_RUNNER", "")
    isCi = bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"))

    if not runner:
        runner = ("ci-" if isCi else "local-") + platform.node()

    commit = (
        os.environ.get("GITHUB_SHA")
        or os.environ.get("BENCHMARK_COMMIT")
        or _gitShortSha()
    )

    return {
        "runner": runner,
        "isCi": isCi,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "commit": commit,
        "timestampUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _gitShortSha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return ""


def getSystemInfo() -> dict[str, Any]:
    """Get system information including CPU and RAM."""

    try:
        cpuInfo = {
            "model": platform.processor(),
            "physicalCores": psutil.cpu_count(logical=False),
            "logicalCores": psutil.cpu_count(logical=True),
            "frequencyMHz": psutil.cpu_freq().current
            if psutil.cpu_freq()
            else "Unknown",
        }

        ramInfo = {
            "totalGB": round(psutil.virtual_memory().total / (1024**3), 2),
            "availableGB": round(psutil.virtual_memory().available / (1024**3), 2),
        }

        return {
            "cpu": cpuInfo,
            "ram": ramInfo,
        }
    except Exception as e:
        print(f"Error getting system info: {str(e)}")
        return {"error": str(e)}


@dataclass
class Decoder:
    name: str
    decoder: Callable[[str], dict[str, Any]] | Callable[[str, Any], dict[str, Any]]
    cooling: float | int = 3
    videoInfo: Any | None = None


def _aggregateIterations(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-iteration decoder runs to summary stats.

    Each iteration carries fps / frameCount / elapsedTime / optional error.
    Successful iterations contribute to fps stats; if every iteration errored
    we keep the first error so the consumer still sees a failure.
    """
    successful = [it for it in iterations if "error" not in it and it.get("fps", 0) > 0]
    fpsValues = [it["fps"] for it in successful]
    frameCounts = [it.get("frameCount", 0) for it in successful]

    if not successful:
        firstError = next((it["error"] for it in iterations if "error" in it), "no successful runs")
        return {
            "error": firstError,
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
            "iterations": iterations,
            "runs": len(iterations),
            "successfulRuns": 0,
        }

    medianFps = statistics.median(fpsValues)
    meanFps = statistics.fmean(fpsValues)
    stdFps = statistics.pstdev(fpsValues) if len(fpsValues) > 1 else 0.0
    cv = (stdFps / meanFps) if meanFps > 0 else 0.0

    return {
        "frameCount": frameCounts[0] if frameCounts else 0,
        "elapsedTime": statistics.fmean([it["elapsedTime"] for it in successful]),
        # `fps` is the headline number used by plots & older history rows: use median
        "fps": medianFps,
        "fpsMedian": medianFps,
        "fpsMean": meanFps,
        "fpsStd": stdFps,
        "fpsMin": min(fpsValues),
        "fpsMax": max(fpsValues),
        "fpsCv": cv,
        "runs": len(iterations),
        "successfulRuns": len(successful),
        "iterations": iterations,
    }


def _runDecoderIteration(decoder: "Decoder", videoPath: str) -> dict[str, Any]:
    decoderFct = decoder.decoder
    if decoderFct.__code__.co_argcount == 1:
        return decoderFct(videoPath)
    return decoderFct(videoPath, decoder.videoInfo)


def runBenchmark(
    videoPath: str,
    coolingPeriod: int = 3,
    systemInfo: dict = {},
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Run benchmark on all decoders and return the results.

    Args:
        videoPath: Path to the video file to benchmark
        coolingPeriod: Time in seconds to wait between decoder tests
        runs: Number of timed iterations per decoder (median is reported)
        warmup: Untimed warmup iterations to prime the OS page cache and lazy
            initialisation paths inside each library
    """
    print("Getting video information...")
    videoInfo = getVideoInfo(videoPath)

    decoders: list[Decoder] = [
        Decoder(name="Nelux", decoder=decodeWithCeLux, cooling=coolingPeriod),
        Decoder(name="PyAV", decoder=decodeWithPyAV, cooling=coolingPeriod),
        Decoder(name="OpenCV", decoder=decodeWithOpenCV, cooling=coolingPeriod),
        Decoder(name="torchaudio", decoder=decodeWithTorchaudio, cooling=coolingPeriod),
        Decoder(name="TorchCodec", decoder=decodeWithTorchCodec, cooling=coolingPeriod),
        Decoder(name="Decord", decoder=decodeWithDecord, cooling=coolingPeriod),
        Decoder(
            name="VideoReaderRS", decoder=decodeWithVideoReaderRS, cooling=coolingPeriod
        ),
        Decoder(
            name="VideoReaderRS YUV420toRGB",
            decoder=decodeWithVideoReaderRSFast,
            cooling=coolingPeriod,
        ),
        Decoder(
            name="FFmpeg-Subprocess",
            decoder=decodeWithFFMPEG_RGB24,
            cooling=coolingPeriod,
            videoInfo=videoInfo,
        ),
        Decoder(
            name="FFMPEGCV (Block)",
            decoder=decodeWithFFMPEGCV_Block,
            cooling=coolingPeriod,
        ),
        Decoder(
            name="FFmpegCV-NoBlock",
            decoder=decodeWithFFMPEGCV_NoBlock,
            cooling=coolingPeriod,
        ),
        Decoder(
            name="Imageio-ffmpeg",
            decoder=decodeWithImageioFFMPEG,
            cooling=coolingPeriod,
        ),
        Decoder(
            name="FFmpeg-python",
            decoder=decodeWithFFmpegPython,
            cooling=coolingPeriod,
            videoInfo=videoInfo,
        ),
        Decoder(name="Deffcode", decoder=decodeWithDeffcode, cooling=coolingPeriod),
        Decoder(
            name="Max Theoretical RGB24",
            decoder=decodeWithMaxTheoreticalRGB,
            cooling=coolingPeriod,
        ),
        Decoder(
            name="Max Theoretical YUV420",
            decoder=decodeWithMaxTheoreticalYUV420,
            cooling=coolingPeriod,
        ),
    ]

    decodingResults: dict[str, Any] = {}
    for i, decoder in enumerate(decoders):
        print(
            lightcyan(
                f"\n({i + 1}/{len(decoders)}) Running {decoder.name} decoder "
                f"[{warmup} warmup + {runs} timed]..."
            )
        )

        for w in range(warmup):
            try:
                _runDecoderIteration(decoder, videoPath)
            except Exception as warmupErr:
                # Warmup failures are expected for unsupported backends; the timed
                # runs below will record the same error in the aggregated result.
                print(f"  warmup {w + 1}/{warmup} failed: {warmupErr}")
            time.sleep(min(1, decoder.cooling))

        iterations: list[dict[str, Any]] = []
        for r in range(runs):
            try:
                result = _runDecoderIteration(decoder, videoPath)
            except Exception as runErr:
                result = {
                    "error": str(runErr),
                    "frameCount": 0,
                    "elapsedTime": 0,
                    "fps": 0,
                }
            iterations.append(result)
            print(
                f"  run {r + 1}/{runs}: "
                f"{result.get('fps', 0):.2f} fps "
                f"({result.get('frameCount', 0)} frames)"
                + (f" ERROR: {result['error']}" if "error" in result else "")
            )
            if r < runs - 1:
                time.sleep(decoder.cooling)

        decodingResults[decoder.name] = _aggregateIterations(iterations)
        time.sleep(decoder.cooling)

    print(lightcyan("\nBenchmark completed."))

    results = {
        "videoPath": videoPath,
        "videoInfo": videoInfo,
        "systemInfo": systemInfo,
        "runnerInfo": getRunnerInfo(),
        "config": {"runs": runs, "warmup": warmup, "coolingPeriod": coolingPeriod},
        "decoders": decodingResults,
    }

    return results


def getVideoInfo(videoPath: str) -> dict[str, Any]:
    """Get video information using OpenCV."""
    try:
        cap = cv2.VideoCapture(videoPath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        durationSeconds = frameCount / fps if fps > 0 else 0

        cap.release()

        return {
            "width": width,
            "height": height,
            "frameCount": frameCount,
            "fps": fps,
            "durationSeconds": durationSeconds,
        }
    except Exception as e:
        print(f"Error getting video info: {str(e)}")
        return {
            "error": str(e),
            "width": 0,
            "height": 0,
            "frameCount": 0,
            "fps": 0,
            "durationSeconds": 0,
        }


def saveResults(results: dict[str, Any], outputPath: str) -> None:
    """Save benchmark results to a JSON file."""
    with open(outputPath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {outputPath}")


def createPerformanceDiagram(results: dict[str, Any], outputPath: str) -> None:
    """Create a bar chart of decoder performance and save it to PNG."""
    try:
        decoderNames = []
        fpsValues = []

        for decoderName, data in results["decoders"].items():
            if "error" not in data and data.get("fps", 0) > 0:
                decoderNames.append(decoderName)
                fpsValues.append(data["fps"])
            elif "error" in data:
                print(
                    f"Skipping {decoderName} in diagram due to error: {data['error']}"
                )
            else:
                print(f"Skipping {decoderName} in diagram due to zero FPS.")

        if not decoderNames:
            print(lightcyan("No valid decoder results to plot."))
            return

        plt.figure(figsize=(14, 8))
        bars = plt.bar(
            decoderNames,
            fpsValues,
            width=0.3,
            color=[
                "#3498db",
                "#2ecc71",
                "#e74c3c",
                "#f39c12",
                "#9b59b6",
                "#1abc9c",
                "#34495e",
                "#f1c40f",
                "#e67e22",
                "#95a5a6",
                "#7f8c8d",
                "#d35400",
                "#c0392b",
            ][: len(decoderNames)],
        )

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(1, height * 0.02),
                f"{height:.1f} fps",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.xlabel("Decoders", fontsize=12)
        plt.ylabel("Performance (FPS)", fontsize=12)
        plt.title("Video Decoders Performance Comparison", fontsize=16)
        plt.ylim(0, max(fpsValues) * 1.1)

        plt.xticks(rotation=45, ha="right")

        if "systemInfo" in results and "error" not in results["systemInfo"]:
            systemInfo = results["systemInfo"]
            cpuInfo = systemInfo.get("cpu", {})
            ramInfo = systemInfo.get("ram", {})
            systemText = (
                f"CPU: {cpuInfo.get('model', 'N/A')} ({cpuInfo.get('logicalCores', '?')} cores)\n"
                f"RAM: {ramInfo.get('totalGB', '?')} GB total"
            )
            plt.figtext(0.02, 0.02, systemText, fontsize=10, va="bottom", ha="left")

        if "videoInfo" in results and "error" not in results["videoInfo"]:
            videoInfo = results["videoInfo"]
            videoText = (
                f"Video: {os.path.basename(results.get('videoPath', 'N/A'))}\n"
                f"Resolution: {videoInfo.get('width', '?')}x{videoInfo.get('height', '?')}\n"
                f"Duration: {videoInfo.get('durationSeconds', 0):.1f}s ({videoInfo.get('frameCount', '?')} frames)"
            )
            plt.figtext(0.98, 0.98, videoText, fontsize=10, ha="right", va="top")

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
        plt.savefig(outputPath, dpi=300)
        print(f"Performance diagram saved to {outputPath}")

    except Exception as e:
        print(f"Error creating performance diagram: {str(e)}")
        traceback.print_exc()


def generate_frame_count_markdown(results: dict[str, Any], outputPath: str) -> None:
    """Generate a markdown table comparing frame counts to OpenCV and save it."""
    decoders = results.get("decoders", {})
    opencv_count = None
    if "OpenCV" in decoders and "frameCount" in decoders["OpenCV"]:
        opencv_count = decoders["OpenCV"]["frameCount"]
    else:
        opencv_count = None

    lines = []
    lines.append("| Decoder | Frames Decoded | Matches OpenCV? |")
    lines.append("|---------|----------------|-----------------|")

    for decoder, data in decoders.items():
        frame_count = data.get("frameCount", "N/A")
        if opencv_count is not None and isinstance(frame_count, int):
            match = "✅" if frame_count == opencv_count else "❌"
        else:
            match = "N/A"
        lines.append(f"| {decoder} | {frame_count} | {match} |")

    markdown = "\n".join(lines)
    with open(outputPath, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Frame count comparison markdown saved to {outputPath}")


def printResultsSummary(results: dict[str, Any]) -> None:
    """Print a summary of the benchmark results."""
    print(lightcyan("\n===== BENCHMARK RESULTS ====="))
    videoInfo = results.get("videoInfo", {})
    systemInfo = results.get("systemInfo", {})

    print(f"Video: {os.path.basename(results.get('videoPath', 'N/A'))}")
    print(f"Resolution: {videoInfo.get('width', '?')}x{videoInfo.get('height', '?')}")
    print(f"Duration: {videoInfo.get('durationSeconds', 0):.2f} seconds")
    print(f"Frame count: {videoInfo.get('frameCount', '?')}")

    if "error" not in systemInfo:
        cpuInfo = systemInfo.get("cpu", {})
        ramInfo = systemInfo.get("ram", {})
        print(lightcyan("\nSystem Information:"))
        print(
            f"CPU: {cpuInfo.get('model', 'N/A')} ({cpuInfo.get('logicalCores', '?')} cores)"
        )
        print(f"RAM: {ramInfo.get('totalGB', '?')} GB total")
    elif "error" in systemInfo:
        print(f"\nSystem Information: Error - {systemInfo['error']}")

    print(lightcyan("\nDecoder Performance (fps):"))

    validDecoders = {}
    for decoderName, data in results.get("decoders", {}).items():
        if "error" in data:
            print(f"  {decoderName.ljust(26)}: ERROR - {data['error']}")
        elif "fps" in data:
            median = data.get("fpsMedian", data["fps"])
            std = data.get("fpsStd", 0.0)
            cv = data.get("fpsCv", 0.0) * 100
            mn = data.get("fpsMin", median)
            mx = data.get("fpsMax", median)
            runs = data.get("successfulRuns", 1)
            total = data.get("runs", 1)
            print(
                f"  {decoderName.ljust(26)}: "
                f"{median:6.2f} fps  (mean={data.get('fpsMean', median):6.2f} "
                f"std={std:5.2f} cv={cv:4.1f}% "
                f"min={mn:6.2f} max={mx:6.2f}  {runs}/{total} runs)"
            )
            if data["fps"] > 0:
                validDecoders[decoderName] = data
        else:
            print(f"  {decoderName.ljust(26)}: No FPS data available.")

    if validDecoders:
        fastestDecoder = max(
            validDecoders.items(), key=lambda item: item[1].get("fps", 0)
        )
        fastestDecoderName = fastestDecoder[0]
        fastestDecoderFps = fastestDecoder[1].get("fps", 0)
        print(f"\nFastest decoder: {fastestDecoderName} ({fastestDecoderFps:.2f} fps)")
    else:
        print(
            lightcyan("\nNo valid decoders ran successfully to determine the fastest.")
        )

    # Display color conversion analysis results if available
    if "colorConversionAnalysis" in results:
        colorAnalysis = results["colorConversionAnalysis"]
        if "error" not in colorAnalysis:
            print(lightcyan("\n===== COLOR CONVERSION ANALYSIS ====="))
            print("Analyzing YUV420p to RGB24 conversion differences...")

            # Display video color properties
            if "videoColorProperties" in results:
                colorProps = results["videoColorProperties"]
                if "error" not in colorProps:
                    print(f"Video format: {colorProps.get('pix_fmt', 'unknown')}")
                    print(f"Color range: {colorProps.get('color_range', 'unknown')}")
                    print(f"Color space: {colorProps.get('color_space', 'unknown')}")

            # Display summary results for each comparison
            summaries = colorAnalysis.get("summary", {})
            if summaries:
                for comparison_name, summary in summaries.items():
                    decoder_name = comparison_name.replace("FFmpeg_Default_vs_", "")
                    print(f"\nFFmpeg vs {decoder_name}:")

                    # Key metrics
                    brightness_diff = summary.get("brightness_diff_mean", 0)
                    mae_mean = summary.get("mae_mean", 0)
                    psnr_mean = summary.get("psnr_mean", 0)

                    print(f"  Average brightness difference: {brightness_diff:.2f}")
                    print(f"  Mean Absolute Error: {mae_mean:.2f}")
                    print(f"  Peak Signal-to-Noise Ratio: {psnr_mean:.2f} dB")

                    # Interpretations
                    interpretations = summary.get("interpretation", [])
                    if interpretations:
                        print("  Issues detected:")
                        for interpretation in interpretations:
                            print(f"    - {interpretation}")
                    else:
                        print("  No significant color conversion issues detected.")
            else:
                print("No comparison summaries available.")
        else:
            print(f"\nColor conversion analysis failed: {colorAnalysis['error']}")


def main() -> None:
    """Main function to run the benchmark."""

    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="",
        required=False,
        help="Use a custom input video file (overrides default videos).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("BENCHMARK_RUNS", "3")),
        help="Number of timed iterations per decoder (median is reported). Default 3.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("BENCHMARK_WARMUP", "1")),
        help="Number of untimed warmup iterations before the timed runs. Default 1.",
    )
    parser.add_argument(
        "--cooling",
        type=int,
        default=int(os.environ.get("BENCHMARK_COOLING", "3")),
        help="Seconds to wait between iterations and decoders. Default 3.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip appending results to history/.",
    )

    arguments = parser.parse_args()

    # Originals (commondatastorage.googleapis.com / gtv-videos-bucket) started
    # returning HTTP 403 in early 2026 regardless of User-Agent — the bucket is
    # gated. test-videos.co.uk is the most stable public mirror that still
    # serves direct mp4 downloads without a CDN gate.
    defaultVideos = [
        {
            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_30MB.mp4",
            "path": os.path.join("videos", "BigBuckBunny_720p.mp4"),
            "results": "720p_results.json",
            "diagram": "720p_diagram.png",
        },
        {
            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_30MB.mp4",
            "path": os.path.join("videos", "BigBuckBunny_360p.mp4"),
            "results": "360p_results.json",
            "diagram": "360p_diagram.png",
        },
    ]

    if arguments.input:
        videoPath = absolutePath(arguments.input)

        basename = os.path.splitext(os.path.basename(videoPath))[0]

        videos = [
            {
                "path": videoPath,
                "results": f"{basename}_results.json",
                "diagram": f"{basename}_performance.png",
            }
        ]

    else:
        videos = defaultVideos

        os.makedirs("videos", exist_ok=True)

        for video in videos:
            if not os.path.isfile(video["path"]):
                downloadVideo(video["url"], video["path"])

    systemInfo = getSystemInfo()

    for i, video in enumerate(videos):
        time.sleep(3)  # Cooldown before starting the next video

        videoPath = video["path"]

        resultsPath = video["results"]

        diagramPath = video["diagram"]

        markdownPath = os.path.splitext(resultsPath)[0] + "_frame_count_comparison.md"

        if not os.path.isfile(videoPath):
            print(f"Error: Expected video file at {videoPath}, but it's not a file.")

            continue

        print(
            lightcyan(
                f"\nStarting benchmark {i + 1}/{len(videos)}: {os.path.basename(videoPath)}..."
            )
        )

        results = runBenchmark(
            videoPath=videoPath,
            coolingPeriod=arguments.cooling,
            systemInfo=systemInfo,
            runs=arguments.runs,
            warmup=arguments.warmup,
        )

        printResultsSummary(results)

        saveResults(results, resultsPath)

        generate_frame_count_markdown(results, markdownPath)

        createPerformanceDiagram(results, diagramPath)

        if not arguments.no_history:
            from src.history import appendHistory, plotHistoryTrends

            resolutionKey = os.path.splitext(os.path.basename(resultsPath))[0].replace(
                "_results", ""
            )
            historyPath = os.path.join("history", f"{resolutionKey}_history.json")
            trendPath = os.path.join("history", f"{resolutionKey}_trend.png")
            appendHistory(results, historyPath, resolutionKey)
            plotHistoryTrends(historyPath, trendPath, resolutionKey)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    main()
