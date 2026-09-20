import os
import json
import signal
import statistics
import subprocess
import time
import urllib.request
import traceback
import platform
import hashlib
from importlib.metadata import version, PackageNotFoundError
import psutil
import cv2
from src.reporting import createPerformanceDiagram, generate_frame_count_markdown

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
    runner = os.environ.get("BENCHMARK_RUNNER") or os.environ.get("RUNNER_NAME", "")
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
        "runnerName": os.environ.get("RUNNER_NAME", platform.node()),
        "runnerImage": os.environ.get("ImageVersion", ""),
        "runId": os.environ.get("GITHUB_RUN_ID"),
        "runAttempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
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
        cpuModel = platform.processor()
        if os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8") as cpuFile:
                cpuModel = next((line.split(":", 1)[1].strip() for line in cpuFile if line.startswith("model name")), cpuModel)
        cpuInfo = {
            "model": cpuModel,
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
    with open(videoPath, "rb") as videoFile:
        videoInfo["sha256"] = hashlib.file_digest(videoFile, "sha256").hexdigest()

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
                expectedFrames = videoInfo.get("frameCount", 0)
                if "error" not in result and expectedFrames and result.get("frameCount") != expectedFrames:
                    result["error"] = f"Decoded {result.get('frameCount', 0)} of {expectedFrames} expected frames"
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

    packages = {}
    for package in ("nelux", "torch", "av", "opencv-python", "torchcodec", "decord", "torchaudio", "video-reader-rs", "ffmpegcv", "imageio-ffmpeg", "ffmpeg-python", "deffcode", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    decoderPackages = {
        "Nelux": "nelux", "PyAV": "av", "OpenCV": "opencv-python",
        "TorchCodec": "torchcodec", "Decord": "decord", "torchaudio": "torchaudio",
        "VideoReaderRS": "video-reader-rs", "VideoReaderRS YUV420toRGB": "video-reader-rs",
        "FFMPEGCV (Block)": "ffmpegcv", "FFmpegCV-NoBlock": "ffmpegcv",
        "Imageio-ffmpeg": "imageio-ffmpeg", "FFmpeg-python": "ffmpeg-python", "Deffcode": "deffcode",
    }
    ffmpegVersion = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    for name, data in decodingResults.items():
        data["version"] = packages[decoderPackages[name]] if name in decoderPackages else ffmpegVersion

    results = {
        "videoPath": videoPath,
        "videoInfo": videoInfo,
        "systemInfo": systemInfo,
        "runnerInfo": getRunnerInfo(),
        "environment": {"packages": packages, "ffmpeg": ffmpegVersion},
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
    if arguments.runs < 1 or arguments.warmup < 0 or arguments.cooling < 0:
        parser.error("runs must be positive; warmup and cooling must be nonnegative")

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
