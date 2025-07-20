import os
import json
import signal
import time
import urllib.request
import traceback
import platform
import psutil
import cv2
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from dataclasses import dataclass
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
from src.backends.basswoodav import decodeWithBasswoodAV
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
from src.backends.celux import decodeWithCeLux
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

    urllib.request.urlretrieve(url, outputPath)
    print(f"Video downloaded to {outputPath}")

    return outputPath


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


def runBenchmark(
    videoPath: str, coolingPeriod: int = 3, systemInfo: dict = {}
) -> dict[str, Any]:
    """Run benchmark on all decoders and return the results.

    Args:
        videoPath: Path to the video file to benchmark
        coolingPeriod: Time in seconds to wait between decoder tests
    """
    print("Getting video information...")
    videoInfo = getVideoInfo(videoPath)

    decoders: list[Decoder] = [
        Decoder(name="CeLux", decoder=decodeWithCeLux, cooling=coolingPeriod),
        Decoder(name="PyAV", decoder=decodeWithPyAV, cooling=coolingPeriod),
        Decoder(name="BasswoodAV", decoder=decodeWithBasswoodAV, cooling=coolingPeriod),
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
            lightcyan(f"\n({i + 1}/{len(decoders)}) Running {decoder.name} decoder...")
        )
        decoderFCT = decoder.decoder
        decodingResults[decoder.name] = (
            decoderFCT(videoPath)
            if decoderFCT.__code__.co_argcount == 1
            else decoderFCT(videoPath, decoder.videoInfo)
        )
        time.sleep(decoder.cooling)

    print(lightcyan("\nBenchmark completed."))

    results = {
        "videoPath": videoPath,
        "videoInfo": videoInfo,
        "systemInfo": systemInfo,
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
            print(f"  {decoderName.ljust(10)}: ERROR - {data['error']}")
        elif "fps" in data:
            print(
                f"  {decoderName.ljust(10)}: {data['fps']:.2f} fps ({data.get('elapsedTime', 0):.2f} seconds)"
            )
            if data["fps"] > 0:
                validDecoders[decoderName] = data
        else:
            print(f"  {decoderName.ljust(10)}: No FPS data available.")

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
