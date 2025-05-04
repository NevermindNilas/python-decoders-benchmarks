from argparse import ArgumentParser
from dataclasses import dataclass
from enum import IntEnum
import os
import json
import signal
import time
import urllib.request
import traceback
import platform
from typing import Any, Callable

import cv2

import psutil
import matplotlib.pyplot as plt

from src.backends.ffmpeg import decodeWithFFMPEG_RGB24
from src.backends.imageio import decodeWithImageioFFMPEG
from src.backends.opencv import decodeWithOpenCV
from src.backends.pyav import decodeWithPyAV
from src.backends.torchaudio import decodeWithTorchaudio
from src.backends.ffmpegcv import decodeWithFFMPEGCV_Block, decodeWithFFMPEGCV_NoBlock
from src.backends.decord import decodeWithDecord
from src.backends.deffcode import decodeWithDeffcode
from src.backends.maxTheoretical import decodeWithMaxTheoretical
from src.backends.basswoodav import decodeWithBasswoodAV
from src.backends.videoreaderrs import decodeWithVideoReaderRS
from src.backends.torchcodec import decodeWithTorchCodec


class ColorCode(IntEnum):
    lightgreen = 92
    lightcyan = 96

def _color_str_template(color: ColorCode) -> str:
    return "\033[%dm{}\033[00m" % (color.value)

def lightgreen(*values: object) -> str:
    return _color_str_template(ColorCode.lightgreen).format(values[0])

def lightcyan(*values: object) -> str:
    return _color_str_template(ColorCode.lightcyan).format(values[0])

def absolute_path(path: str) -> str:
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
    video_info: Any | None = None


def runBenchmark(videoPath: str, coolingPeriod: int = 3) -> dict[str, Any]:
    """Run benchmark on all decoders and return the results.

    Args:
        videoPath: Path to the video file to benchmark
        coolingPeriod: Time in seconds to wait between decoder tests
    """
    print("Getting video information...")
    videoInfo = getVideoInfo(videoPath)
    systemInfo = getSystemInfo()

    decoders: list[Decoder] = [
        Decoder(
            name="PyAV", decoder=decodeWithPyAV, cooling=coolingPeriod
        ),
        Decoder(
            name="BasswoodAV", decoder=decodeWithBasswoodAV, cooling=coolingPeriod
        ),
        Decoder(
            name="OpenCV", decoder=decodeWithOpenCV, cooling=coolingPeriod
        ),
        Decoder(
            name="torchaudio", decoder=decodeWithTorchaudio, cooling=coolingPeriod
        ),
        Decoder(
            name="TorchCodec", decoder=decodeWithTorchCodec, cooling=coolingPeriod
        ),
        Decoder(
            name="Decord", decoder=decodeWithDecord, cooling=coolingPeriod
        ),
        Decoder(
            name="VideoReaderRS", decoder=decodeWithVideoReaderRS, cooling=coolingPeriod
        ),
        Decoder(
            name="FFMPEGCV (Block)", decoder=decodeWithFFMPEGCV_Block, cooling=coolingPeriod
        ),
        Decoder(
            name="FFmpeg-Subprocess", decoder=decodeWithFFMPEG_RGB24, cooling=coolingPeriod, video_info=videoInfo
        ),
        Decoder(
            name="Imageio-ffmpeg", decoder=decodeWithImageioFFMPEG, cooling=coolingPeriod
        ),
        Decoder(
            name="FFmpegCV-NoBlock", decoder=decodeWithFFMPEGCV_NoBlock, cooling=coolingPeriod
        ),
        Decoder(
            name="Deffcode", decoder=decodeWithDeffcode, cooling=coolingPeriod
        ),
        Decoder(
            name="Max Theoretical", decoder=decodeWithMaxTheoretical, cooling=coolingPeriod
        ),
    ]

    decoding_results: dict[str, Any] = {}
    for i, decoder in enumerate(decoders):
        print(lightcyan(f"\n({i + 1}/{len(decoders)}) Running {decoder.name} decoder..."))
        decoder_fct = decoder.decoder
        decoding_results[decoder.name] = (
            decoder_fct(videoPath)
            if decoder_fct.__code__.co_argcount == 1
            else decoder_fct(videoPath, decoder.video_info)
        )
        time.sleep(decoder.cooling)

    print(lightcyan("\nBenchmark completed."))

    results = {
        "videoPath": videoPath,
        "videoInfo": videoInfo,
        "systemInfo": systemInfo,
        "decoders": decoding_results,
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
            width=0.5,
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
        print(lightcyan("\nNo valid decoders ran successfully to determine the fastest."))


def main() -> None:
    """Main function to run the benchmark."""
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="",
        required=False,
        help="Use a custom input video file.",
    )
    arguments = parser.parse_args()
    # Input video
    videoFilename: str = "sample_720p.mp4"
    videoUrl: str = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"
    videoDir: str = "videos"
    videoPath: str = os.path.join(videoDir, videoFilename)
    # Results
    resultsPath: str = "results.json"
    diagramPath: str = "decoder_performance.png"

    # Use custom video file
    if arguments.input:
        videoPath = absolute_path(arguments.input)

    # If not exists, download
    if not os.path.isfile(videoPath):
        os.makedirs(videoDir, exist_ok=True)
        downloadVideo(videoUrl, videoPath)

    # Verify that it exists
    if not os.path.isfile(videoPath):
        print(f"Error: Expected video file at {videoPath}, but it's not a file.")
        return

    print(lightcyan("\nStarting benchmark..."))
    results = runBenchmark(videoPath)

    printResultsSummary(results)
    saveResults(results, resultsPath)

    createPerformanceDiagram(results, diagramPath)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
