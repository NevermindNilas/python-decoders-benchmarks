import subprocess
import time
import numpy as np
import cv2
from typing import Any, Dict, List, Tuple
import tempfile
import os


def extractFrameWithFFMPEG_YUV420toRGB24(
    videoPath: str, frameIndex: int, videoInfo: dict
) -> np.ndarray:
    """Extract a specific frame using FFmpeg YUV420p to RGB24 conversion (limited range)."""
    width = videoInfo["width"]
    height = videoInfo["height"]

    cmd = [
        "ffmpeg",
        "-i",
        videoPath,
        "-vf",
        f"select=eq(n\\,{frameIndex})",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vframes",
        "1",
        "-v",
        "quiet",
        "-",
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rawFrame, stderr = process.communicate()

    if len(rawFrame) != width * height * 3:
        raise ValueError(f"Expected {width * height * 3} bytes, got {len(rawFrame)}")

    return np.frombuffer(rawFrame, dtype=np.uint8).reshape(height, width, 3)


def extractFrameWithFFMPEG_FullRange(
    videoPath: str, frameIndex: int, videoInfo: dict
) -> np.ndarray:
    """Extract a specific frame using FFmpeg with explicit full range conversion."""
    width = videoInfo["width"]
    height = videoInfo["height"]

    cmd = [
        "ffmpeg",
        "-i",
        videoPath,
        "-vf",
        f"select=eq(n\\,{frameIndex}),scale=in_range=limited:out_range=full",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vframes",
        "1",
        "-v",
        "quiet",
        "-",
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rawFrame, stderr = process.communicate()

    if len(rawFrame) != width * height * 3:
        raise ValueError(f"Expected {width * height * 3} bytes, got {len(rawFrame)}")

    return np.frombuffer(rawFrame, dtype=np.uint8).reshape(height, width, 3)


def extractFrameWithOpenCV(videoPath: str, frameIndex: int) -> np.ndarray:
    """Extract a specific frame using OpenCV."""
    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not extract frame {frameIndex}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extractFrameWithPyAV(videoPath: str, frameIndex: int) -> np.ndarray:
    """Extract a specific frame using PyAV."""
    try:
        import av

        container = av.open(videoPath)
        stream = container.streams.video[0]

        frameCount = 0
        for frame in container.decode(video=0):
            if frameCount == frameIndex:
                frameArray = frame.to_ndarray(format="rgb24")
                container.close()
                return frameArray
            frameCount += 1

        container.close()
        raise ValueError(f"Could not extract frame {frameIndex}")
    except ImportError:
        raise ValueError("PyAV not available")


def extractFrameWithBasswoodAV(videoPath: str, frameIndex: int) -> np.ndarray:
    """Extract a specific frame using BasswoodAV."""
    try:
        import bv

        container = bv.open(videoPath)
        frameCount = 0

        for frame in container.decode(video=0):
            if frameCount == frameIndex:
                frameArray = frame.to_ndarray(format="rgb24")
                container.close()
                return frameArray
            frameCount += 1

        container.close()
        raise ValueError(f"Could not extract frame {frameIndex}")
    except ImportError:
        raise ValueError("BasswoodAV not available")


def calculateFrameDifference(
    frame1: np.ndarray, frame2: np.ndarray
) -> Dict[str, float]:
    """Calculate various difference metrics between two frames."""
    if frame1.shape != frame2.shape:
        raise ValueError("Frames must have the same shape")

    # Calculate absolute difference
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))

    # Mean Absolute Error (MAE)
    mae = np.mean(diff)

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(diff**2))

    # Peak Signal-to-Noise Ratio (PSNR)
    mse = np.mean(diff**2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # Structural Similarity Index (SSIM) - simplified version
    mu1 = np.mean(frame1)
    mu2 = np.mean(frame2)
    sigma1_sq = np.var(frame1)
    sigma2_sq = np.var(frame2)
    sigma12 = np.mean((frame1 - mu1) * (frame2 - mu2))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    # Histogram comparison
    hist1 = np.histogram(frame1, bins=256, range=(0, 256))[0]
    hist2 = np.histogram(frame2, bins=256, range=(0, 256))[0]
    hist_correlation = np.corrcoef(hist1, hist2)[0, 1]

    # Color range statistics
    min1, max1 = np.min(frame1), np.max(frame1)
    min2, max2 = np.min(frame2), np.max(frame2)

    # Calculate brightness difference (contrast issue indicator)
    brightness1 = np.mean(frame1)
    brightness2 = np.mean(frame2)
    brightness_diff = brightness2 - brightness1

    # Calculate range utilization (limited vs full range indicator)
    range_utilization1 = (max1 - min1) / 255.0
    range_utilization2 = (max2 - min2) / 255.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "histogram_correlation": float(hist_correlation)
        if not np.isnan(hist_correlation)
        else 0.0,
        "brightness_diff": float(brightness_diff),
        "range_diff_min": float(abs(min1 - min2)),
        "range_diff_max": float(abs(max1 - max2)),
        "range_utilization_diff": float(abs(range_utilization1 - range_utilization2)),
        "frame1_stats": {
            "min": int(min1),
            "max": int(max1),
            "mean": float(brightness1),
            "range_utilization": float(range_utilization1),
        },
        "frame2_stats": {
            "min": int(min2),
            "max": int(max2),
            "mean": float(brightness2),
            "range_utilization": float(range_utilization2),
        },
    }


def analyzeColorConversionDifferences(
    videoPath: str, videoInfo: dict, framesToTest: List[int] = None
) -> Dict[str, Any]:
    """Analyze color conversion differences between FFmpeg YUV420p->RGB24 and other decoders."""

    if framesToTest is None:
        # Test frames at different positions: beginning, 25%, 50%, 75%, end
        totalFrames = videoInfo.get("frameCount", 100)
        framesToTest = [
            0,
            totalFrames // 4,
            totalFrames // 2,
            3 * totalFrames // 4,
            max(0, totalFrames - 1),
        ]

    results = {
        "tested_frames": framesToTest,
        "comparisons": {},
        "summary": {},
        "analysis_notes": {
            "purpose": "Analyze color conversion differences between FFmpeg YUV420p->RGB24 and other decoders",
            "yuv420p_issue": "YUV420p typically uses limited range (16-235) while RGB24 uses full range (0-255)",
            "expected_differences": "FFmpeg default conversion may show contrast boost compared to other decoders",
        },
    }

    decoders = {
        "FFmpeg_Default": lambda v, f, vi: extractFrameWithFFMPEG_YUV420toRGB24(
            v, f, vi
        ),
        "FFmpeg_FullRange": lambda v, f, vi: extractFrameWithFFMPEG_FullRange(v, f, vi),
        "OpenCV": lambda v, f, vi: extractFrameWithOpenCV(v, f),
        "PyAV": lambda v, f, vi: extractFrameWithPyAV(v, f),
        "BasswoodAV": lambda v, f, vi: extractFrameWithBasswoodAV(v, f),
    }

    # Extract frames from each decoder
    extractedFrames = {}
    for decoderName, extractorFunc in decoders.items():
        extractedFrames[decoderName] = {}
        for frameIndex in framesToTest:
            try:
                print(f"Extracting frame {frameIndex} with {decoderName}...")
                frame = extractorFunc(videoPath, frameIndex, videoInfo)
                extractedFrames[decoderName][frameIndex] = frame
                print(f"Successfully extracted frame {frameIndex} with {decoderName}")
            except Exception as e:
                print(f"Failed to extract frame {frameIndex} with {decoderName}: {e}")
                extractedFrames[decoderName][frameIndex] = None

    # Compare FFmpeg Default with other decoders
    ffmpeg_default_frames = extractedFrames.get("FFmpeg_Default", {})

    for otherDecoderName, otherFrames in extractedFrames.items():
        if otherDecoderName == "FFmpeg_Default":
            continue

        comparison_key = f"FFmpeg_Default_vs_{otherDecoderName}"
        results["comparisons"][comparison_key] = {}

        for frameIndex in framesToTest:
            ffmpeg_frame = ffmpeg_default_frames.get(frameIndex)
            other_frame = otherFrames.get(frameIndex)

            if ffmpeg_frame is not None and other_frame is not None:
                try:
                    diff_metrics = calculateFrameDifference(ffmpeg_frame, other_frame)
                    results["comparisons"][comparison_key][frameIndex] = diff_metrics
                    print(
                        f"Compared frame {frameIndex}: FFmpeg_Default vs {otherDecoderName}"
                    )
                except Exception as e:
                    print(f"Failed to compare frame {frameIndex}: {e}")
                    results["comparisons"][comparison_key][frameIndex] = {
                        "error": str(e)
                    }
            else:
                missing = []
                if ffmpeg_frame is None:
                    missing.append("FFmpeg_Default")
                if other_frame is None:
                    missing.append(otherDecoderName)
                results["comparisons"][comparison_key][frameIndex] = {
                    "error": f"Missing frames from: {', '.join(missing)}"
                }

    # Calculate summary statistics
    for comparison_name, frame_comparisons in results["comparisons"].items():
        valid_comparisons = [
            metrics for metrics in frame_comparisons.values() if "error" not in metrics
        ]

        if valid_comparisons:
            summary = {}
            metrics_to_summarize = [
                "mae",
                "rmse",
                "psnr",
                "ssim",
                "histogram_correlation",
                "brightness_diff",
                "range_utilization_diff",
            ]

            for metric in metrics_to_summarize:
                values = [
                    comp[metric]
                    for comp in valid_comparisons
                    if not np.isnan(comp[metric]) and not np.isinf(comp[metric])
                ]
                if values:
                    summary[f"{metric}_mean"] = float(np.mean(values))
                    summary[f"{metric}_std"] = float(np.std(values))
                    summary[f"{metric}_min"] = float(np.min(values))
                    summary[f"{metric}_max"] = float(np.max(values))

            # Add interpretation
            brightness_diff_mean = summary.get("brightness_diff_mean", 0)
            range_util_diff_mean = summary.get("range_utilization_diff_mean", 0)

            interpretation = []
            if abs(brightness_diff_mean) > 5:
                if brightness_diff_mean > 0:
                    interpretation.append(
                        "Other decoder produces brighter images than FFmpeg"
                    )
                else:
                    interpretation.append(
                        "FFmpeg produces brighter images than other decoder (possible contrast boost)"
                    )

            if range_util_diff_mean > 0.1:
                interpretation.append(
                    "Significant difference in color range utilization detected"
                )

            if summary.get("mae_mean", 0) > 10:
                interpretation.append("Significant color differences detected")

            summary["interpretation"] = interpretation
            results["summary"][comparison_name] = summary

    return results


def analyzeVideoColorProperties(videoPath: str) -> Dict[str, Any]:
    """Analyze the color properties of the input video using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            videoPath,
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if process.returncode != 0:
            return {"error": f"ffprobe failed: {error.decode()}"}

        import json

        data = json.loads(output.decode())

        if "streams" not in data or len(data["streams"]) == 0:
            return {"error": "No video stream found"}

        stream = data["streams"][0]

        return {
            "pix_fmt": stream.get("pix_fmt", "unknown"),
            "color_space": stream.get("color_space", "unknown"),
            "color_range": stream.get("color_range", "unknown"),
            "color_primaries": stream.get("color_primaries", "unknown"),
            "color_transfer": stream.get("color_trc", "unknown"),
            "width": stream.get("width", 0),
            "height": stream.get("height", 0),
            "notes": {
                "yuv420p_range_issue": "If pix_fmt is yuv420p and color_range is 'tv' or 'limited', conversion to RGB may cause contrast issues",
                "expected_range": "Limited range: Y=16-235, UV=16-240; Full range: Y=0-255, UV=0-255",
            },
        }

    except Exception as e:
        return {"error": str(e)}
