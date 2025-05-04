import time
import ffmpegcv
from typing import Any


def decodeWithFFMPEGCV_Block(videoPath: str) -> dict[str, Any]:
    """Decode video using ffmpegcv (with Block) and return metrics."""
    try:
        print("Decoding with ffmpegcv (with Block)...")
        startTime = time.time()

        cap = ffmpegcv.VideoCapture(videoPath, pix_fmt="rgb24")
        frameCount = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frameCount += 1

        cap.release()
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(
            f"ffmpegcv (with Block): Processed {frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in ffmpegcv (with Block) decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }


def decodeWithFFMPEGCV_NoBlock(videoPath: str) -> dict[str, Any]:
    """Decode video using ffmpegcv (without Block) and return metrics."""
    try:
        print("Decoding with ffmpegcv (without Block)...")
        startTime = time.time()

        cap = ffmpegcv.noblock(ffmpegcv.VideoCapture, videoPath, pix_fmt="rgb24")
        frameCount = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frameCount += 1

        cap.release()
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(
            f"ffmpegcv (without Block): Processed {frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in ffmpegcv (without Block) decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
