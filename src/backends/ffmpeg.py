import subprocess
import time
import numpy as np
import cv2
from typing import Dict, Any


def decodeWithFFMPEG_RGB24(videoPath: str, videoInfo) -> Dict[str, Any]:
    """Decode video using FFMPEG with rawvideo format and return metrics."""
    try:
        print("Decoding with FFMPEG (rawvideo, rgb24)...")
        startTime = time.time()

        # Get video info for dimensions needed for numpy reshaping
        width = videoInfo["width"]
        height = videoInfo["height"]

        cmd = [
            "ffmpeg",
            "-i",
            videoPath,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",  # Use RGB24 format (3 channels)
            "-v",
            "quiet",
            "-",  # Output to stdout
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        processedFrames = 0
        bytesPerFrame = width * height * 3

        while True:
            rawFrame = process.stdout.read(bytesPerFrame)
            if len(rawFrame) < bytesPerFrame:
                break

            frameArray = np.frombuffer(rawFrame, dtype=np.uint8).reshape(
                height, width, 3
            )
            frameArray = np.transpose(frameArray, (2, 0, 1))  # CHW format

            processedFrames += 1

        process.stdout.close()
        process.wait()

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"FFMPEG (RGB): Processed {processedFrames} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": processedFrames,
            "elapsedTime": elapsedTime,
            "fps": processedFrames / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in FFMPEG RGB decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }


def decodeWithFFMPEG_YUV420(videoPath: str, videoInfo) -> Dict[str, Any]:
    """Decode video using FFMPEG with yuv420p format and return metrics."""
    try:
        print("Decoding with FFMPEG (rawvideo, yuv420p)...")
        startTime = time.time()

        width = videoInfo["width"]
        height = videoInfo["height"]

        cmd = [
            "ffmpeg",
            "-i",
            videoPath,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-v",
            "quiet",
            "-",
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        processedFrames = 0
        bytesPerFrame = (
            width * height * 3 // 2
        )  # YUV420 is 12 bits per pixel (1.5 bytes)

        while True:
            rawFrame = process.stdout.read(bytesPerFrame)
            if len(rawFrame) < bytesPerFrame:
                break

            # Convert YUV to RGB directly using OpenCV
            frameArray = cv2.cvtColor(
                np.frombuffer(rawFrame, dtype=np.uint8).reshape(height * 3 // 2, width),
                cv2.COLOR_YUV2RGB_I420,
            )

            # Transpose to CHW format for consistency with other decoders
            frameArray = np.transpose(frameArray, (2, 0, 1))

            processedFrames += 1

        process.stdout.close()
        process.wait()

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"FFMPEG (YUV): Processed {processedFrames} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": processedFrames,
            "elapsedTime": elapsedTime,
            "fps": processedFrames / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in FFMPEG YUV decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
