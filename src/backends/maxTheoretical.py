import time
import subprocess
import platform
import cv2
from typing import Dict, Any


def decodeWithMaxTheoretical(videoPath: str) -> Dict[str, Any]:
    """
    Measure theoretical maximum decoding speed by using ffmpeg with output
    piped to devnull. This eliminates overhead from Python processing.
    """
    try:
        print("Measuring theoretical maximum decoding speed...")
        startTime = time.time()

        cmd = [
            "ffmpeg",
            "-i",
            videoPath,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-v",
            "quiet",
            "-stats",
            "-y",
            "/dev/null",
        ]

        if platform.system() == "Windows":
            cmd[-1] = "NUL"

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()

        frameCount = 0
        try:
            cap = cv2.VideoCapture(videoPath)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            print(f"Warning: Failed to get frame count with OpenCV: {str(e)}")

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"Max Theoretical: Processed ~{frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in Max Theoretical decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
