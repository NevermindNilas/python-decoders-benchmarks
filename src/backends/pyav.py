import av
import time
from typing import Any


def decodeWithPyAV(videoPath: str) -> dict[str, Any]:
    """Decode video using PyAV and return the frame count."""
    try:
        print("Decoding with PyAV...")

        container = av.open(videoPath)
        frameCount = 0

        startTime = time.time()
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")  # ensure RGB
            frameCount += 1

        container.close()
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(f"PyAV: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in PyAV decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
