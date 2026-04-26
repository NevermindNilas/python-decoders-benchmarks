import time
from typing import Any

try:
    import bv  # type: ignore
except ImportError:
    bv = None
    print("BasswoodAV error: not installed (no wheel for this Python yet).")


def decodeWithBasswoodAV(videoPath: str) -> dict[str, Any]:
    """Decode video using BasswoodAV and return the frame count."""
    try:
        if bv is None:
            raise ImportError("bv module not available")

        print("Decoding with BasswoodAV...")

        container = bv.open(videoPath)
        frameCount = 0

        startTime = time.time()

        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")  # ensure RGB
            frameCount += 1

        container.close()
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(f"BasswoodAV: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in BasswoodAV decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
