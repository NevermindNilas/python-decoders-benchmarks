import bv
import time
from typing import Dict, Any


def decodeWithBasswoodAV(videoPath: str) -> Dict[str, Any]:
    """Decode video using BasswoodAV and return the frame count."""
    try:
        print("Decoding with BasswoodAV...")
        startTime = time.time()

        container = bv.open(videoPath)
        frameCount = 0

        for frame in container.decode(video=0):
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
