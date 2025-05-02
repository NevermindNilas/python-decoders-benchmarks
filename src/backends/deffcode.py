import time

from deffcode import FFdecoder
from typing import Dict, Any


def decodeWithDeffcode(videoPath: str) -> Dict[str, Any]:
    """Decode video using FFDecoder and return metrics."""
    try:
        print("Decoding with FFDecoder...")
        startTime = time.time()

        decoder = FFdecoder(videoPath, verbose=False).formulate()
        frameCount = 0

        for frame in decoder.generateFrame():
            if frame is None:
                break
            frameCount += 1

        decoder.terminate()
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(f"FFDecoder: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in FFDecoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
