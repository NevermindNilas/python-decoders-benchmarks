import time
import decord
from typing import Dict, Any


def decodeWithDecord(videoPath: str) -> Dict[str, Any]:
    """Decode video using Decord and return metrics."""
    try:
        print("Decoding with Decord...")
        startTime = time.time()

        vr = decord.VideoReader(videoPath, ctx=decord.cpu(0))
        frameCount = 0

        for i in range(len(vr)):
            frame = vr[i]
            frameCount += 1

        endTime = time.time()
        elapsedTime = endTime - startTime

        # Use len(vr) as the official frame count from the reader
        actualFrameCount = len(vr)
        print(
            f"Decord: Processed {actualFrameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": actualFrameCount,
            "elapsedTime": elapsedTime,
            "fps": actualFrameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in Decord decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
