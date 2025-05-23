import time
import imageio_ffmpeg
import numpy as np
from typing import Any


def decodeWithImageioFFMPEG(videoPath: str) -> dict[str, Any]:
    """Decode video using imageio-ffmpeg and return metrics."""
    try:
        print("Decoding with imageio-ffmpeg...")

        reader = imageio_ffmpeg.read_frames(videoPath)
        meta = next(reader)
        width = meta["size"][0]
        height = meta["size"][1]

        startTime = time.time()
        frameCount = 1
        for frame in reader:
            frame = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
            frameCount += 1

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"imageio-ffmpeg: Processed {frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in imageio-ffmpeg decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
