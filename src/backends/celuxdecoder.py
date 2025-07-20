import time
from typing import Any

try:
    from celux import VideoReader
except ImportError:
    print("CeLux error: Not supported on this platform or missing dependencies.")


def decodeWithCeLux(videoPath: str) -> dict[str, Any]:
    """
    Decode video using CeLux and return the frame count, elapsed time, and fps.
    """
    try:
        print("Decoding with CeLux...")

        reader = VideoReader(videoPath)
        frameCount = 0

        startTime = time.time()
        for frame in reader:
            # frame is a torch tensor (H, W, C) by default
            frameCount += 1
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(f"CeLux: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in CeLux decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }


if __name__ == "__main__":
    videoPath = r"F:\testVideos\output_bt601.mp4"
    result = decodeWithCeLux(videoPath)
    print(result)
