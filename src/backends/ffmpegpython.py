import time
import ffmpeg
import numpy as np
from typing import Any


def decodeWithFFmpegPython(videoPath: str, videoInfo) -> dict[str, Any]:
    """Decode video using ffmpeg-python and return metrics."""
    try:
        print("Decoding with ffmpeg-python...")
        startTime = time.time()

        width = videoInfo["width"]
        height = videoInfo["height"]
        bytesPerFrame = width * height * 3

        process = (
            ffmpeg.input(videoPath)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", v="quiet")
            .run_async(pipe_stdout=True)
        )

        frameCount = 0
        while True:
            inBytes = process.stdout.read(bytesPerFrame)
            if not inBytes:
                break
            if len(inBytes) != bytesPerFrame:
                print(
                    f"Warning: Incomplete frame data received. Expected {bytesPerFrame}, got {len(inBytes)}"
                )
                continue

            frame = np.frombuffer(inBytes, np.uint8).reshape([height, width, 3])
            frameCount += 1

        process.wait()
        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"ffmpeg-python: Processed {frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in ffmpeg-python decoder: {str(e)}")
        # Ensure process is terminated if it exists and is running
        if "process" in locals() and process.poll() is None:
            process.terminate()
            process.wait()
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
