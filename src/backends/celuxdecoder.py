import time
from typing import Any

try:
    import torch  # Nelux links to torch's native runtime; load it first.
    from nelux import VideoReader
    _importError = None
except (ImportError, OSError) as neluxError:
    _importError = str(neluxError)
    try:
        # Fall back only if Nelux is absent, never hide a broken Nelux install.
        if not isinstance(neluxError, ModuleNotFoundError) or neluxError.name != "nelux":
            raise neluxError
        from celux import VideoReader
    except (ImportError, OSError):
        VideoReader = None
        print(f"Nelux import error: {_importError}")


def decodeWithCeLux(videoPath: str) -> dict[str, Any]:
    """
    Decode video using Nelux (formerly CeLux) and return the frame count,
    elapsed time, and fps.
    """
    try:
        if VideoReader is None:
            raise ImportError(_importError or "nelux/celux module not available")
        print("Decoding with Nelux...")

        frameCount = 0
        with VideoReader(videoPath, backend="numpy", decode_accelerator="cpu", force_8bit=True) as reader:
            startTime = time.perf_counter()
            for frame in reader:
                # RGB24 numpy output, matching the CPU decoders.
                frameCount += 1
            endTime = time.perf_counter()

        elapsedTime = endTime - startTime
        print(f"Nelux: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in Nelux decoder: {str(e)}")
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
