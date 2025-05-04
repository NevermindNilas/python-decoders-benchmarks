import time

try:
    from video_reader import PyVideoReader
except ImportError:
    print("PyVideoReader error: https://github.com/gcanat/video_reader-rs/issues/52")

from typing import Any


def decodeWithVideoReaderRS(videoPath: str) -> dict[str, Any]:
    """Decode video using VideoReaderRS and return metrics."""
    try:
        print("Decoding with VideoReaderRS...")
        startTime = time.time()

        reader = PyVideoReader(videoPath)
        frameCount = 0

        # This is from the offiical readme.md,
        # It doesn't seem to natively support a simple frame iterator,
        # May not be 100% accurate since the decode may happen async on rust side.
        # And simply yield 800 frames at a time which can artifically improve performance.
        # It should default to np rgb hwc format. But take it with a grain of salt since it doesn't work on windows
        # and I don't want to bother with WSL.

        # Maybe using chunksize=1 would be more accurate, but potentially slower.
        chunkSize = 800
        videoLenght = reader.get_shape()[0]

        frameCount = 0
        for i in range(0, videoLenght, chunkSize):
            end = min(i + chunkSize, videoLenght)
            frames = reader.decode(
                start_frame=i,
                end_frame=end,
            )
            for _ in frames:
                frameCount += 1
                pass

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"VideoReaderRS: Processed {frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in VideoReaderRS decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }


def decodeWithVideoReaderRSFast(videoPath: str) -> dict[str, Any]:
    """Decode video using VideoReaderRS and return metrics."""
    try:
        print("Decoding with VideoReaderRS...")
        startTime = time.time()

        reader = PyVideoReader(videoPath)
        frameCount = 0

        # there's a decode_fast function which is apparently async,
        # I am not sure if that implies decoding a chunk of say 800 frames and while doing so
        # another chunk of 800 frames are decoded in the background.
        # or this just continuosly decodes frames and whenever chunk is ready it returns it.
        # This is from the offiical readme.md,
        chunkSize = 800
        videoLenght = reader.get_shape()[0]

        frameCount = 0
        for i in range(0, videoLenght, chunkSize):
            end = min(i + chunkSize, videoLenght)
            frames = reader.decode_fast(
                start_frame=i,
                end_frame=end,
            )
            for _ in frames:
                frameCount += 1
                pass

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(
            f"VideoReaderRS: Processed {frameCount} frames in {elapsedTime:.2f} seconds"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in VideoReaderRS decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
