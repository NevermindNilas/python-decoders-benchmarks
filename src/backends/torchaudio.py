import time
import torchaudio
from typing import Any


def decodeWithTorchaudio(videoPath: str) -> dict[str, Any]:
    """Decode video using torchaudio and return metrics."""
    try:
        print("Decoding with torchaudio...")
        startTime = time.time()

        streamer = torchaudio.io.StreamReader(src=videoPath)
        videoStreamIndex = -1
        for i in range(streamer.num_src_streams):
            info = streamer.get_src_stream_info(i)
            if info.media_type == "video":
                videoStreamIndex = i
                streamer.add_basic_video_stream(
                    frames_per_chunk=1, stream_index=videoStreamIndex
                )
                break

        if videoStreamIndex == -1:
            raise RuntimeError("No video stream found in the file.")

        frameCount = 0
        for (chunk,) in streamer.stream():
            frameCount += chunk.shape[0]

        endTime = time.time()
        elapsedTime = endTime - startTime

        print(f"torchaudio: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in torchaudio decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
