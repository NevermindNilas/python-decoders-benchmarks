import time
import torch
from torchcodec.decoders import VideoDecoder
from typing import Dict, Any


def decodeWithTorchCodec(videoPath: str) -> Dict[str, Any]:
    """Decode video using torchaudio and return metrics."""
    try:
        print("Decoding with torchaudio...")
        startTime = time.time()

        decoder = VideoDecoder(
            videoPath,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )  # maybe github will bless me with a NVIDIA GPU one day

        totalFrames = decoder.metadata.num_frames

        frameCount = 0
        for i in range(totalFrames):
            # Get frame at index i
            frame = decoder[i].cpu().numpy()  # rgb24 chw cpu
            frameCount += 1

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
