import time
import cv2
from typing import Any


def decodeWithOpenCV(videoPath: str) -> dict[str, Any]:
    """Decode video using OpenCV and return the frame count."""
    try:
        print("Decoding with OpenCV...")

        cap = cv2.VideoCapture(videoPath)
        frameCount = 0
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        startTime = time.time()
        # Switched to for loops for potentially better performance
        # May not be as accurate as using while loop due to potentially skipping frames
        # OpenCV cv2.CAP_PROP_FRAME_COUNT is not always accurate, hope that it works.
        for _ in range(totalFrames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameCount += 1

        cap.release()
        endTime = time.time()

        elapsedTime = endTime - startTime
        print(f"OpenCV: Processed {frameCount} frames in {elapsedTime:.2f} seconds")

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
        }
    except Exception as e:
        print(f"Error in OpenCV decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }
