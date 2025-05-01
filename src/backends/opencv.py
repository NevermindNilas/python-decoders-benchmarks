import time
import cv2
from typing import Dict, Any


def decodeWithOpenCV(videoPath: str) -> Dict[str, Any]:
    """Decode video using OpenCV and return the frame count."""
    try:
        print("Decoding with OpenCV...")
        startTime = time.time()

        cap = cv2.VideoCapture(videoPath)
        frameCount = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
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
