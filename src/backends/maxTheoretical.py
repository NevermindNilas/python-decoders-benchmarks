import re
import subprocess
import time
import cv2
from typing import Any


# `ffmpeg -benchmark` prints a final line like:
#   "bench: utime=1.23s stime=0.05s rtime=2.10s"
# rtime is the wall-clock spent inside ffmpeg, excluding fork/exec, dynamic
# linker, and Python `subprocess.run` housekeeping — what we actually want
# for a "theoretical max" measurement.
_BENCH_RTIME = re.compile(r"bench:\s.*?rtime=([0-9.]+)s")


def _runMaxTheoretical(videoPath: str, pixFmt: str | None, label: str) -> dict[str, Any]:
    try:
        print(f"Measuring theoretical maximum decoding speed ({label})...")
        frameCount = 0
        try:
            cap = cv2.VideoCapture(videoPath)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            print(f"Warning: Failed to get frame count with OpenCV: {str(e)}")

        cmd = [
            "ffmpeg",
            "-benchmark",
            "-threads", "0",
            "-i", videoPath,
            "-an",
            "-fps_mode", "passthrough",
        ]
        if pixFmt:
            cmd += ["-pix_fmt", pixFmt]
        # `-f null -` discards output through the null muxer: no rawvideo
        # serialization, no write to /dev/null. The pipeline still runs
        # demux + decode (+ swscale when pixFmt is set), so this measures
        # those stages without muxer / I/O noise.
        cmd += ["-f", "null", "-"]

        startTime = time.perf_counter()
        process = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        wallTime = time.perf_counter() - startTime

        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {process.returncode}: "
                f"{process.stderr[-400:] if process.stderr else '(no stderr)'}"
            )

        match = _BENCH_RTIME.search(process.stderr or "")
        if match:
            elapsedTime = float(match.group(1))
            timingSource = "ffmpeg-benchmark-rtime"
        else:
            # Fallback if a custom ffmpeg build omits the benchmark line.
            elapsedTime = wallTime
            timingSource = "subprocess-wallclock"

        print(
            f"Max Theoretical ({label}): ~{frameCount} frames in "
            f"{elapsedTime:.3f}s ({timingSource}; subprocess wall {wallTime:.3f}s)"
        )

        return {
            "frameCount": frameCount,
            "elapsedTime": elapsedTime,
            "fps": frameCount / elapsedTime if elapsedTime > 0 else 0,
            "timingSource": timingSource,
            "subprocessWallTime": wallTime,
        }
    except Exception as e:
        print(f"Error in Max Theoretical ({label}) decoder: {str(e)}")
        return {
            "error": str(e),
            "frameCount": 0,
            "elapsedTime": 0,
            "fps": 0,
        }


def decodeWithMaxTheoreticalRGB(videoPath: str) -> dict[str, Any]:
    """Theoretical max for the RGB24-target pipeline.

    Includes demux + decode + swscale to RGB24, dropped through the null
    muxer. Comparable against backends that emit RGB24 numpy arrays.
    """
    return _runMaxTheoretical(videoPath, pixFmt="rgb24", label="RGB24")


def decodeWithMaxTheoreticalYUV420(videoPath: str) -> dict[str, Any]:
    """Theoretical max with the source's native chroma format (yuv420p).

    When the input is already yuv420p (h264/web content typically is),
    ffmpeg recognises the request as a no-op and bypasses swscale. This
    is the floor for "demux + decode" cost on this clip — RGB-targeted
    backends cannot beat this number.
    """
    return _runMaxTheoretical(videoPath, pixFmt="yuv420p", label="YUV420p")


if __name__ == "__main__":
    videoPath = r"F:\testVideos\output_bt601.mp4"
    print(decodeWithMaxTheoreticalRGB(videoPath))
    print(decodeWithMaxTheoreticalYUV420(videoPath))
