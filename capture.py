"""Camera capture threads and device enumeration.

Provides CameraThread for real USB cameras and DummyCameraThread for
hardware-free development. Both emit frames, stats, and recording-finished
signals compatible with the MainWindow slots.
"""

import collections
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Common resolutions to probe
RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1920, 1080),
]

# Codec fourcc mapping
CODECS = {
    "FFV1": cv2.VideoWriter_fourcc(*"FFV1"),
    "MJPEG": cv2.VideoWriter_fourcc(*"MJPG"),
    "H.264": cv2.VideoWriter_fourcc(*"avc1"),
}

CODEC_EXTENSIONS = {
    "FFV1": ".avi",
    "MJPEG": ".avi",
    "H.264": ".mp4",
}


def _get_device_names_windows() -> list[str]:
    """Get DirectShow video device names in order using ffmpeg.

    ffmpeg's dshow device listing matches OpenCV's device index ordering
    since both enumerate DirectShow video input devices the same way.
    """
    import re
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=5,
        )
        # ffmpeg prints device list to stderr
        output = result.stderr
        names = []
        for line in output.splitlines():
            # Match lines like: [dshow @ ...] "Device Name" (video)
            m = re.search(r'"(.+?)"\s+\(video\)', line)
            if m:
                names.append(m.group(1))
        return names
    except FileNotFoundError:
        log.debug("ffmpeg not found, cannot enumerate device names")
        return []
    except Exception as e:
        log.debug("Failed to get DirectShow device names: %s", e)
        return []


def enumerate_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """Probe device indices and return list of (index, name) for those that open.

    On Windows, uses DirectShow enumeration for correct device name ordering.
    """
    ds_names = _get_device_names_windows() if _IS_WINDOWS else []
    available = []
    # Suppress OpenCV warnings during device probing
    prev = os.environ.get("OPENCV_LOG_LEVEL", "")
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    try:
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                name = ds_names[i] if i < len(ds_names) else f"Camera {i}"
                available.append((i, name))
                cap.release()
    finally:
        if prev:
            os.environ["OPENCV_LOG_LEVEL"] = prev
        else:
            os.environ.pop("OPENCV_LOG_LEVEL", None)
    return available


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert grayscale frames to BGR so VideoWriter can encode them."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def probe_resolutions(cap: cv2.VideoCapture) -> list[tuple[int, int]]:
    """Try setting common resolutions and return the ones that stick."""
    supported = []
    for w, h in RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) not in supported:
            supported.append((actual_w, actual_h))
    return supported


class CameraThread(QThread):
    """Captures frames from a USB camera in a background thread."""

    frame_ready = pyqtSignal(int, np.ndarray)  # camera_index, filtered frame
    raw_frame_ready = pyqtSignal(int, np.ndarray)  # camera_index, raw frame
    error = pyqtSignal(int, str)  # camera_index, message
    recording_finished = pyqtSignal(int)  # camera_index
    stats_updated = pyqtSignal(int, float, float, float)  # camera_index, fps, bitrate_mbps, avg_write_latency_ms

    def __init__(self, camera_index: int, device_id: int = 0):
        """Initialise the camera thread.

        Args:
            camera_index: Logical index (0 = top, 1 = front).
            device_id: OS device index passed to cv2.VideoCapture.
        """
        super().__init__()
        self.camera_index = camera_index
        self.device_id = device_id
        self._running = False
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._target_fps = 30.0
        self._target_width = 640
        self._target_height = 480
        self._codec = "FFV1"
        self._output_path: Optional[str] = None
        self._duration_seconds: Optional[float] = None
        self._record_start_time: Optional[float] = None
        self._frame_count = 0
        self._frame_timestamps: collections.deque = collections.deque()
        self._last_stats_time = 0.0
        self._last_file_size = 0
        self._write_latencies: collections.deque = collections.deque()
        self._pipeline = None  # Optional FilterPipeline
        self._record_raw = False
        self._raw_writer: Optional[cv2.VideoWriter] = None
        self._raw_output_path: Optional[str] = None

    def set_resolution(self, width: int, height: int):
        """Set the target capture resolution."""
        self._target_width = width
        self._target_height = height

    def set_fps(self, fps: float):
        """Set the target capture frame rate."""
        self._target_fps = fps

    def set_codec(self, codec: str):
        """Set the video codec name (must be a key in CODECS)."""
        self._codec = codec

    def set_pipeline(self, pipeline):
        """Attach a FilterPipeline to apply to each frame."""
        self._pipeline = pipeline

    def set_record_raw(self, enabled: bool, raw_path: Optional[str] = None):
        """Enable recording of raw (unfiltered) frames alongside filtered output."""
        self._record_raw = enabled
        self._raw_output_path = raw_path

    def start_recording(self, output_path: str, duration_seconds: Optional[float] = None):
        """Begin recording frames to disk.

        Args:
            output_path: File path for the output video.
            duration_seconds: Auto-stop after this many seconds, or None for manual stop.
        """
        self._output_path = output_path
        self._duration_seconds = duration_seconds
        self._record_start_time = None
        self._frame_count = 0
        self._frame_timestamps.clear()
        self._write_latencies.clear()
        self._last_stats_time = 0.0
        self._last_file_size = 0
        self._recording = True

    def stop_recording(self):
        """Signal the thread to stop recording after the current frame."""
        self._recording = False

    def stop(self):
        """Signal the thread to exit its run loop."""
        self._running = False
        self._recording = False

    def _emit_stats(self):
        """Compute and emit FPS, bitrate, and write latency stats (~1Hz)."""
        now = time.monotonic()
        if now - self._last_stats_time < 1.0:
            return
        self._last_stats_time = now

        # FPS from rolling window
        fps = 0.0
        if len(self._frame_timestamps) >= 2:
            span = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if span > 0:
                fps = (len(self._frame_timestamps) - 1) / span

        # Bitrate from file size delta
        bitrate = 0.0
        if self._output_path:
            try:
                size = os.path.getsize(self._output_path)
                if self._last_file_size > 0 and self._record_start_time:
                    elapsed = now - self._record_start_time
                    if elapsed > 0:
                        bitrate = (size * 8) / elapsed / 1_000_000  # Mbps
                self._last_file_size = size
            except OSError:
                pass

        # Average write latency
        avg_latency_ms = 0.0
        if self._write_latencies:
            avg_latency_ms = sum(self._write_latencies) / len(self._write_latencies)

        self.stats_updated.emit(self.camera_index, fps, bitrate, avg_latency_ms)

    def run(self):
        """Main capture loop: read frames, apply pipeline, write to disk, pace to target FPS."""
        self._running = True
        cap = cv2.VideoCapture(self.device_id)

        if not cap.isOpened():
            self.error.emit(self.camera_index, f"Cannot open camera {self.device_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        log.info(
            "Camera %d started: %dx%d @ %.1f fps",
            self.camera_index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )

        consecutive_failures = 0
        max_failures_before_reopen = 5
        frame_interval = 1.0 / self._target_fps
        next_frame_time = time.monotonic()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures_before_reopen:
                    self.error.emit(self.camera_index,
                                    f"Reopening camera {self.device_id} after {consecutive_failures} failures")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(self.device_id)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
                        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                    consecutive_failures = 0
                else:
                    time.sleep(0.5)
                next_frame_time = time.monotonic()
                continue
            consecutive_failures = 0

            # Emit raw frame, then apply pipeline for filtered frame
            self.raw_frame_ready.emit(self.camera_index, frame)
            filtered = self._pipeline.apply(frame) if self._pipeline else frame
            self.frame_ready.emit(self.camera_index, filtered)

            if self._recording:
                # Ensure BGR for VideoWriter (pipeline may output grayscale)
                filtered_bgr = _ensure_bgr(filtered)

                if self._writer is None:
                    fourcc = CODECS.get(self._codec, CODECS["FFV1"])
                    h, w = filtered_bgr.shape[:2]
                    self._writer = cv2.VideoWriter(
                        self._output_path, fourcc, self._target_fps, (w, h)
                    )
                    if not self._writer.isOpened():
                        self.error.emit(self.camera_index, "Failed to open VideoWriter")
                        self._recording = False
                        self._writer = None
                        continue
                    self._record_start_time = time.monotonic()
                    # Open raw writer if needed
                    if self._record_raw and self._raw_output_path:
                        rh, rw = frame.shape[:2]
                        self._raw_writer = cv2.VideoWriter(
                            self._raw_output_path, fourcc, self._target_fps, (rw, rh)
                        )

                t0 = time.monotonic()
                self._writer.write(filtered_bgr)
                if self._raw_writer is not None:
                    self._raw_writer.write(frame)
                write_ms = (time.monotonic() - t0) * 1000.0
                self._write_latencies.append(write_ms)
                # Keep only last ~1 second of latency samples
                while len(self._write_latencies) > self._target_fps:
                    self._write_latencies.popleft()
                self._frame_count += 1
                self._frame_timestamps.append(time.monotonic())
                # Keep only last ~1 second of timestamps
                while len(self._frame_timestamps) > 1 and \
                        self._frame_timestamps[-1] - self._frame_timestamps[0] > 1.5:
                    self._frame_timestamps.popleft()

                self._emit_stats()

                if self._duration_seconds and self._record_start_time and \
                        time.monotonic() - self._record_start_time >= self._duration_seconds:
                    self._recording = False
                    self._writer.release()
                    self._writer = None
                    if self._raw_writer is not None:
                        self._raw_writer.release()
                        self._raw_writer = None
                    self.recording_finished.emit(self.camera_index)
            else:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                if self._raw_writer is not None:
                    self._raw_writer.release()
                    self._raw_writer = None

            # Pace the loop to approximate target FPS, accounting for work done
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Falling behind — reset to avoid burst catching up
                next_frame_time = time.monotonic()

        if self._writer is not None:
            self._writer.release()
        if self._raw_writer is not None:
            self._raw_writer.release()
        cap.release()
        log.info("Camera %d stopped", self.camera_index)


class DummyCameraThread(QThread):
    """Generates synthetic test pattern frames for development without hardware."""

    frame_ready = pyqtSignal(int, np.ndarray)
    raw_frame_ready = pyqtSignal(int, np.ndarray)
    error = pyqtSignal(int, str)
    recording_finished = pyqtSignal(int)
    stats_updated = pyqtSignal(int, float, float, float)  # camera_index, fps, bitrate_mbps, avg_write_latency_ms

    def __init__(self, camera_index: int, device_id: int = 0):
        """Initialise the dummy camera thread.

        Args:
            camera_index: Logical index (0 = top, 1 = front).
            device_id: Ignored, accepted for API compatibility.
        """
        super().__init__()
        self.camera_index = camera_index
        self.device_id = device_id
        self._running = False
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._target_fps = 30.0
        self._target_width = 640
        self._target_height = 480
        self._codec = "FFV1"
        self._output_path: Optional[str] = None
        self._duration_seconds: Optional[float] = None
        self._record_start_time: Optional[float] = None
        self._frame_count = 0
        self._frame_timestamps: collections.deque = collections.deque()
        self._last_stats_time = 0.0
        self._last_file_size = 0
        self._write_latencies: collections.deque = collections.deque()
        self._tick = 0
        self._pipeline = None
        self._record_raw = False
        self._raw_writer: Optional[cv2.VideoWriter] = None
        self._raw_output_path: Optional[str] = None

    def set_resolution(self, width: int, height: int):
        """Set the target frame dimensions."""
        self._target_width = width
        self._target_height = height

    def set_fps(self, fps: float):
        """Set the target frame rate."""
        self._target_fps = fps

    def set_codec(self, codec: str):
        """Set the video codec name."""
        self._codec = codec

    def set_pipeline(self, pipeline):
        """Attach a FilterPipeline to apply to each frame."""
        self._pipeline = pipeline

    def set_record_raw(self, enabled: bool, raw_path: Optional[str] = None):
        """Enable recording of raw (unfiltered) frames alongside filtered output."""
        self._record_raw = enabled
        self._raw_output_path = raw_path

    def start_recording(self, output_path: str, duration_seconds: Optional[float] = None):
        """Begin recording generated frames to disk."""
        self._output_path = output_path
        self._duration_seconds = duration_seconds
        self._record_start_time = None
        self._frame_count = 0
        self._frame_timestamps.clear()
        self._write_latencies.clear()
        self._last_stats_time = 0.0
        self._last_file_size = 0
        self._recording = True

    def stop_recording(self):
        """Signal the thread to stop recording."""
        self._recording = False

    def stop(self):
        """Signal the thread to exit its run loop."""
        self._running = False
        self._recording = False

    def _emit_stats(self):
        """Compute and emit FPS, bitrate, and write latency stats (~1Hz)."""
        now = time.monotonic()
        if now - self._last_stats_time < 1.0:
            return
        self._last_stats_time = now

        fps = 0.0
        if len(self._frame_timestamps) >= 2:
            span = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if span > 0:
                fps = (len(self._frame_timestamps) - 1) / span

        bitrate = 0.0
        if self._output_path:
            try:
                size = os.path.getsize(self._output_path)
                if self._last_file_size > 0 and self._record_start_time:
                    elapsed = now - self._record_start_time
                    if elapsed > 0:
                        bitrate = (size * 8) / elapsed / 1_000_000
                self._last_file_size = size
            except OSError:
                pass

        avg_latency_ms = 0.0
        if self._write_latencies:
            avg_latency_ms = sum(self._write_latencies) / len(self._write_latencies)

        self.stats_updated.emit(self.camera_index, fps, bitrate, avg_latency_ms)

    def _generate_frame(self) -> np.ndarray:
        """Generate a synthetic test frame with colored noise and a moving bar."""
        h, w = self._target_height, self._target_width
        # Base colored noise
        frame = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)

        # Tint based on camera index (blue-ish for cam 0, green-ish for cam 1)
        if self.camera_index == 0:
            frame[:, :, 0] += 60  # blue channel
        else:
            frame[:, :, 1] += 60  # green channel

        # Moving horizontal bar
        bar_y = (self._tick * 3) % h
        bar_end = min(bar_y + 20, h)
        frame[bar_y:bar_end, :, :] = 200

        # Label
        label = f"Camera {self.camera_index} [DUMMY]"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self._recording:
            cv2.putText(frame, "REC", (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self._tick += 1
        return frame

    def run(self):
        """Main loop: generate synthetic frames, apply pipeline, write to disk, pace to target FPS."""
        self._running = True
        log.info("Dummy camera %d started: %dx%d", self.camera_index, self._target_width, self._target_height)

        frame_interval = 1.0 / self._target_fps
        next_frame_time = time.monotonic()
        while self._running:
            frame = self._generate_frame()
            self.raw_frame_ready.emit(self.camera_index, frame)
            filtered = self._pipeline.apply(frame) if self._pipeline else frame
            self.frame_ready.emit(self.camera_index, filtered)

            if self._recording:
                filtered_bgr = _ensure_bgr(filtered)

                if self._writer is None:
                    fourcc = CODECS.get(self._codec, CODECS["FFV1"])
                    h, w = filtered_bgr.shape[:2]
                    self._writer = cv2.VideoWriter(
                        self._output_path, fourcc, self._target_fps, (w, h)
                    )
                    self._record_start_time = time.monotonic()
                    if self._record_raw and self._raw_output_path:
                        rh, rw = frame.shape[:2]
                        self._raw_writer = cv2.VideoWriter(
                            self._raw_output_path, fourcc, self._target_fps, (rw, rh)
                        )

                t0 = time.monotonic()
                self._writer.write(filtered_bgr)
                if self._raw_writer is not None:
                    self._raw_writer.write(frame)
                write_ms = (time.monotonic() - t0) * 1000.0
                self._write_latencies.append(write_ms)
                while len(self._write_latencies) > self._target_fps:
                    self._write_latencies.popleft()
                self._frame_count += 1
                self._frame_timestamps.append(time.monotonic())
                while len(self._frame_timestamps) > 1 and \
                        self._frame_timestamps[-1] - self._frame_timestamps[0] > 1.5:
                    self._frame_timestamps.popleft()

                self._emit_stats()

                if self._duration_seconds and self._record_start_time and \
                        time.monotonic() - self._record_start_time >= self._duration_seconds:
                    self._recording = False
                    self._writer.release()
                    self._writer = None
                    if self._raw_writer is not None:
                        self._raw_writer.release()
                        self._raw_writer = None
                    self.recording_finished.emit(self.camera_index)
            else:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                if self._raw_writer is not None:
                    self._raw_writer.release()
                    self._raw_writer = None

            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.monotonic()

        if self._writer is not None:
            self._writer.release()
        if self._raw_writer is not None:
            self._raw_writer.release()
        log.info("Dummy camera %d stopped", self.camera_index)
