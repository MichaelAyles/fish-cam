"""Main GUI window for Fish-On.

Provides dual-camera preview, recording controls, serial relay pump control,
and an events CSV sidecar log for each capture session.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import psutil
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from capture import (
    CODEC_EXTENSIONS,
    CODECS,
    RESOLUTIONS,
    CameraThread,
    DummyCameraThread,
    enumerate_cameras,
)
from config import load_config, save_config
from pipeline import PipelineManager
from pipeline_editor import PipelineEditorDialog
from relay import MockRelayController, RelayController, scan_ports

log = logging.getLogger(__name__)

_CAM_LABELS = {0: "Top View", 1: "Side View"}
_CAM_FILE_TAGS = {0: "top", 1: "front"}


def frame_to_pixmap(frame: np.ndarray, target_width: int = 480, target_height: int = 360) -> QPixmap:
    """Convert an OpenCV BGR or grayscale frame to a QPixmap, preserving aspect ratio."""
    if frame.ndim == 2:
        h, w = frame.shape
        qimg = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        target_width, target_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class MainWindow(QMainWindow):
    """Primary application window with dual-camera preview and recording controls."""

    def __init__(self, dummy: bool = False):
        """Initialise the main window, cameras, and relay.

        Args:
            dummy: If True, use synthetic cameras and a mock relay.
        """
        super().__init__()
        self.dummy = dummy
        self.setWindowTitle("Fish-On" + (" [DUMMY]" if dummy else ""))
        self.setMinimumSize(1100, 700)

        self._cfg = load_config()
        self._recording = False
        self._record_start: Optional[float] = None  # monotonic
        self._record_start_utc: Optional[datetime] = None  # wall clock
        self._record_out_dir: Optional[Path] = None
        self._cam_stats: dict[int, tuple[float, float, float]] = {}  # camera_index -> (fps, bitrate, latency_ms)
        self._stats_log: list[dict] = []
        self._last_system_stats_time = 0.0
        self._pump_auto_enabled = False
        self._pump_on_time = 0.0  # seconds after capture start
        self._pump_off_time = 0.0
        self._pump_triggered_on = False
        self._pump_triggered_off = False
        self._output_dir = self._cfg.get("output_dir", str(Path.home() / "FishOn"))
        self._latest_raw_frames: dict[int, np.ndarray] = {}
        self._latest_filtered_frames: dict[int, np.ndarray] = {}
        # Keep old name for backward compat in _on_tick
        self._latest_frames: dict[int, np.ndarray] = {}

        # Pipeline manager
        self._pipeline_manager = PipelineManager()
        saved_pipeline = self._cfg.get("pipeline")
        if saved_pipeline:
            try:
                self._pipeline_manager.load_dict(saved_pipeline)
            except Exception as e:
                log.warning("Failed to restore pipeline: %s", e)

        # Events CSV sidecar
        self._event_csv_file = None
        self._event_csv_writer = None

        # Relay
        self._relay: Union[RelayController, MockRelayController, None] = None
        if dummy:
            self._relay = MockRelayController()
            self._relay.open()

        # Available camera devices: list of (index, name)
        self._available_devices = [(0, "Dummy 0"), (1, "Dummy 1")] if dummy else enumerate_cameras()

        # Camera threads (created after UI so combos exist)
        self._cam0 = None
        self._cam1 = None

        self._build_ui()

        # Auto-connect relay if a usbserial port is selected
        if not dummy and "usbserial" in self._port_combo.currentText().lower():
            self._connect_relay()

        # Start cameras with initial combo selections
        self._start_camera(0)
        self._start_camera(1)

        # Preview + status timer at ~15fps
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(66)

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        """Build the main window layout with previews, controls, and status bar."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: 2x2 camera preview grid
        # Row 0: Top Raw (left) | Top Filtered (right)
        # Row 1: Side Raw (left) | Side Filtered (right)
        preview_grid = QGridLayout()
        preview_grid.addWidget(QLabel("Top Raw"), 0, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        preview_grid.addWidget(QLabel("Top Filtered"), 0, 1, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        preview_grid.addWidget(QLabel("Side Raw"), 2, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        preview_grid.addWidget(QLabel("Side Filtered"), 2, 1, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self._preview_raw0 = QLabel("Top Raw: waiting...")
        self._preview_raw1 = QLabel("Side Raw: waiting...")
        self._preview_filt0 = QLabel("Top Filtered: waiting...")
        self._preview_filt1 = QLabel("Side Filtered: waiting...")
        all_previews = [self._preview_raw0, self._preview_raw1,
                        self._preview_filt0, self._preview_filt1]
        for lbl in all_previews:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(320, 240)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            lbl.setStyleSheet("background-color: #222; color: #aaa;")
        preview_grid.addWidget(self._preview_raw0, 1, 0)
        preview_grid.addWidget(self._preview_filt0, 1, 1)
        preview_grid.addWidget(self._preview_raw1, 3, 0)
        preview_grid.addWidget(self._preview_filt1, 3, 1)
        root.addLayout(preview_grid, stretch=3)

        # Right: controls panel
        controls = QVBoxLayout()
        controls.addWidget(self._build_camera_group())
        controls.addWidget(self._build_capture_group())
        controls.addWidget(self._build_pump_group())
        controls.addWidget(self._build_output_group())
        self._pipeline_btn = QPushButton("Edit Pipeline...")
        self._pipeline_btn.clicked.connect(self._show_pipeline_editor)
        controls.addWidget(self._pipeline_btn)
        controls.addStretch()
        root.addLayout(controls, stretch=1)

        # Pipeline editor popup (non-modal dialog)
        self._pipeline_dialog = PipelineEditorDialog(self._pipeline_manager, self)
        self._pipeline_dialog.editor.pipeline_changed.connect(self._on_pipeline_changed)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_recording = QLabel("Idle")
        self._status_elapsed = QLabel("00:00.0")
        self._status_pump = QLabel("Pump: OFF")
        self._status_pump_countdown = QLabel("")
        self._status_disk = QLabel("")
        self._status_stats = QLabel("")
        self._status_system = QLabel("")
        self._status_bar.addWidget(self._status_recording)
        self._status_bar.addWidget(self._status_elapsed)
        self._status_bar.addWidget(self._status_stats)
        self._status_bar.addWidget(self._status_pump)
        self._status_bar.addWidget(self._status_pump_countdown)
        self._status_bar.addPermanentWidget(self._status_system)
        self._status_bar.addPermanentWidget(self._status_disk)

    def _build_camera_group(self) -> QGroupBox:
        """Build the camera device selection group."""
        group = QGroupBox("Cameras")
        layout = QVBoxLayout(group)

        # Camera 0 device selector
        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Top View:"))
        self._cam0_combo = QComboBox()
        self._populate_cam_combo(self._cam0_combo, default_index=self._cfg.get("cam0_device", 0))
        self._cam0_combo.currentIndexChanged.connect(lambda: self._on_cam_device_changed(0))
        row0.addWidget(self._cam0_combo)
        layout.addLayout(row0)

        # Camera 1 device selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Side View:"))
        self._cam1_combo = QComboBox()
        self._populate_cam_combo(self._cam1_combo, default_index=self._cfg.get("cam1_device", 1))
        self._cam1_combo.currentIndexChanged.connect(lambda: self._on_cam_device_changed(1))
        row1.addWidget(self._cam1_combo)
        layout.addLayout(row1)

        # Rescan button
        rescan_btn = QPushButton("Rescan Cameras")
        rescan_btn.clicked.connect(self._rescan_cameras)
        layout.addWidget(rescan_btn)

        return group

    def _populate_cam_combo(self, combo: QComboBox, default_index: int = 0):
        """Populate a camera combo box with available devices."""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None", -1)
        for dev_id, dev_name in self._available_devices:
            combo.addItem(f"{dev_id}: {dev_name}", dev_id)
        # Select the device matching default_index if available
        for i in range(combo.count()):
            if combo.itemData(i) == default_index:
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)

    def _rescan_cameras(self):
        """Re-enumerate available camera devices and refresh combo boxes."""
        self._available_devices = [(0, "Dummy 0"), (1, "Dummy 1")] if self.dummy else enumerate_cameras()
        old0 = self._cam0_combo.currentData()
        old1 = self._cam1_combo.currentData()
        self._populate_cam_combo(self._cam0_combo, old0 if old0 is not None else 0)
        self._populate_cam_combo(self._cam1_combo, old1 if old1 is not None else 1)

    def _start_camera(self, camera_index: int):
        """Start a camera thread for the given camera index."""
        combo = self._cam0_combo if camera_index == 0 else self._cam1_combo
        device_id = combo.currentData()
        if device_id is None or device_id < 0:
            return

        tag = _CAM_FILE_TAGS.get(camera_index, "top")
        CamClass = DummyCameraThread if self.dummy else CameraThread
        cam = CamClass(camera_index=camera_index, device_id=device_id)
        cam.raw_frame_ready.connect(self._on_raw_frame)
        cam.frame_ready.connect(self._on_filtered_frame)
        cam.error.connect(self._on_camera_error)
        cam.recording_finished.connect(self._on_recording_finished)
        cam.stats_updated.connect(self._on_stats_updated)
        cam.set_pipeline(self._pipeline_manager.get_pipeline(tag))

        if camera_index == 0:
            self._cam0 = cam
        else:
            self._cam1 = cam
        cam.start()

    def _stop_camera(self, camera_index: int):
        """Stop a camera thread and clear its preview."""
        cam = self._cam0 if camera_index == 0 else self._cam1
        if cam is not None:
            cam.stop()
            cam.wait(2000)
        if camera_index == 0:
            self._cam0 = None
        else:
            self._cam1 = None
        self._latest_frames.pop(camera_index, None)
        self._latest_raw_frames.pop(camera_index, None)
        self._latest_filtered_frames.pop(camera_index, None)
        raw_lbl = self._preview_raw0 if camera_index == 0 else self._preview_raw1
        filt_lbl = self._preview_filt0 if camera_index == 0 else self._preview_filt1
        for lbl in (raw_lbl, filt_lbl):
            lbl.clear()
            lbl.setText(f"{_CAM_LABELS.get(camera_index, f'Camera {camera_index}')}: no device")

    def _on_cam_device_changed(self, camera_index: int):
        """Handle camera device combo box change."""
        if self._recording:
            return
        self._stop_camera(camera_index)
        combo = self._cam0_combo if camera_index == 0 else self._cam1_combo
        device_id = combo.currentData()
        if device_id is not None and device_id >= 0:
            self._start_camera(camera_index)

    def _build_capture_group(self) -> QGroupBox:
        """Build the capture settings group (prefix, duration, resolution, fps, codec)."""
        group = QGroupBox("Capture")
        layout = QVBoxLayout(group)

        # Video prefix
        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Prefix:"))
        self._prefix_input = QLineEdit(self._cfg.get("video_prefix", ""))
        self._prefix_input.setPlaceholderText("(timestamp)")
        self._prefix_input.setToolTip("Video filename prefix. Leave blank to use timestamp.")
        prefix_row.addWidget(self._prefix_input)
        layout.addLayout(prefix_row)

        # Duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duration (mm:ss):"))
        self._duration_input = QLineEdit(self._cfg.get("duration", "05:00"))
        self._duration_input.setMaximumWidth(80)
        dur_row.addWidget(self._duration_input)
        layout.addLayout(dur_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self._res_combo = QComboBox()
        for w, h in RESOLUTIONS:
            self._res_combo.addItem(f"{w}x{h}", (w, h))
        # Restore saved resolution
        saved_res = self._cfg.get("resolution", "1280x720")
        idx = self._res_combo.findText(saved_res)
        self._res_combo.setCurrentIndex(idx if idx >= 0 else 0)
        res_row.addWidget(self._res_combo)
        layout.addLayout(res_row)

        # FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self._fps_combo = QComboBox()
        for fps in [15, 24, 25, 30, 60]:
            self._fps_combo.addItem(str(fps), fps)
        saved_fps = self._cfg.get("fps", 30)
        idx = self._fps_combo.findData(saved_fps)
        self._fps_combo.setCurrentIndex(idx if idx >= 0 else 3)
        fps_row.addWidget(self._fps_combo)
        layout.addLayout(fps_row)

        # Codec
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("Codec:"))
        self._codec_combo = QComboBox()
        for name in CODECS:
            self._codec_combo.addItem(name)
        saved_codec = self._cfg.get("codec", "MJPEG")
        idx = self._codec_combo.findText(saved_codec)
        if idx >= 0:
            self._codec_combo.setCurrentIndex(idx)
        codec_row.addWidget(self._codec_combo)
        layout.addLayout(codec_row)

        # Start/Stop button
        self._record_btn = QPushButton("Start Recording")
        self._record_btn.clicked.connect(self._toggle_recording)
        self._record_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        layout.addWidget(self._record_btn)

        return group

    def _build_pump_group(self) -> QGroupBox:
        """Build the pump control group (serial port, auto timing, manual buttons)."""
        group = QGroupBox("Pump Control")
        layout = QVBoxLayout(group)

        # Serial port selection
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self._port_combo = QComboBox()
        self._refresh_ports()
        port_row.addWidget(self._port_combo)
        rescan_btn = QPushButton("Rescan")
        rescan_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(rescan_btn)
        layout.addLayout(port_row)

        # Connect / Test
        connect_row = QHBoxLayout()
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._connect_relay)
        connect_row.addWidget(self._connect_btn)
        self._test_btn = QPushButton("Test Relay")
        self._test_btn.clicked.connect(self._test_relay)
        connect_row.addWidget(self._test_btn)
        layout.addLayout(connect_row)

        # Manual on/off
        manual_row = QHBoxLayout()
        self._pump_on_btn = QPushButton("Pump ON")
        self._pump_on_btn.clicked.connect(self._pump_on)
        manual_row.addWidget(self._pump_on_btn)
        self._pump_off_btn = QPushButton("Pump OFF")
        self._pump_off_btn.clicked.connect(self._pump_off)
        manual_row.addWidget(self._pump_off_btn)
        layout.addLayout(manual_row)

        # Auto timing
        self._pump_auto_check = QCheckBox("Auto pump during capture")
        self._pump_auto_check.setChecked(self._cfg.get("pump_auto", True))
        layout.addWidget(self._pump_auto_check)

        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("ON at (mm:ss):"))
        self._pump_on_input = QLineEdit(str(self._cfg.get("pump_on_time", "02:00")))
        self._pump_on_input.setMaximumWidth(60)
        auto_row.addWidget(self._pump_on_input)
        auto_row.addWidget(QLabel("OFF at (mm:ss):"))
        self._pump_off_input = QLineEdit(str(self._cfg.get("pump_off_time", "04:00")))
        self._pump_off_input.setMaximumWidth(60)
        auto_row.addWidget(self._pump_off_input)
        layout.addLayout(auto_row)

        # State indicator
        self._pump_indicator = QLabel("Relay: disconnected")
        layout.addWidget(self._pump_indicator)

        return group

    def _build_output_group(self) -> QGroupBox:
        """Build the output directory selection group."""
        group = QGroupBox("Output")
        layout = QVBoxLayout(group)
        self._output_label = QLabel(self._output_dir)
        self._output_label.setWordWrap(True)
        layout.addWidget(self._output_label)
        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._pick_output_dir)
        btn_row.addWidget(browse_btn)
        open_btn = QPushButton("Open Folder")
        open_btn.clicked.connect(self._open_output_dir)
        btn_row.addWidget(open_btn)
        layout.addLayout(btn_row)
        return group

    # ── Slots ────────────────────────────────────────────────────────

    def _on_raw_frame(self, camera_index: int, frame: np.ndarray):
        """Store the latest raw (unfiltered) frame."""
        self._latest_raw_frames[camera_index] = frame

    def _on_filtered_frame(self, camera_index: int, frame: np.ndarray):
        """Store the latest filtered frame for preview rendering."""
        self._latest_filtered_frames[camera_index] = frame
        self._latest_frames[camera_index] = frame

    def _show_pipeline_editor(self):
        """Show the pipeline editor popup window."""
        self._pipeline_dialog.show()
        self._pipeline_dialog.raise_()
        self._pipeline_dialog.activateWindow()

    def _on_pipeline_changed(self):
        """Handle pipeline editor changes — warn if recording."""
        if self._recording:
            self._pipeline_dialog.editor.show_warning(
                "Pipeline changed during recording — filtered output will reflect the new settings."
            )
        # Re-inject pipelines into camera threads
        for cam_idx, cam in [(0, self._cam0), (1, self._cam1)]:
            if cam is not None:
                tag = _CAM_FILE_TAGS.get(cam_idx, "top")
                cam.set_pipeline(self._pipeline_manager.get_pipeline(tag))

    def _on_stats_updated(self, camera_index: int, fps: float, bitrate: float, latency_ms: float):
        """Update the status bar with per-camera recording statistics."""
        self._cam_stats[camera_index] = (fps, bitrate, latency_ms)
        parts = []
        for idx in sorted(self._cam_stats):
            f, b, lat = self._cam_stats[idx]
            tag = _CAM_FILE_TAGS.get(idx, f"cam{idx}")
            parts.append(f"{tag}: {f:.1f}fps {b:.1f}Mb/s {lat:.1f}ms")
        self._status_stats.setText(" | ".join(parts))

        # Accumulate for stats log
        if self._recording:
            elapsed = time.monotonic() - self._record_start if self._record_start else 0

            abs_utc = (self._record_start_utc + timedelta(seconds=elapsed)).isoformat(timespec="milliseconds") + "Z" \
                if self._record_start_utc else datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            self._stats_log.append({
                "datetime_utc": abs_utc,
                "time_s": round(elapsed, 1),
                "camera": camera_index,
                "fps": round(fps, 2),
                "bitrate_mbps": round(bitrate, 2),
                "write_latency_ms": round(latency_ms, 2),
            })

    def _on_camera_error(self, camera_index: int, message: str):
        """Log camera errors with the human-readable camera label."""
        cam_label = _CAM_LABELS.get(camera_index, f"Camera {camera_index}")
        log.warning("%s error: %s", cam_label, message)

    def _on_recording_finished(self, camera_index: int):
        """Handle a camera finishing its recording duration."""
        log.info("%s recording finished", _CAM_LABELS.get(camera_index, f"Camera {camera_index}"))
        # If both cameras are done (or the other wasn't recording), stop
        cam0_done = self._cam0 is None or not self._cam0._recording
        cam1_done = self._cam1 is None or not self._cam1._recording
        if cam0_done and cam1_done:
            self._stop_recording()

    def _on_tick(self):
        """Called at ~15fps for preview updates, elapsed time, pump countdown, and auto-trigger."""
        # Update previews (raw + filtered for each camera)
        for idx, raw_lbl, filt_lbl in [
            (0, self._preview_raw0, self._preview_filt0),
            (1, self._preview_raw1, self._preview_filt1),
        ]:
            if idx in self._latest_raw_frames:
                pm = frame_to_pixmap(self._latest_raw_frames[idx], raw_lbl.width(), raw_lbl.height())
                raw_lbl.setPixmap(pm)
            if idx in self._latest_filtered_frames:
                pm = frame_to_pixmap(self._latest_filtered_frames[idx], filt_lbl.width(), filt_lbl.height())
                filt_lbl.setPixmap(pm)

        # Update elapsed time
        if self._recording and self._record_start:
            elapsed = time.monotonic() - self._record_start
            mins = int(elapsed) // 60
            secs = elapsed % 60
            self._status_elapsed.setText(f"{mins:02d}:{secs:04.1f}")

            # Auto pump logic
            if self._pump_auto_enabled and self._relay:
                if not self._pump_triggered_on and elapsed >= self._pump_on_time:
                    self._relay.send_on()
                    self._pump_triggered_on = True
                    self._log_event("pump_on")
                    self._update_pump_indicator()
                if not self._pump_triggered_off and elapsed >= self._pump_off_time:
                    self._relay.send_off()
                    self._pump_triggered_off = True
                    self._log_event("pump_off")
                    self._update_pump_indicator()

            # Pump countdown in status bar
            self._update_pump_countdown(elapsed)
        else:
            self._status_pump_countdown.setText("")

        # Update pump indicator
        self._update_pump_indicator()

        # Update system stats (~1s interval)
        now = time.monotonic()
        if now - self._last_system_stats_time >= 1.0:
            self._last_system_stats_time = now
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            self._status_system.setText(f"CPU {cpu:.0f}% | RAM {ram:.0f}%")

        # Update disk space
        try:
            usage = shutil.disk_usage(self._output_dir)
            free_gb = usage.free / (1024 ** 3)
            self._status_disk.setText(f"Disk free: {free_gb:.1f} GB")
        except Exception:
            self._status_disk.setText("")

    def _update_pump_countdown(self, elapsed: float):
        """Update the pump countdown label in the status bar."""
        if not self._pump_auto_enabled:
            self._status_pump_countdown.setText("")
            return

        if not self._pump_triggered_on:
            remaining = self._pump_on_time - elapsed
            if remaining > 0:
                m, s = divmod(int(remaining), 60)
                self._status_pump_countdown.setText(f"Pump ON in {m:02d}:{s:02d}")
                return

        if not self._pump_triggered_off:
            remaining = self._pump_off_time - elapsed
            if remaining > 0:
                m, s = divmod(int(remaining), 60)
                self._status_pump_countdown.setText(f"Pump OFF in {m:02d}:{s:02d}")
                return

        self._status_pump_countdown.setText("")

    def _toggle_recording(self):
        """Toggle between starting and stopping a recording session."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Validate settings, open output files, and begin recording on all active cameras."""
        # Parse duration
        duration_secs = self._parse_duration(self._duration_input.text())
        if duration_secs is None:
            self._status_recording.setText("Invalid duration")
            return

        # Disk space warning
        try:
            usage = shutil.disk_usage(self._output_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 10:
                reply = QMessageBox.warning(
                    self, "Low Disk Space",
                    f"Only {free_gb:.1f} GB free. Continue recording?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
        except Exception:
            pass

        # Get settings
        res = self._res_combo.currentData()
        fps = self._fps_combo.currentData()
        codec = self._codec_combo.currentText()

        # Apply settings to cameras
        for cam in (self._cam0, self._cam1):
            if cam is not None:
                cam.set_resolution(res[0], res[1])
                cam.set_fps(fps)
                cam.set_codec(codec)

        # Build folder and file prefix
        ext = CODEC_EXTENSIONS.get(codec, ".avi")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self._prefix_input.text().strip()
        folder_name = f"{prefix}_{timestamp}" if prefix else timestamp
        out_dir = Path(self._output_dir) / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        self._record_out_dir = out_dir
        self._stats_log = []

        file_prefix = f"{prefix}_" if prefix else ""

        # Open events CSV
        self._open_event_csv(out_dir, file_prefix)

        # Save pipeline sidecar JSON
        try:
            self._pipeline_manager.save_json(str(out_dir / f"{file_prefix}pipeline.json"))
        except Exception as e:
            log.error("Failed to save pipeline sidecar: %s", e)

        # Reset stateful filters (e.g. background subtractors) for a clean capture
        self._pipeline_manager.reset_all()

        # Start recording on active cameras
        # If pipeline has enabled filters: record both raw and filtered
        # If no filters: record raw only
        for cam, tag in [(self._cam0, "top"), (self._cam1, "side")]:
            if cam is not None:
                pipeline = self._pipeline_manager.get_pipeline(tag)
                if pipeline.has_enabled_filters():
                    raw_path = str(out_dir / f"{file_prefix}{tag}_raw{ext}")
                    cam.set_record_raw(True, raw_path)
                    cam.start_recording(str(out_dir / f"{file_prefix}{tag}_filt{ext}"), duration_secs)
                else:
                    cam.set_record_raw(False)
                    cam.start_recording(str(out_dir / f"{file_prefix}{tag}_raw{ext}"), duration_secs)

        self._recording = True
        self._record_start = time.monotonic()
        self._record_start_utc = datetime.utcnow()
        self._record_btn.setText("Stop Recording")
        self._record_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #c0392b; color: white;")
        self._status_recording.setText("Recording")

        self._log_event("capture_start")

        # Pump auto setup
        self._pump_auto_enabled = self._pump_auto_check.isChecked()
        self._pump_triggered_on = False
        self._pump_triggered_off = False
        on_secs = self._parse_duration(self._pump_on_input.text())
        off_secs = self._parse_duration(self._pump_off_input.text())
        if on_secs is not None and off_secs is not None:
            self._pump_on_time = on_secs
            self._pump_off_time = off_secs
        else:
            self._pump_auto_enabled = False

        # Disable controls during recording
        self._set_controls_enabled(False)

    def _stop_recording(self):
        """Stop all cameras, close the events CSV, and write the stats log."""
        self._log_event("capture_stop")

        if self._cam0 is not None:
            self._cam0.stop_recording()
        if self._cam1 is not None:
            self._cam1.stop_recording()

        # Close events CSV
        self._close_event_csv()

        # Write stats log
        self._write_stats_log()

        self._recording = False
        self._record_start = None
        self._record_start_utc = None
        self._record_out_dir = None
        self._cam_stats.clear()
        self._status_stats.setText("")
        self._record_btn.setText("Start Recording")
        self._record_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self._status_recording.setText("Idle")
        self._status_elapsed.setText("00:00.0")
        self._status_pump_countdown.setText("")

        # Turn off pump if auto was enabled
        if self._pump_auto_enabled and self._relay:
            self._relay.send_off()
            self._pump_auto_enabled = False

        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable recording-related controls."""
        self._prefix_input.setEnabled(enabled)
        self._duration_input.setEnabled(enabled)
        self._res_combo.setEnabled(enabled)
        self._fps_combo.setEnabled(enabled)
        self._codec_combo.setEnabled(enabled)
        self._cam0_combo.setEnabled(enabled)
        self._cam1_combo.setEnabled(enabled)

    def _parse_duration(self, text: str) -> Optional[float]:
        """Parse mm:ss or plain seconds into a float."""
        text = text.strip()
        try:
            if ":" in text:
                parts = text.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            return float(text)
        except (ValueError, IndexError):
            return None

    # ── Events CSV ──────────────────────────────────────────────────

    def _open_event_csv(self, out_dir: Path, file_prefix: str):
        """Open the events CSV sidecar file for writing."""
        try:
            csv_path = out_dir / f"{file_prefix}events.csv"
            self._event_csv_file = open(csv_path, "w", newline="")
            self._event_csv_writer = csv.writer(self._event_csv_file)
            self._event_csv_writer.writerow(["datetime_utc", "timestamp_ms", "frame_top", "frame_front", "event"])
        except Exception as e:
            log.error("Failed to open events CSV: %s", e)
            self._event_csv_file = None
            self._event_csv_writer = None

    def _close_event_csv(self):
        """Flush and close the events CSV file."""
        if self._event_csv_file:
            try:
                self._event_csv_file.close()
            except Exception as e:
                log.error("Failed to close events CSV: %s", e)
        self._event_csv_file = None
        self._event_csv_writer = None

    def _log_event(self, event: str):
        """Write a row to the events CSV with current timestamp and frame counts."""
        if not self._event_csv_writer:
            return
        elapsed_ms = 0
        if self._record_start:
            elapsed_ms = int((time.monotonic() - self._record_start) * 1000)
        from datetime import timedelta
        abs_utc = (self._record_start_utc + timedelta(milliseconds=elapsed_ms)).isoformat(timespec="milliseconds") + "Z" \
            if self._record_start_utc else datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        frame_top = self._cam0._frame_count if self._cam0 else 0
        frame_front = self._cam1._frame_count if self._cam1 else 0
        try:
            self._event_csv_writer.writerow([abs_utc, elapsed_ms, frame_top, frame_front, event])
            self._event_csv_file.flush()
        except Exception as e:
            log.error("Failed to write event: %s", e)

    # ── Relay ────────────────────────────────────────────────────────

    def _refresh_ports(self):
        """Refresh the serial port combo box with available ports."""
        self._port_combo.clear()
        if self.dummy:
            self._port_combo.addItem("MOCK")
        else:
            for port in scan_ports():
                self._port_combo.addItem(port)
            # Auto-select first usbserial port
            for i in range(self._port_combo.count()):
                if "usbserial" in self._port_combo.itemText(i).lower():
                    self._port_combo.setCurrentIndex(i)
                    break

    def _connect_relay(self):
        """Open a serial connection to the selected relay port."""
        if self.dummy:
            self._pump_indicator.setText("Relay: connected (mock)")
            return

        port = self._port_combo.currentText()
        if not port:
            return

        try:
            if self._relay:
                self._relay.close()
            self._relay = RelayController(port)
            self._relay.open()
            self._pump_indicator.setText(f"Relay: connected ({port})")
        except Exception as e:
            self._pump_indicator.setText(f"Relay error: {e}")
            log.error("Relay connect failed: %s", e)

    def _test_relay(self):
        """Run a quick on/off relay test cycle."""
        if self._relay:
            self._relay.test_relay()

    def _pump_on(self):
        """Manually turn the pump relay on."""
        if self._relay:
            self._relay.send_on()
            if self._recording:
                self._log_event("pump_on")
            self._update_pump_indicator()

    def _pump_off(self):
        """Manually turn the pump relay off."""
        if self._relay:
            self._relay.send_off()
            if self._recording:
                self._log_event("pump_off")
            self._update_pump_indicator()

    def _update_pump_indicator(self):
        """Update the pump state indicator in both the sidebar and status bar."""
        if self._relay:
            state = "ON" if self._relay.is_on else "OFF"
            self._status_pump.setText(f"Pump: {state}")
            self._pump_indicator.setText(f"Relay: {state}")
        else:
            self._status_pump.setText("Pump: N/A")

    # ── Output ───────────────────────────────────────────────────────

    def _pick_output_dir(self):
        """Open a directory picker to choose the output folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self._output_dir)
        if path:
            self._output_dir = path
            self._output_label.setText(path)

    def _open_output_dir(self):
        """Open the output directory in the system file manager."""
        path = self._output_dir
        Path(path).mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ── Stats Log ────────────────────────────────────────────────────

    def _write_stats_log(self):
        """Write accumulated per-second stats to a JSONL file."""
        if not self._record_out_dir or not self._stats_log:
            return
        log_path = self._record_out_dir / "capture_stats.jsonl"
        try:
            with open(log_path, "w") as f:
                # First line: session metadata
                meta = {
                    "type": "session",
                    "timestamp": datetime.now().isoformat(),
                    "duration_input": self._duration_input.text(),
                    "resolution": self._res_combo.currentText(),
                    "fps": self._fps_combo.currentData(),
                    "codec": self._codec_combo.currentText(),
                    "output_dir": str(self._record_out_dir),
                    "pump_auto": self._pump_auto_check.isChecked(),
                    "pump_on_time": self._pump_on_input.text(),
                    "pump_off_time": self._pump_off_input.text(),
                    "pipeline": self._pipeline_manager.to_dict(),
                }
                f.write(json.dumps(meta) + "\n")
                # One line per stats sample
                for sample in self._stats_log:
                    f.write(json.dumps(sample) + "\n")
            log.info("Stats log written to %s (%d samples)", log_path, len(self._stats_log))
        except Exception as e:
            log.error("Failed to write stats log: %s", e)

    # ── Config ───────────────────────────────────────────────────────

    def _save_current_config(self):
        """Persist all current settings to the config file."""
        self._cfg.update({
            "output_dir": self._output_dir,
            "video_prefix": self._prefix_input.text().strip(),
            "duration": self._duration_input.text(),
            "resolution": self._res_combo.currentText(),
            "fps": self._fps_combo.currentData(),
            "codec": self._codec_combo.currentText(),
            "cam0_device": self._cam0_combo.currentData(),
            "cam1_device": self._cam1_combo.currentData(),
            "pump_auto": self._pump_auto_check.isChecked(),
            "pump_on_time": self._pump_on_input.text(),
            "pump_off_time": self._pump_off_input.text(),
            "pump_port": self._port_combo.currentText(),
            "pipeline": self._pipeline_manager.to_dict(),
        })
        save_config(self._cfg)

    # ── Cleanup ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Save config, stop cameras, and close the relay on exit."""
        self._save_current_config()
        self._close_event_csv()
        for cam in (self._cam0, self._cam1):
            if cam is not None:
                cam.stop()
        for cam in (self._cam0, self._cam1):
            if cam is not None:
                cam.wait(2000)
        if self._relay:
            self._relay.close()
        event.accept()
