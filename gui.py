import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
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
from relay import MockRelayController, RelayController, scan_ports

log = logging.getLogger(__name__)


def frame_to_pixmap(frame: np.ndarray, target_width: int = 480, target_height: int = 360) -> QPixmap:
    """Convert an OpenCV BGR frame to a QPixmap that fits within target bounds, preserving aspect ratio."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        target_width, target_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class MainWindow(QMainWindow):
    def __init__(self, dummy: bool = False):
        super().__init__()
        self.dummy = dummy
        self.setWindowTitle("Fish Capture" + (" [DUMMY]" if dummy else ""))
        self.setMinimumSize(1100, 700)

        self._cfg = load_config()
        self._recording = False
        self._record_start: Optional[float] = None
        self._record_out_dir: Optional[Path] = None
        self._cam_stats: dict[int, tuple[float, float, float]] = {}  # camera_index -> (fps, bitrate, latency_ms)
        self._stats_log: list[dict] = []
        self._last_system_stats_time = 0.0
        self._pump_auto_enabled = False
        self._pump_on_time = 0.0  # seconds after capture start
        self._pump_off_time = 0.0
        self._pump_triggered_on = False
        self._pump_triggered_off = False
        self._output_dir = self._cfg.get("output_dir", str(Path.home() / "fish-capture"))
        self._latest_frames: dict[int, np.ndarray] = {}

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
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: camera previews stacked vertically
        preview_layout = QVBoxLayout()
        self._preview0 = QLabel("Camera 0: waiting...")
        self._preview1 = QLabel("Camera 1: waiting...")
        for lbl in (self._preview0, self._preview1):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(480, 360)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            lbl.setStyleSheet("background-color: #222; color: #aaa;")
        preview_layout.addWidget(self._preview0)
        preview_layout.addWidget(self._preview1)
        root.addLayout(preview_layout, stretch=3)

        # Right: controls panel
        controls = QVBoxLayout()
        controls.addWidget(self._build_camera_group())
        controls.addWidget(self._build_capture_group())
        controls.addWidget(self._build_pump_group())
        controls.addWidget(self._build_output_group())
        controls.addWidget(self._build_pipeline_group())
        controls.addStretch()
        root.addLayout(controls, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_recording = QLabel("Idle")
        self._status_elapsed = QLabel("00:00.0")
        self._status_pump = QLabel("Pump: OFF")
        self._status_disk = QLabel("")
        self._status_stats = QLabel("")
        self._status_system = QLabel("")
        self._status_bar.addWidget(self._status_recording)
        self._status_bar.addWidget(self._status_elapsed)
        self._status_bar.addWidget(self._status_stats)
        self._status_bar.addWidget(self._status_pump)
        self._status_bar.addPermanentWidget(self._status_system)
        self._status_bar.addPermanentWidget(self._status_disk)

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Cameras")
        layout = QVBoxLayout(group)

        # Camera 0 device selector
        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Cam 0:"))
        self._cam0_combo = QComboBox()
        self._populate_cam_combo(self._cam0_combo, default_index=self._cfg.get("cam0_device", 0))
        self._cam0_combo.currentIndexChanged.connect(lambda: self._on_cam_device_changed(0))
        row0.addWidget(self._cam0_combo)
        layout.addLayout(row0)

        # Camera 1 device selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Cam 1:"))
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
        self._available_devices = [(0, "Dummy 0"), (1, "Dummy 1")] if self.dummy else enumerate_cameras()
        old0 = self._cam0_combo.currentData()
        old1 = self._cam1_combo.currentData()
        self._populate_cam_combo(self._cam0_combo, old0 if old0 is not None else 0)
        self._populate_cam_combo(self._cam1_combo, old1 if old1 is not None else 1)

    def _start_camera(self, camera_index: int):
        combo = self._cam0_combo if camera_index == 0 else self._cam1_combo
        device_id = combo.currentData()
        if device_id is None or device_id < 0:
            return

        CamClass = DummyCameraThread if self.dummy else CameraThread
        cam = CamClass(camera_index=camera_index, device_id=device_id)
        cam.frame_ready.connect(self._on_frame)
        cam.error.connect(self._on_camera_error)
        cam.recording_finished.connect(self._on_recording_finished)
        cam.stats_updated.connect(self._on_stats_updated)

        if camera_index == 0:
            self._cam0 = cam
        else:
            self._cam1 = cam
        cam.start()

    def _stop_camera(self, camera_index: int):
        cam = self._cam0 if camera_index == 0 else self._cam1
        if cam is not None:
            cam.stop()
            cam.wait(2000)
        if camera_index == 0:
            self._cam0 = None
        else:
            self._cam1 = None
        self._latest_frames.pop(camera_index, None)
        label = self._preview0 if camera_index == 0 else self._preview1
        label.clear()
        label.setText(f"Camera {camera_index}: no device")

    def _on_cam_device_changed(self, camera_index: int):
        if self._recording:
            return
        self._stop_camera(camera_index)
        combo = self._cam0_combo if camera_index == 0 else self._cam1_combo
        device_id = combo.currentData()
        if device_id is not None and device_id >= 0:
            self._start_camera(camera_index)

    def _build_capture_group(self) -> QGroupBox:
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
        saved_res = self._cfg.get("resolution", "640x480")
        idx = self._res_combo.findText(saved_res)
        self._res_combo.setCurrentIndex(idx if idx >= 0 else 0)
        res_row.addWidget(self._res_combo)
        layout.addLayout(res_row)

        # FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self._fps_combo = QComboBox()
        for fps in [15, 24, 30, 60]:
            self._fps_combo.addItem(str(fps), fps)
        saved_fps = self._cfg.get("fps", 30)
        idx = self._fps_combo.findData(saved_fps)
        self._fps_combo.setCurrentIndex(idx if idx >= 0 else 2)
        fps_row.addWidget(self._fps_combo)
        layout.addLayout(fps_row)

        # Codec
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("Codec:"))
        self._codec_combo = QComboBox()
        for name in CODECS:
            self._codec_combo.addItem(name)
        saved_codec = self._cfg.get("codec", "FFV1")
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
        auto_row.addWidget(QLabel("ON at (s):"))
        self._pump_on_input = QLineEdit(str(self._cfg.get("pump_on_time", 120)))
        self._pump_on_input.setMaximumWidth(60)
        auto_row.addWidget(self._pump_on_input)
        auto_row.addWidget(QLabel("OFF at (s):"))
        self._pump_off_input = QLineEdit(str(self._cfg.get("pump_off_time", 240)))
        self._pump_off_input.setMaximumWidth(60)
        auto_row.addWidget(self._pump_off_input)
        layout.addLayout(auto_row)

        # State indicator
        self._pump_indicator = QLabel("Relay: disconnected")
        layout.addWidget(self._pump_indicator)

        return group

    def _build_output_group(self) -> QGroupBox:
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

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Processing Pipeline")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Coming soon — filter pipeline controls will appear here."))
        return group

    # ── Slots ────────────────────────────────────────────────────────

    def _on_frame(self, camera_index: int, frame: np.ndarray):
        self._latest_frames[camera_index] = frame

    def _on_stats_updated(self, camera_index: int, fps: float, bitrate: float, latency_ms: float):
        self._cam_stats[camera_index] = (fps, bitrate, latency_ms)
        parts = []
        for idx in sorted(self._cam_stats):
            f, b, lat = self._cam_stats[idx]
            parts.append(f"Cam{idx}: {f:.1f}fps {b:.1f}Mb/s {lat:.1f}ms")
        self._status_stats.setText(" | ".join(parts))

        # Accumulate for stats log
        if self._recording:
            elapsed = time.monotonic() - self._record_start if self._record_start else 0
            self._stats_log.append({
                "time_s": round(elapsed, 1),
                "camera": camera_index,
                "fps": round(fps, 2),
                "bitrate_mbps": round(bitrate, 2),
                "write_latency_ms": round(latency_ms, 2),
            })

    def _on_camera_error(self, camera_index: int, message: str):
        log.warning("Camera %d error: %s", camera_index, message)

    def _on_recording_finished(self, camera_index: int):
        log.info("Camera %d recording finished", camera_index)
        # If both cameras are done (or the other wasn't recording), stop
        cam0_done = self._cam0 is None or not self._cam0._recording
        cam1_done = self._cam1 is None or not self._cam1._recording
        if cam0_done and cam1_done:
            self._stop_recording()

    def _on_tick(self):
        """Called at ~15fps for preview updates, elapsed time, and pump auto-trigger."""
        # Update previews
        for idx, label in [(0, self._preview0), (1, self._preview1)]:
            if idx in self._latest_frames:
                pm = frame_to_pixmap(self._latest_frames[idx], label.width(), label.height())
                label.setPixmap(pm)

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
                    self._update_pump_indicator()
                if not self._pump_triggered_off and elapsed >= self._pump_off_time:
                    self._relay.send_off()
                    self._pump_triggered_off = True
                    self._update_pump_indicator()

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

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        # Parse duration
        duration_secs = self._parse_duration(self._duration_input.text())
        if duration_secs is None:
            self._status_recording.setText("Invalid duration")
            return

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

        # Start recording on active cameras
        if self._cam0 is not None:
            self._cam0.start_recording(str(out_dir / f"{file_prefix}cam0{ext}"), duration_secs)
        if self._cam1 is not None:
            self._cam1.start_recording(str(out_dir / f"{file_prefix}cam1{ext}"), duration_secs)

        self._recording = True
        self._record_start = time.monotonic()
        self._record_btn.setText("Stop Recording")
        self._record_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #c0392b; color: white;")
        self._status_recording.setText("Recording")

        # Pump auto setup
        self._pump_auto_enabled = self._pump_auto_check.isChecked()
        self._pump_triggered_on = False
        self._pump_triggered_off = False
        try:
            self._pump_on_time = float(self._pump_on_input.text())
            self._pump_off_time = float(self._pump_off_input.text())
        except ValueError:
            self._pump_auto_enabled = False

        # Disable controls during recording
        self._set_controls_enabled(False)

    def _stop_recording(self):
        if self._cam0 is not None:
            self._cam0.stop_recording()
        if self._cam1 is not None:
            self._cam1.stop_recording()

        # Write stats log
        self._write_stats_log()

        self._recording = False
        self._record_start = None
        self._record_out_dir = None
        self._cam_stats.clear()
        self._status_stats.setText("")
        self._record_btn.setText("Start Recording")
        self._record_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self._status_recording.setText("Idle")
        self._status_elapsed.setText("00:00.0")

        # Turn off pump if auto was enabled
        if self._pump_auto_enabled and self._relay:
            self._relay.send_off()
            self._pump_auto_enabled = False

        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        self._prefix_input.setEnabled(enabled)
        self._duration_input.setEnabled(enabled)
        self._res_combo.setEnabled(enabled)
        self._fps_combo.setEnabled(enabled)
        self._codec_combo.setEnabled(enabled)
        self._cam0_combo.setEnabled(enabled)
        self._cam1_combo.setEnabled(enabled)

    def _parse_duration(self, text: str) -> Optional[float]:
        """Parse mm:ss or just seconds."""
        text = text.strip()
        try:
            if ":" in text:
                parts = text.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            return float(text)
        except (ValueError, IndexError):
            return None

    # ── Relay ────────────────────────────────────────────────────────

    def _refresh_ports(self):
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
        if self._relay:
            self._relay.test_relay()

    def _pump_on(self):
        if self._relay:
            self._relay.send_on()
            self._update_pump_indicator()

    def _pump_off(self):
        if self._relay:
            self._relay.send_off()
            self._update_pump_indicator()

    def _update_pump_indicator(self):
        if self._relay:
            state = "ON" if self._relay.is_on else "OFF"
            self._status_pump.setText(f"Pump: {state}")
            self._pump_indicator.setText(f"Relay: {state}")
        else:
            self._status_pump.setText("Pump: N/A")

    # ── Output ───────────────────────────────────────────────────────

    def _pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self._output_dir)
        if path:
            self._output_dir = path
            self._output_label.setText(path)

    def _open_output_dir(self):
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
        })
        save_config(self._cfg)

    # ── Cleanup ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._save_current_config()
        for cam in (self._cam0, self._cam1):
            if cam is not None:
                cam.stop()
        for cam in (self._cam0, self._cam1):
            if cam is not None:
                cam.wait(2000)
        if self._relay:
            self._relay.close()
        event.accept()
