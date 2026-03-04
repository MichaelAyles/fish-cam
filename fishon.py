#!/usr/bin/env python3
"""Fish-On — entry point.

Sets up logging (console + rotating file), applies the dark theme,
and launches the main GUI window.
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PyQt6.QtWidgets import QApplication

import sys

from gui import MainWindow

DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #ddd;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #3c3f41;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px 6px;
    color: #ddd;
}
QPushButton {
    background-color: #365880;
    border: 1px solid #4a6d8c;
    border-radius: 3px;
    padding: 5px 12px;
    color: #ddd;
}
QPushButton:hover {
    background-color: #3e6d9e;
}
QPushButton:pressed {
    background-color: #2d4a6a;
}
QCheckBox {
    spacing: 6px;
}
QStatusBar {
    background-color: #313335;
    color: #aaa;
}
QLabel {
    background-color: transparent;
}
QSlider::groove:horizontal {
    border: 1px solid #555;
    height: 6px;
    background: #3c3f41;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #5a8fbf;
    border: 1px solid #4a6d8c;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QDoubleSpinBox {
    background-color: #3c3f41;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px 6px;
    color: #ddd;
}
QTabWidget::pane {
    border: 1px solid #555;
    background-color: #2b2b2b;
}
QTabBar::tab {
    background-color: #313335;
    color: #aaa;
    padding: 6px 12px;
    border: 1px solid #555;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #2b2b2b;
    color: #ddd;
}
QTabBar::tab:hover {
    background-color: #3c3f41;
}
QListWidget {
    background-color: #2b2b2b;
    border: 1px solid #555;
    border-radius: 3px;
}
QListWidget::item {
    border-bottom: 1px solid #3c3f41;
    padding: 2px;
}
QListWidget::item:selected {
    background-color: #365880;
}
"""


def _setup_logging():
    """Configure console + rotating file logging."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler
    log_dir = Path.home() / ".fishon"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "fishon.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def main():
    """Parse arguments, set up logging, and launch the application."""
    parser = argparse.ArgumentParser(description="Fish-On Capture System")
    parser.add_argument(
        "--dummy", action="store_true",
        help="Run with synthetic cameras and mock relay (no hardware needed)",
    )
    args = parser.parse_args()

    _setup_logging()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow(dummy=args.dummy)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
