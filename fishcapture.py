#!/usr/bin/env python3
"""Fish Tank Research Capture System — entry point."""

import argparse
import logging
import sys

from PyQt6.QtWidgets import QApplication

from gui import MainWindow


def main():
    parser = argparse.ArgumentParser(description="Fish Tank Research Capture System")
    parser.add_argument(
        "--dummy", action="store_true",
        help="Run with synthetic cameras and mock relay (no hardware needed)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    app = QApplication(sys.argv)
    window = MainWindow(dummy=args.dummy)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
