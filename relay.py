"""USB serial relay controller for pump automation.

Supports a 4-byte relay protocol over serial (A0 01 {ON/OFF} {checksum}).
Includes a mock implementation for hardware-free development.
"""

import logging
from typing import Optional

import serial
import serial.tools.list_ports

log = logging.getLogger(__name__)

# 4-byte relay protocol: A0 01 {01=ON|00=OFF} {checksum}
_PREFIX = bytes([0xA0, 0x01])
_CMD_ON = bytes([0xA0, 0x01, 0x01, 0xA2])
_CMD_OFF = bytes([0xA0, 0x01, 0x00, 0xA1])


def scan_ports() -> list[str]:
    """Return list of available serial port device names."""
    return [p.device for p in serial.tools.list_ports.comports()]


class RelayController:
    """Controls a USB serial relay using 4-byte protocol."""

    def __init__(self, port: str, baudrate: int = 9600):
        """Initialise the relay controller.

        Args:
            port: Serial port device name (e.g. COM3 or /dev/ttyUSB0).
            baudrate: Serial baud rate.
        """
        self.port = port
        self.baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._is_on = False

    def open(self):
        """Open the serial port connection."""
        self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
        log.info("Relay opened on %s", self.port)

    def close(self):
        """Turn off the relay and close the serial port."""
        if self._ser and self._ser.is_open:
            try:
                self.send_off()
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
            log.info("Relay closed")
        self._ser = None

    def send_on(self):
        """Send the relay ON command."""
        if self._ser and self._ser.is_open:
            self._ser.write(_CMD_ON)
            self._is_on = True
            log.info("Relay ON")

    def send_off(self):
        """Send the relay OFF command."""
        if self._ser and self._ser.is_open:
            self._ser.write(_CMD_OFF)
            self._is_on = False
            log.info("Relay OFF")

    def test_relay(self):
        """Quick on/off cycle for hardware verification."""
        import time
        self.send_on()
        time.sleep(0.5)
        self.send_off()

    @property
    def is_on(self) -> bool:
        """Return True if the relay is currently energised."""
        return self._is_on


class MockRelayController:
    """Mock relay that logs commands instead of sending serial data."""

    def __init__(self, port: str = "MOCK", baudrate: int = 9600):
        """Initialise the mock relay.

        Args:
            port: Ignored, accepted for API compatibility.
            baudrate: Ignored, accepted for API compatibility.
        """
        self.port = port
        self.baudrate = baudrate
        self._is_on = False

    def open(self):
        """Simulate opening the serial port."""
        log.info("[Mock] Relay opened on %s", self.port)

    def close(self):
        """Simulate closing the serial port."""
        self._is_on = False
        log.info("[Mock] Relay closed")

    def send_on(self):
        """Simulate sending the relay ON command."""
        self._is_on = True
        log.info("[Mock] Relay ON")

    def send_off(self):
        """Simulate sending the relay OFF command."""
        self._is_on = False
        log.info("[Mock] Relay OFF")

    def test_relay(self):
        """Quick on/off cycle for mock verification."""
        import time
        self.send_on()
        time.sleep(0.5)
        self.send_off()

    @property
    def is_on(self) -> bool:
        """Return True if the mock relay is currently on."""
        return self._is_on
