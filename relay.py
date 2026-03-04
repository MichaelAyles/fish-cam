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
        self.port = port
        self.baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._is_on = False

    def open(self):
        self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
        log.info("Relay opened on %s", self.port)

    def close(self):
        if self._ser and self._ser.is_open:
            self.send_off()
            self._ser.close()
            log.info("Relay closed")
        self._ser = None

    def send_on(self):
        if self._ser and self._ser.is_open:
            self._ser.write(_CMD_ON)
            self._is_on = True
            log.info("Relay ON")

    def send_off(self):
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
        return self._is_on


class MockRelayController:
    """Mock relay that logs commands instead of sending serial data."""

    def __init__(self, port: str = "MOCK", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self._is_on = False

    def open(self):
        log.info("[Mock] Relay opened on %s", self.port)

    def close(self):
        self._is_on = False
        log.info("[Mock] Relay closed")

    def send_on(self):
        self._is_on = True
        log.info("[Mock] Relay ON")

    def send_off(self):
        self._is_on = False
        log.info("[Mock] Relay OFF")

    def test_relay(self):
        import time
        self.send_on()
        time.sleep(0.5)
        self.send_off()

    @property
    def is_on(self) -> bool:
        return self._is_on
