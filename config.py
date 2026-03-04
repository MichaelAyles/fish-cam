import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".fishcapture" / "config.json"

DEFAULTS = {
    "output_dir": str(Path.home() / "fish-capture"),
    "duration": "05:00",
    "resolution": "640x480",
    "fps": 30,
    "codec": "FFV1",
    "cam0_device": 0,
    "cam1_device": 1,
    "pump_port": "",
    "pump_auto": True,
    "pump_on_time": 120,
    "pump_off_time": 240,
    "video_prefix": "",
}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load config from JSON file, falling back to defaults for missing keys."""
    config = dict(DEFAULTS)
    if path.exists():
        try:
            with open(path) as f:
                stored = json.load(f)
            config.update(stored)
            log.info("Config loaded from %s", path)
        except Exception as e:
            log.warning("Failed to load config: %s", e)
    return config


def save_config(config: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH):
    """Save config to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config saved to %s", path)
