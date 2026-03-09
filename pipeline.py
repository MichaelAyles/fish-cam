"""Filter pipeline engine and per-camera/shared pipeline manager.

Provides thread-safe ordered filter application and JSON serialisation
for saving/loading pipeline configurations.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from filters import BaseFilter, filter_from_dict

log = logging.getLogger(__name__)


class FilterPipeline:
    """Thread-safe ordered list of filters applied to frames."""

    def __init__(self):
        self._filters: list[BaseFilter] = []
        self._lock = threading.Lock()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all enabled filters in order. Errors are caught per-filter."""
        with self._lock:
            filters = list(self._filters)
        for f in filters:
            if not f.enabled:
                f._error = None
                continue
            try:
                frame = f.apply(frame)
                f._error = None
            except Exception as e:
                f._error = str(e)
                log.debug("Filter %s error: %s", f.NAME, e)
        return frame

    def add_filter(self, filt: BaseFilter, index: Optional[int] = None):
        with self._lock:
            if index is None:
                self._filters.append(filt)
            else:
                self._filters.insert(index, filt)

    def remove_filter(self, index: int):
        with self._lock:
            if 0 <= index < len(self._filters):
                self._filters.pop(index)

    def move_filter(self, from_idx: int, to_idx: int):
        with self._lock:
            if from_idx == to_idx:
                return
            if 0 <= from_idx < len(self._filters):
                f = self._filters.pop(from_idx)
                to_idx = min(to_idx, len(self._filters))
                self._filters.insert(to_idx, f)

    def clear(self):
        with self._lock:
            self._filters.clear()

    def get_filters(self) -> list[BaseFilter]:
        with self._lock:
            return list(self._filters)

    def reset_stateful(self):
        """Reset state on all stateful filters (e.g. background subtractors)."""
        with self._lock:
            for f in self._filters:
                f.reset_state()

    def to_list(self) -> list[dict]:
        with self._lock:
            return [f.to_dict() for f in self._filters]

    def load_list(self, data: list[dict]):
        filters = [filter_from_dict(d) for d in data]
        with self._lock:
            self._filters = filters

    def has_enabled_filters(self) -> bool:
        """Return True if at least one filter is enabled."""
        with self._lock:
            return any(f.enabled for f in self._filters)

    def __len__(self):
        with self._lock:
            return len(self._filters)


class PipelineManager:
    """Manages shared or per-camera filter pipelines."""

    def __init__(self):
        self.shared = True
        self._shared_pipeline = FilterPipeline()
        self._per_camera: dict[str, FilterPipeline] = {
            "top": FilterPipeline(),
            "front": FilterPipeline(),
        }

    def get_pipeline(self, camera_tag: str) -> FilterPipeline:
        if self.shared:
            return self._shared_pipeline
        return self._per_camera.get(camera_tag, self._shared_pipeline)

    def apply(self, frame: np.ndarray, camera_tag: str) -> np.ndarray:
        return self.get_pipeline(camera_tag).apply(frame)

    def reset_all(self):
        self._shared_pipeline.reset_stateful()
        for p in self._per_camera.values():
            p.reset_stateful()

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "shared": self.shared,
            "pipelines": {
                "shared": self._shared_pipeline.to_list(),
                "top": self._per_camera["top"].to_list(),
                "front": self._per_camera["front"].to_list(),
            },
        }

    def load_dict(self, data: dict):
        self.shared = data.get("shared", True)
        pipelines = data.get("pipelines", {})
        if "shared" in pipelines:
            self._shared_pipeline.load_list(pipelines["shared"])
        if "top" in pipelines:
            self._per_camera["top"].load_list(pipelines["top"])
        if "front" in pipelines:
            self._per_camera["front"].load_list(pipelines["front"])

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_json(self, path: str):
        with open(path) as f:
            self.load_dict(json.load(f))
