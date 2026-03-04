"""Image filter implementations for the processing pipeline.

Provides a registry of OpenCV-based filters, each with typed parameter
definitions, pre-allocated caches, and a uniform apply(frame) interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

# ── Parameter definition ────────────────────────────────────────────

@dataclass
class ParamDef:
    """Describes a single tuneable parameter for a filter."""
    name: str
    label: str
    param_type: str  # "int", "float", "bool", "choice"
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list[str]] = None


# ── Base class ──────────────────────────────────────────────────────

class BaseFilter(ABC):
    """Abstract base class for all pipeline filters."""

    NAME: str = ""
    CATEGORY: str = ""
    PARAM_DEFS: list[ParamDef] = []

    def __init__(self):
        self.enabled = True
        self._error: Optional[str] = None
        self._params: dict[str, Any] = {}
        for pd in self.PARAM_DEFS:
            self._params[pd.name] = pd.default
        self._rebuild_cache()

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return the result."""
        ...

    def set_param(self, name: str, value: Any):
        """Update a parameter value and rebuild internal caches."""
        self._params[name] = value
        self._rebuild_cache()

    def get_param(self, name: str) -> Any:
        return self._params.get(name)

    def _rebuild_cache(self):
        """Pre-allocate kernels or objects after a param change. Override in subclasses."""
        pass

    def reset_state(self):
        """Reset any accumulated state (e.g. background model). Override in subclasses."""
        pass

    def to_dict(self) -> dict:
        return {
            "name": self.NAME,
            "enabled": self.enabled,
            "params": dict(self._params),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseFilter":
        f = cls()
        f.enabled = data.get("enabled", True)
        for k, v in data.get("params", {}).items():
            f._params[k] = v
        f._rebuild_cache()
        return f


# ── Registry ────────────────────────────────────────────────────────

FILTER_REGISTRY: dict[str, type[BaseFilter]] = {}


def register_filter(cls: type[BaseFilter]) -> type[BaseFilter]:
    """Decorator to register a filter class in the global registry."""
    FILTER_REGISTRY[cls.NAME] = cls
    return cls


def filter_from_dict(data: dict) -> BaseFilter:
    """Factory: create a filter instance from a serialised dict."""
    cls = FILTER_REGISTRY[data["name"]]
    return cls.from_dict(data)


# ── Helper ──────────────────────────────────────────────────────────

def _ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _odd(v: int) -> int:
    """Ensure a value is odd and >= 1."""
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1


# ── Filter implementations ─────────────────────────────────────────

@register_filter
class ConvertColor(BaseFilter):
    NAME = "ConvertColor"
    CATEGORY = "Color"
    PARAM_DEFS = [
        ParamDef("conversion", "Conversion", "choice", "BGR2GRAY",
                 choices=["BGR2GRAY", "BGR2HSV", "BGR2Lab", "GRAY2BGR"]),
    ]

    _CODES = {
        "BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "BGR2Lab": cv2.COLOR_BGR2LAB,
        "GRAY2BGR": cv2.COLOR_GRAY2BGR,
    }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        code = self._CODES.get(self._params["conversion"])
        if code is None:
            return frame
        try:
            return cv2.cvtColor(frame, code)
        except cv2.error:
            return frame


@register_filter
class GaussianBlur(BaseFilter):
    NAME = "GaussianBlur"
    CATEGORY = "Blur"
    PARAM_DEFS = [
        ParamDef("ksize", "Kernel Size", "int", 5, 1, 31, 2),
        ParamDef("sigma", "Sigma", "float", 0.0, 0.0, 20.0, 0.1),
    ]

    def _rebuild_cache(self):
        self._ksize = _odd(self._params["ksize"])

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (self._ksize, self._ksize), self._params["sigma"])


@register_filter
class MedianBlur(BaseFilter):
    NAME = "MedianBlur"
    CATEGORY = "Blur"
    PARAM_DEFS = [
        ParamDef("ksize", "Kernel Size", "int", 5, 3, 31, 2),
    ]

    def _rebuild_cache(self):
        self._ksize = _odd(max(3, self._params["ksize"]))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(frame, self._ksize)


@register_filter
class BilateralFilter(BaseFilter):
    NAME = "BilateralFilter"
    CATEGORY = "Blur"
    PARAM_DEFS = [
        ParamDef("d", "Diameter", "int", 9, 1, 15, 1),
        ParamDef("sigma_color", "Sigma Color", "float", 75.0, 1.0, 300.0, 1.0),
        ParamDef("sigma_space", "Sigma Space", "float", 75.0, 1.0, 300.0, 1.0),
    ]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            frame, self._params["d"],
            self._params["sigma_color"], self._params["sigma_space"],
        )


@register_filter
class Threshold(BaseFilter):
    NAME = "Threshold"
    CATEGORY = "Threshold"
    PARAM_DEFS = [
        ParamDef("thresh", "Threshold", "int", 128, 0, 255, 1),
        ParamDef("maxval", "Max Value", "int", 255, 0, 255, 1),
        ParamDef("type", "Type", "choice", "BINARY",
                 choices=["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV", "OTSU"]),
    ]

    _TYPES = {
        "BINARY": cv2.THRESH_BINARY,
        "BINARY_INV": cv2.THRESH_BINARY_INV,
        "TRUNC": cv2.THRESH_TRUNC,
        "TOZERO": cv2.THRESH_TOZERO,
        "TOZERO_INV": cv2.THRESH_TOZERO_INV,
        "OTSU": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(frame)
        t = self._TYPES.get(self._params["type"], cv2.THRESH_BINARY)
        _, result = cv2.threshold(gray, self._params["thresh"], self._params["maxval"], t)
        return result


@register_filter
class AdaptiveThreshold(BaseFilter):
    NAME = "AdaptiveThreshold"
    CATEGORY = "Threshold"
    PARAM_DEFS = [
        ParamDef("maxval", "Max Value", "int", 255, 0, 255, 1),
        ParamDef("method", "Method", "choice", "GAUSSIAN",
                 choices=["GAUSSIAN", "MEAN"]),
        ParamDef("block_size", "Block Size", "int", 11, 3, 99, 2),
        ParamDef("C", "Constant C", "int", 2, -50, 50, 1),
    ]

    _METHODS = {
        "GAUSSIAN": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        "MEAN": cv2.ADAPTIVE_THRESH_MEAN_C,
    }

    def _rebuild_cache(self):
        self._block = _odd(max(3, self._params["block_size"]))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(frame)
        method = self._METHODS.get(self._params["method"], cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        return cv2.adaptiveThreshold(
            gray, self._params["maxval"], method,
            cv2.THRESH_BINARY, self._block, self._params["C"],
        )


@register_filter
class InRange(BaseFilter):
    NAME = "InRange"
    CATEGORY = "Threshold"
    PARAM_DEFS = [
        ParamDef("low_0", "Low Ch0", "int", 0, 0, 255, 1),
        ParamDef("low_1", "Low Ch1", "int", 0, 0, 255, 1),
        ParamDef("low_2", "Low Ch2", "int", 0, 0, 255, 1),
        ParamDef("high_0", "High Ch0", "int", 255, 0, 255, 1),
        ParamDef("high_1", "High Ch1", "int", 255, 0, 255, 1),
        ParamDef("high_2", "High Ch2", "int", 255, 0, 255, 1),
    ]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        p = self._params
        low = np.array([p["low_0"], p["low_1"], p["low_2"]], dtype=np.uint8)
        high = np.array([p["high_0"], p["high_1"], p["high_2"]], dtype=np.uint8)
        if frame.ndim == 2:
            return cv2.inRange(frame, low[0], high[0])
        return cv2.inRange(frame, low, high)


@register_filter
class Erode(BaseFilter):
    NAME = "Erode"
    CATEGORY = "Morphology"
    PARAM_DEFS = [
        ParamDef("ksize", "Kernel Size", "int", 3, 1, 31, 2),
        ParamDef("iterations", "Iterations", "int", 1, 1, 20, 1),
    ]

    def _rebuild_cache(self):
        k = _odd(self._params["ksize"])
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.erode(frame, self._kernel, iterations=self._params["iterations"])


@register_filter
class Dilate(BaseFilter):
    NAME = "Dilate"
    CATEGORY = "Morphology"
    PARAM_DEFS = [
        ParamDef("ksize", "Kernel Size", "int", 3, 1, 31, 2),
        ParamDef("iterations", "Iterations", "int", 1, 1, 20, 1),
    ]

    def _rebuild_cache(self):
        k = _odd(self._params["ksize"])
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.dilate(frame, self._kernel, iterations=self._params["iterations"])


@register_filter
class MorphologyEx(BaseFilter):
    NAME = "MorphologyEx"
    CATEGORY = "Morphology"
    PARAM_DEFS = [
        ParamDef("op", "Operation", "choice", "CLOSE",
                 choices=["OPEN", "CLOSE", "GRADIENT", "TOPHAT", "BLACKHAT"]),
        ParamDef("ksize", "Kernel Size", "int", 5, 1, 31, 2),
        ParamDef("shape", "Shape", "choice", "RECT",
                 choices=["RECT", "ELLIPSE", "CROSS"]),
        ParamDef("iterations", "Iterations", "int", 1, 1, 20, 1),
    ]

    _OPS = {
        "OPEN": cv2.MORPH_OPEN,
        "CLOSE": cv2.MORPH_CLOSE,
        "GRADIENT": cv2.MORPH_GRADIENT,
        "TOPHAT": cv2.MORPH_TOPHAT,
        "BLACKHAT": cv2.MORPH_BLACKHAT,
    }
    _SHAPES = {
        "RECT": cv2.MORPH_RECT,
        "ELLIPSE": cv2.MORPH_ELLIPSE,
        "CROSS": cv2.MORPH_CROSS,
    }

    def _rebuild_cache(self):
        k = _odd(self._params["ksize"])
        shape = self._SHAPES.get(self._params["shape"], cv2.MORPH_RECT)
        self._kernel = cv2.getStructuringElement(shape, (k, k))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        op = self._OPS.get(self._params["op"], cv2.MORPH_CLOSE)
        return cv2.morphologyEx(frame, op, self._kernel, iterations=self._params["iterations"])


@register_filter
class Canny(BaseFilter):
    NAME = "Canny"
    CATEGORY = "Edge"
    PARAM_DEFS = [
        ParamDef("threshold1", "Threshold 1", "int", 50, 0, 500, 1),
        ParamDef("threshold2", "Threshold 2", "int", 150, 0, 500, 1),
        ParamDef("aperture", "Aperture", "int", 3, 3, 7, 2),
    ]

    def _rebuild_cache(self):
        self._aperture = _odd(max(3, min(7, self._params["aperture"])))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = _ensure_gray(frame)
        return cv2.Canny(gray, self._params["threshold1"], self._params["threshold2"],
                         apertureSize=self._aperture)


@register_filter
class BackgroundSubtractor(BaseFilter):
    NAME = "BackgroundSubtractor"
    CATEGORY = "Background"
    PARAM_DEFS = [
        ParamDef("method", "Method", "choice", "MOG2", choices=["MOG2", "KNN"]),
        ParamDef("history", "History", "int", 500, 10, 5000, 10),
        ParamDef("threshold", "Threshold", "float", 16.0, 1.0, 100.0, 1.0),
        ParamDef("detect_shadows", "Detect Shadows", "bool", True),
        ParamDef("learning_rate", "Learning Rate", "float", -1.0, -1.0, 1.0, 0.01),
    ]

    def _rebuild_cache(self):
        p = self._params
        if p["method"] == "KNN":
            self._subtractor = cv2.createBackgroundSubtractorKNN(
                history=p["history"], dist2Threshold=p["threshold"],
                detectShadows=p["detect_shadows"],
            )
        else:
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=p["history"], varThreshold=p["threshold"],
                detectShadows=p["detect_shadows"],
            )

    def reset_state(self):
        self._rebuild_cache()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        lr = self._params["learning_rate"]
        return self._subtractor.apply(frame, learningRate=lr if lr >= 0 else -1)


@register_filter
class BrightnessContrast(BaseFilter):
    NAME = "BrightnessContrast"
    CATEGORY = "Color"
    PARAM_DEFS = [
        ParamDef("brightness", "Brightness", "int", 0, -127, 127, 1),
        ParamDef("contrast", "Contrast", "float", 1.0, 0.0, 3.0, 0.05),
    ]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(frame, alpha=self._params["contrast"],
                                   beta=self._params["brightness"])


@register_filter
class Normalize(BaseFilter):
    NAME = "Normalize"
    CATEGORY = "Color"
    PARAM_DEFS = [
        ParamDef("clip_limit", "Clip Limit", "float", 2.0, 0.1, 40.0, 0.5),
        ParamDef("grid_size", "Grid Size", "int", 8, 2, 32, 1),
    ]

    def _rebuild_cache(self):
        self._clahe = cv2.createCLAHE(
            clipLimit=self._params["clip_limit"],
            tileGridSize=(self._params["grid_size"], self._params["grid_size"]),
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return self._clahe.apply(frame)
        # Apply CLAHE to L channel of LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@register_filter
class CropROI(BaseFilter):
    NAME = "CropROI"
    CATEGORY = "ROI"
    PARAM_DEFS = [
        ParamDef("x", "X", "int", 0, 0, 10000, 1),
        ParamDef("y", "Y", "int", 0, 0, 10000, 1),
        ParamDef("w", "Width", "int", 0, 0, 10000, 1),
        ParamDef("h", "Height", "int", 0, 0, 10000, 1),
    ]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        p = self._params
        fh, fw = frame.shape[:2]
        x, y = max(0, p["x"]), max(0, p["y"])
        w = p["w"] if p["w"] > 0 else fw - x
        h = p["h"] if p["h"] > 0 else fh - y
        x2, y2 = min(x + w, fw), min(y + h, fh)
        if x2 <= x or y2 <= y:
            return frame
        return frame[y:y2, x:x2].copy()


@register_filter
class MaskROI(BaseFilter):
    NAME = "MaskROI"
    CATEGORY = "ROI"
    PARAM_DEFS = [
        ParamDef("x", "X", "int", 0, 0, 10000, 1),
        ParamDef("y", "Y", "int", 0, 0, 10000, 1),
        ParamDef("w", "Width", "int", 0, 0, 10000, 1),
        ParamDef("h", "Height", "int", 0, 0, 10000, 1),
        ParamDef("invert", "Invert (keep outside)", "bool", False),
    ]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        p = self._params
        fh, fw = frame.shape[:2]
        x, y = max(0, p["x"]), max(0, p["y"])
        w = p["w"] if p["w"] > 0 else fw - x
        h = p["h"] if p["h"] > 0 else fh - y
        mask = np.zeros((fh, fw), dtype=np.uint8)
        mask[y:min(y + h, fh), x:min(x + w, fw)] = 255
        if p["invert"]:
            mask = cv2.bitwise_not(mask)
        if frame.ndim == 2:
            return cv2.bitwise_and(frame, frame, mask=mask)
        return cv2.bitwise_and(frame, frame, mask=mask)


@register_filter
class Flip(BaseFilter):
    NAME = "Flip"
    CATEGORY = "Transform"
    PARAM_DEFS = [
        ParamDef("mode", "Mode", "choice", "Horizontal",
                 choices=["Horizontal", "Vertical", "Both"]),
    ]

    _CODES = {"Horizontal": 1, "Vertical": 0, "Both": -1}

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.flip(frame, self._CODES.get(self._params["mode"], 1))


@register_filter
class Rotate(BaseFilter):
    NAME = "Rotate"
    CATEGORY = "Transform"
    PARAM_DEFS = [
        ParamDef("angle", "Angle", "choice", "90 CW",
                 choices=["90 CW", "90 CCW", "180"]),
    ]

    _CODES = {
        "90 CW": cv2.ROTATE_90_CLOCKWISE,
        "90 CCW": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.rotate(frame, self._CODES.get(self._params["angle"], cv2.ROTATE_90_CLOCKWISE))
