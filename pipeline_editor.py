"""Pipeline editor dock widget for building and tuning filter chains.

Provides a drag-drop filter list, per-filter parameter cards, categorised
add-filter menus, and an ROI overlay for spatial filters.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from filters import FILTER_REGISTRY, BaseFilter, ParamDef
from pipeline import FilterPipeline, PipelineManager

log = logging.getLogger(__name__)


# ── Filter card (per-filter parameter editor) ───────────────────────

class FilterCardWidget(QWidget):
    """Expandable card showing filter name, enable toggle, and parameter editors."""

    param_changed = pyqtSignal()
    remove_requested = pyqtSignal()

    def __init__(self, filt: BaseFilter, parent=None):
        super().__init__(parent)
        self.filter = filt
        self._expanded = False
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(75)
        self._debounce.timeout.connect(self._apply_params)
        self._param_widgets: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(4)

        self._grip = QLabel("\u2261")  # grip icon
        self._grip.setFixedWidth(16)
        self._grip.setStyleSheet("color: #888; font-size: 16px;")
        header.addWidget(self._grip)

        self._name_btn = QPushButton(filt.NAME)
        self._name_btn.setFlat(True)
        self._name_btn.setStyleSheet("text-align: left; font-weight: bold; padding: 2px 4px;")
        self._name_btn.clicked.connect(self._toggle_expand)
        header.addWidget(self._name_btn, stretch=1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
        self._error_label.setVisible(False)
        header.addWidget(self._error_label)

        self._enable_cb = QCheckBox()
        self._enable_cb.setChecked(filt.enabled)
        self._enable_cb.setToolTip("Enable/disable filter")
        self._enable_cb.stateChanged.connect(self._on_enable_changed)
        header.addWidget(self._enable_cb)

        remove_btn = QPushButton("\u2715")
        remove_btn.setFixedSize(22, 22)
        remove_btn.setToolTip("Remove filter")
        remove_btn.clicked.connect(self.remove_requested.emit)
        header.addWidget(remove_btn)

        layout.addLayout(header)

        # Expandable body
        self._body = QWidget()
        self._body.setVisible(False)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(20, 4, 4, 4)
        body_layout.setSpacing(4)

        for pd in filt.PARAM_DEFS:
            row = QHBoxLayout()
            row.addWidget(QLabel(pd.label + ":"))
            widget = self._create_param_widget(pd)
            row.addWidget(widget)
            body_layout.addLayout(row)
            self._param_widgets[pd.name] = widget

        # Draw ROI button for ROI filters
        if filt.NAME in ("CropROI", "MaskROI"):
            self._draw_roi_btn = QPushButton("Draw ROI")
            self._draw_roi_btn.setToolTip("Click then draw a rectangle on the preview")
            body_layout.addWidget(self._draw_roi_btn)

        layout.addWidget(self._body)

    def _create_param_widget(self, pd: ParamDef) -> QWidget:
        if pd.param_type == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(self.filter.get_param(pd.name)))
            cb.stateChanged.connect(lambda: self._schedule_update())
            return cb
        elif pd.param_type == "choice":
            combo = QComboBox()
            combo.addItems(pd.choices or [])
            current = self.filter.get_param(pd.name)
            idx = combo.findText(str(current))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentTextChanged.connect(lambda: self._schedule_update())
            return combo
        elif pd.param_type == "float":
            spin = QDoubleSpinBox()
            spin.setRange(pd.min_val or 0, pd.max_val or 100)
            spin.setSingleStep(pd.step or 0.1)
            spin.setDecimals(2)
            spin.setValue(float(self.filter.get_param(pd.name)))
            spin.valueChanged.connect(lambda: self._schedule_update())
            return spin
        else:  # int
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(pd.min_val or 0), int(pd.max_val or 255))
            slider.setSingleStep(int(pd.step or 1))
            slider.setValue(int(self.filter.get_param(pd.name)))
            spin = QSpinBox()
            spin.setRange(int(pd.min_val or 0), int(pd.max_val or 255))
            spin.setSingleStep(int(pd.step or 1))
            spin.setValue(int(self.filter.get_param(pd.name)))
            spin.setFixedWidth(60)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            spin.valueChanged.connect(lambda: self._schedule_update())
            h.addWidget(slider, stretch=1)
            h.addWidget(spin)
            container._spin = spin  # keep reference for reading value
            return container

    def _toggle_expand(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)

    def _on_enable_changed(self):
        self.filter.enabled = self._enable_cb.isChecked()
        self.param_changed.emit()

    def _schedule_update(self):
        self._debounce.start()

    def _apply_params(self):
        for pd in self.filter.PARAM_DEFS:
            widget = self._param_widgets.get(pd.name)
            if widget is None:
                continue
            if pd.param_type == "bool":
                self.filter.set_param(pd.name, widget.isChecked())
            elif pd.param_type == "choice":
                self.filter.set_param(pd.name, widget.currentText())
            elif pd.param_type == "float":
                self.filter.set_param(pd.name, widget.value())
            else:  # int
                self.filter.set_param(pd.name, widget._spin.value())
        self.param_changed.emit()

    def update_error_display(self):
        err = self.filter._error
        if err:
            self._error_label.setText(err[:40])
            self._error_label.setVisible(True)
        else:
            self._error_label.setVisible(False)


# ── Filter list widget ──────────────────────────────────────────────

class FilterListWidget(QWidget):
    """Ordered list of filters with drag-drop reorder, add and clear."""

    pipeline_changed = pyqtSignal()

    def __init__(self, pipeline: FilterPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._cards: list[FilterCardWidget] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._add_btn = QPushButton("Add Filter")
        self._add_btn.clicked.connect(self._show_add_menu)
        toolbar.addWidget(self._add_btn)
        clear_btn = QPushButton("Clear Pipeline")
        clear_btn.clicked.connect(self._clear_pipeline)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        # List widget
        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setSpacing(2)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list)

        self._rebuild_list()

    def _rebuild_list(self):
        self._list.clear()
        self._cards.clear()
        for filt in self.pipeline.get_filters():
            self._add_card(filt)

    def _add_card(self, filt: BaseFilter):
        card = FilterCardWidget(filt)
        card.param_changed.connect(self._on_param_changed)
        card.remove_requested.connect(lambda c=card: self._remove_card(c))
        item = QListWidgetItem()
        item.setSizeHint(card.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, card)
        self._cards.append(card)
        # Update size when card expands/collapses
        card._body.installEventFilter(self)

    def eventFilter(self, obj, event):
        # Resize list items when cards expand/collapse
        if event.type() in (event.Type.Show, event.Type.Hide):
            for i in range(self._list.count()):
                item = self._list.item(i)
                w = self._list.itemWidget(item)
                if w:
                    item.setSizeHint(w.sizeHint())
            QTimer.singleShot(0, self._update_item_sizes)
        return super().eventFilter(obj, event)

    def _update_item_sizes(self):
        for i in range(self._list.count()):
            item = self._list.item(i)
            w = self._list.itemWidget(item)
            if w:
                item.setSizeHint(w.sizeHint())

    def _show_add_menu(self):
        menu = QMenu(self)
        # Group by category
        categories: dict[str, list[type[BaseFilter]]] = {}
        for cls in FILTER_REGISTRY.values():
            categories.setdefault(cls.CATEGORY, []).append(cls)
        for cat in sorted(categories):
            submenu = menu.addMenu(cat)
            for cls in sorted(categories[cat], key=lambda c: c.NAME):
                action = submenu.addAction(cls.NAME)
                action.triggered.connect(lambda checked, c=cls: self._add_filter(c))
        menu.exec(self._add_btn.mapToGlobal(self._add_btn.rect().bottomLeft()))

    def _add_filter(self, cls: type[BaseFilter]):
        filt = cls()
        self.pipeline.add_filter(filt)
        self._add_card(filt)
        self.pipeline_changed.emit()

    def _remove_card(self, card: FilterCardWidget):
        idx = self._cards.index(card)
        self.pipeline.remove_filter(idx)
        self._cards.pop(idx)
        self._list.takeItem(idx)
        self.pipeline_changed.emit()

    def _on_rows_moved(self, *args):
        # Sync pipeline order to match list widget order
        new_order: list[BaseFilter] = []
        for i in range(self._list.count()):
            w = self._list.itemWidget(self._list.item(i))
            if isinstance(w, FilterCardWidget):
                new_order.append(w.filter)
        # Rebuild pipeline in new order
        self.pipeline.clear()
        for f in new_order:
            self.pipeline.add_filter(f)
        self._cards = [
            self._list.itemWidget(self._list.item(i))
            for i in range(self._list.count())
            if isinstance(self._list.itemWidget(self._list.item(i)), FilterCardWidget)
        ]
        self.pipeline_changed.emit()

    def _on_param_changed(self):
        self.pipeline_changed.emit()

    def _clear_pipeline(self):
        self.pipeline.clear()
        self._rebuild_list()
        self.pipeline_changed.emit()

    def refresh(self):
        """Rebuild the list from the current pipeline state."""
        self._rebuild_list()

    def update_error_indicators(self):
        for card in self._cards:
            card.update_error_display()


# ── ROI overlay ─────────────────────────────────────────────────────

class ROIOverlay(QWidget):
    """Transparent overlay for drawing ROI rectangles on the preview."""

    roi_selected = pyqtSignal(int, int, int, int)  # x, y, w, h in frame coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self._drawing = False
        self._active = False
        self._start = QPoint()
        self._end = QPoint()
        self._frame_size = (640, 480)  # actual frame dimensions

    def activate(self, frame_w: int, frame_h: int):
        self._active = True
        self._frame_size = (frame_w, frame_h)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.show()
        self.raise_()

    def deactivate(self):
        self._active = False
        self._drawing = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.hide()

    def paintEvent(self, event):
        if not self._drawing:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(QBrush(QColor(0, 255, 0, 30)))
        rect = QRect(self._start, self._end).normalized()
        painter.drawRect(rect)

    def mousePressEvent(self, event):
        if self._active and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._start = event.pos()
            self._end = event.pos()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._end = event.pos()
            self.update()
            # Convert widget coords to frame coords
            rect = QRect(self._start, self._end).normalized()
            fw, fh = self._frame_size
            ww, wh = self.width(), self.height()
            if ww > 0 and wh > 0:
                x = int(rect.x() * fw / ww)
                y = int(rect.y() * fh / wh)
                w = int(rect.width() * fw / ww)
                h = int(rect.height() * fh / wh)
                self.roi_selected.emit(x, y, w, h)
            self.deactivate()


# ── Main editor widget ──────────────────────────────────────────────

class PipelineEditorWidget(QWidget):
    """Main pipeline editor with shared/per-camera modes, save/load."""

    pipeline_changed = pyqtSignal()

    def __init__(self, pipeline_manager: PipelineManager, parent=None):
        super().__init__(parent)
        self._manager = pipeline_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Mode toggles
        self._separate_cb = QCheckBox("Separate pipelines per camera")
        self._separate_cb.setChecked(not pipeline_manager.shared)
        self._separate_cb.stateChanged.connect(self._on_mode_changed)
        layout.addWidget(self._separate_cb)

        # Warning label (shown during recording)
        self._warning_label = QLabel("")
        self._warning_label.setStyleSheet("color: #f39c12; font-size: 11px;")
        self._warning_label.setVisible(False)
        self._warning_label.setWordWrap(True)
        layout.addWidget(self._warning_label)

        # Shared mode: single filter list
        self._shared_list = FilterListWidget(pipeline_manager._shared_pipeline)
        self._shared_list.pipeline_changed.connect(self._on_pipeline_changed)
        layout.addWidget(self._shared_list)

        # Separate mode: tabbed filter lists
        self._tab_widget = QTabWidget()
        self._top_list = FilterListWidget(pipeline_manager._per_camera["top"])
        self._top_list.pipeline_changed.connect(self._on_pipeline_changed)
        self._front_list = FilterListWidget(pipeline_manager._per_camera["front"])
        self._front_list.pipeline_changed.connect(self._on_pipeline_changed)
        self._tab_widget.addTab(self._top_list, "Top View")
        self._tab_widget.addTab(self._front_list, "Side View")
        layout.addWidget(self._tab_widget)

        # Save/Load buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Pipeline...")
        save_btn.clicked.connect(self._save_pipeline)
        btn_row.addWidget(save_btn)
        load_btn = QPushButton("Load Pipeline...")
        load_btn.clicked.connect(self._load_pipeline)
        btn_row.addWidget(load_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

        self._update_mode_visibility()

    def _on_mode_changed(self):
        self._manager.shared = not self._separate_cb.isChecked()
        self._update_mode_visibility()
        self._on_pipeline_changed()

    def _update_mode_visibility(self):
        shared = not self._separate_cb.isChecked()
        self._shared_list.setVisible(shared)
        self._tab_widget.setVisible(not shared)

    def _on_pipeline_changed(self):
        self.pipeline_changed.emit()

    def show_warning(self, message: str, duration_ms: int = 5000):
        self._warning_label.setText(message)
        self._warning_label.setVisible(True)
        QTimer.singleShot(duration_ms, lambda: self._warning_label.setVisible(False))

    def _save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "JSON Files (*.json)")
        if path:
            try:
                self._manager.save_json(path)
            except Exception as e:
                log.error("Failed to save pipeline: %s", e)

    def _load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "JSON Files (*.json)")
        if path:
            try:
                self._manager.load_json(path)
                self._separate_cb.setChecked(not self._manager.shared)
                self._update_mode_visibility()
                self._shared_list.refresh()
                self._top_list.refresh()
                self._front_list.refresh()
                self._on_pipeline_changed()
            except Exception as e:
                log.error("Failed to load pipeline: %s", e)

    def refresh_all(self):
        """Rebuild all filter lists from current pipeline state."""
        self._separate_cb.setChecked(not self._manager.shared)
        self._update_mode_visibility()
        self._shared_list.refresh()
        self._top_list.refresh()
        self._front_list.refresh()


# ── Popup dialog ────────────────────────────────────────────────────

class PipelineEditorDialog(QDialog):
    """Non-modal popup window for editing the filter pipeline."""

    def __init__(self, pipeline_manager: PipelineManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Pipeline")
        self.setMinimumSize(400, 500)
        self.resize(420, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.editor = PipelineEditorWidget(pipeline_manager)
        layout.addWidget(self.editor)
