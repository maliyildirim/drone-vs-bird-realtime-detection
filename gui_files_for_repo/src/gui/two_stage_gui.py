"""
PySide6 GUI for the two-stage drone-vs-bird detection pipeline.

Pipeline:
    image/video frame
        -> YOLO detector: single class "target"
        -> crop detected regions
        -> YOLO classifier: background / bird / drone
        -> reject background and low-confidence crops
        -> render valid drone/bird results

This module intentionally avoids hard-coded local paths. Runtime paths are provided
through scripts/run_gui.py and configs/gui_config.yaml.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)"


@dataclass(frozen=True)
class GuiConfig:
    detector_path: str
    classifier_path: str
    save_dir: str = "outputs"

    detector_conf: float = 0.30
    detector_iou: float = 0.60
    detector_max_det: int = 80
    detector_imgsz_image: int = 960
    detector_imgsz_video: int = 640
    min_box_size_px: int = 7

    classifier_imgsz: int = 224
    classifier_min_conf: float = 0.60
    crop_padding: float = 0.35

    video_frame_skip: int = 1
    display_scale: float = 0.75
    show_confidence: bool = False


@dataclass
class InferenceResult:
    x1: float
    y1: float
    x2: float
    y2: float
    det_conf: float
    cls_conf: float
    label: str


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def bgr_to_qpixmap(image_bgr) -> QPixmap:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width
    q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)


def crop_square_with_padding(image_bgr, x1: float, y1: float, x2: float, y2: float, padding: float):
    height, width = image_bgr.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(box_width, box_height) * (1.0 + 2.0 * padding)

    new_x1 = int(round(center_x - side / 2.0))
    new_y1 = int(round(center_y - side / 2.0))
    new_x2 = int(round(center_x + side / 2.0))
    new_y2 = int(round(center_y + side / 2.0))

    new_x1 = int(clamp(new_x1, 0, width - 1))
    new_y1 = int(clamp(new_y1, 0, height - 1))
    new_x2 = int(clamp(new_x2, 1, width))
    new_y2 = int(clamp(new_y2, 1, height))

    if new_x2 - new_x1 < 2 or new_y2 - new_y1 < 2:
        return None

    return image_bgr[new_y1:new_y2, new_x1:new_x2]


class ImageViewer(QGraphicsView):
    """Image viewer with zoom, drag, and double-click-to-fit support."""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def set_pixmap_first(self, pixmap: QPixmap) -> None:
        self.scene().clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self._pixmap_item)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def update_pixmap(self, pixmap: QPixmap) -> None:
        if self._pixmap_item is None:
            self.set_pixmap_first(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)

    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            return
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)


class TwoStageEngine:
    def __init__(self, config: GuiConfig):
        self.config = config
        detector_path = Path(config.detector_path)
        classifier_path = Path(config.classifier_path)

        if not detector_path.exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}")
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")

        self.cuda_available = torch.cuda.is_available()
        self.device = 0 if self.cuda_available else "cpu"
        self.use_half = bool(self.cuda_available)

        if self.cuda_available:
            torch.backends.cudnn.benchmark = True

        self.detector = YOLO(str(detector_path))
        self.classifier = YOLO(str(classifier_path))
        self.detector_names = self.detector.names
        self.classifier_names = self.classifier.names
        self.classifier_id_to_label = self._build_classifier_label_map()

    def _build_classifier_label_map(self) -> dict[int, str]:
        names = {int(class_id): str(name).strip().lower() for class_id, name in self.classifier_names.items()}
        mapping: dict[int, str] = {}

        for class_id, label in names.items():
            if "background" in label or label in {"bg", "negative", "none"}:
                mapping[class_id] = "background"
            elif "bird" in label:
                mapping[class_id] = "bird"
            elif "drone" in label or "uav" in label:
                mapping[class_id] = "drone"

        if not mapping and len(names) >= 3:
            mapping = {0: "background", 1: "bird", 2: "drone"}
        if not mapping and len(names) >= 2:
            mapping = {0: "drone", 1: "bird"}

        return mapping

    def detect(self, frame_bgr, image_size: int):
        return self.detector.predict(
            source=frame_bgr,
            imgsz=image_size,
            conf=self.config.detector_conf,
            iou=self.config.detector_iou,
            device=self.device,
            half=self.use_half,
            verbose=False,
            max_det=self.config.detector_max_det,
        )[0]

    def classify_crops_batch(self, crops_bgr: list) -> list[tuple[str, float]]:
        if not crops_bgr:
            return []

        results = self.classifier.predict(
            source=crops_bgr,
            imgsz=self.config.classifier_imgsz,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )

        predictions: list[tuple[str, float]] = []
        for result in results:
            probs = result.probs
            top1_id = int(probs.top1)
            top1_conf = float(probs.top1conf)
            label = self.classifier_id_to_label.get(top1_id, "unknown")
            predictions.append((label, top1_conf))

        return predictions

    def run_two_stage(self, frame_bgr, detector_image_size: int) -> tuple[list[InferenceResult], int]:
        detection_result = self.detect(frame_bgr, image_size=detector_image_size)
        boxes = detection_result.boxes

        if boxes is None or len(boxes) == 0:
            return [], 0

        crops = []
        kept_boxes = []

        for index in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy().tolist()
            det_conf = float(boxes.conf[index].cpu().numpy())
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < self.config.min_box_size_px or box_height < self.config.min_box_size_px:
                continue

            crop = crop_square_with_padding(frame_bgr, x1, y1, x2, y2, padding=self.config.crop_padding)
            if crop is None:
                continue

            crop = cv2.resize(crop, (self.config.classifier_imgsz, self.config.classifier_imgsz), interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
            kept_boxes.append((x1, y1, x2, y2, det_conf))

        predictions = self.classify_crops_batch(crops)
        final_results: list[InferenceResult] = []
        rejected_count = 0

        for (x1, y1, x2, y2, det_conf), (label, cls_conf) in zip(kept_boxes, predictions):
            if label == "background":
                rejected_count += 1
                continue
            if label not in {"drone", "bird"}:
                rejected_count += 1
                continue
            if cls_conf < self.config.classifier_min_conf:
                rejected_count += 1
                continue

            final_results.append(InferenceResult(x1=x1, y1=y1, x2=x2, y2=y2, det_conf=det_conf, cls_conf=cls_conf, label=label))

        return final_results, rejected_count

    def run_two_stage_on_image(self, image_path: Path) -> tuple:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        results, rejected_count = self.run_two_stage(image_bgr, detector_image_size=self.config.detector_imgsz_image)
        return image_bgr, results, rejected_count

    @staticmethod
    def render(image_bgr, results: Iterable[InferenceResult], show_confidence: bool):
        output = image_bgr.copy()
        counts = {"drone": 0, "bird": 0}

        for result in results:
            x1, y1, x2, y2 = map(int, [result.x1, result.y1, result.x2, result.y2])
            label = result.label
            counts[label] = counts.get(label, 0) + 1
            color = (0, 255, 0) if label == "drone" else (255, 0, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{label} d:{result.det_conf:.2f} c:{result.cls_conf:.2f}" if show_confidence else label
            cv2.putText(output, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return output, counts

    def save_image(self, image_path: Path, annotated_bgr) -> Path:
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{image_path.stem}_2stage.jpg"
        cv2.imwrite(str(output_path), annotated_bgr)
        return output_path

class VideoDialog(QDialog):
    def __init__(self, engine: TwoStageEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Test - Two-Stage Drone/Bird Pipeline")
        self.resize(1300, 850)

        self.engine = engine
        self.config = engine.config
        self.viewer = ImageViewer()

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.stop_button)
        top_layout.addWidget(self.status_label)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.viewer)
        self.setLayout(layout)

        self.cap = None
        self.writer = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.next_frame)

        self.running = False
        self.frame_index = 0
        self.first_frame = True
        self.show_confidence = self.config.show_confidence
        self.last_frame_raw = None
        self.last_results: list[InferenceResult] = []
        self.fps_start_time = time.time()
        self.processed_frame_count = 0
        self.frame_interval_ms = 40.0
        self.video_start_wall_time = 0.0
        self.output_path: Path | None = None
        self.profile_history = {key: deque(maxlen=60) for key in ("read", "inference", "render", "display", "write", "total")}

    def add_profile_time(self, key: str, value_seconds: float) -> None:
        self.profile_history[key].append(value_seconds)

    def profile_ms(self, key: str) -> float:
        values = self.profile_history.get(key)
        if not values:
            return 0.0
        return sum(values) / len(values) * 1000.0

    def set_show_confidence(self, show_confidence: bool) -> None:
        self.show_confidence = show_confidence
        if self.last_frame_raw is None:
            return

        annotated_display, counts = self.engine.render(self.last_frame_raw, self.last_results, self.show_confidence)
        if self.config.display_scale != 1.0:
            annotated_display = cv2.resize(
                annotated_display,
                (0, 0),
                fx=self.config.display_scale,
                fy=self.config.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        self.viewer.update_pixmap(bgr_to_qpixmap(annotated_display))
        self.status_label.setText(f"Drone: {counts.get('drone', 0)} | Bird: {counts.get('bird', 0)} | frame={self.frame_index}")

    def start(self, video_path: str, show_confidence: bool) -> None:
        self.show_confidence = show_confidence
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = save_dir / f"{Path(video_path).stem}_2stage.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))

        self.running = True
        self.frame_index = 0
        self.first_frame = True
        self.last_frame_raw = None
        self.last_results = []
        self.fps_start_time = time.time()
        self.processed_frame_count = 0
        self.frame_interval_ms = 1000.0 / fps
        self.video_start_wall_time = time.time()

        self.status_label.setText(f"Running... Saving output to: {self.output_path}")
        self.timer.start(1)

    def stop(self) -> None:
        self.running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None
        self.status_label.setText(f"Stopped. Output: {self.output_path}")

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)

    def next_frame(self) -> None:
        if not self.running or self.cap is None:
            return

        total_timer = time.perf_counter()
        read_timer = time.perf_counter()
        ok, frame = self.cap.read()
        self.add_profile_time("read", time.perf_counter() - read_timer)
        if not ok:
            self.stop()
            return

        self.frame_index += 1
        if self.config.video_frame_skip > 1 and self.frame_index % self.config.video_frame_skip != 0:
            if self.writer is not None:
                self.writer.write(frame)
            elapsed_ms = (time.time() - self.video_start_wall_time) * 1000.0
            next_due_ms = self.frame_index * self.frame_interval_ms
            self.timer.start(max(1, int(next_due_ms - elapsed_ms)))
            return

        elapsed_ms = (time.time() - self.video_start_wall_time) * 1000.0
        expected_index = int(elapsed_ms / self.frame_interval_ms)
        skip_count = expected_index - self.frame_index
        if skip_count > 0:
            for _ in range(min(skip_count, 8)):
                ok, _ = self.cap.read()
                if not ok:
                    self.stop()
                    return
                self.frame_index += 1

        inference_timer = time.perf_counter()
        results, rejected_count = self.engine.run_two_stage(frame, detector_image_size=self.config.detector_imgsz_video)
        self.add_profile_time("inference", time.perf_counter() - inference_timer)

        self.last_frame_raw = frame
        self.last_results = results

        render_timer = time.perf_counter()
        annotated_save, counts = self.engine.render(frame, results, show_confidence=False)
        annotated_display, _ = self.engine.render(frame, results, show_confidence=self.show_confidence)
        if self.config.display_scale != 1.0:
            annotated_display = cv2.resize(
                annotated_display,
                (0, 0),
                fx=self.config.display_scale,
                fy=self.config.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        self.add_profile_time("render", time.perf_counter() - render_timer)

        write_timer = time.perf_counter()
        if self.writer is not None:
            self.writer.write(annotated_save)
        self.add_profile_time("write", time.perf_counter() - write_timer)

        display_timer = time.perf_counter()
        pixmap = bgr_to_qpixmap(annotated_display)
        if self.first_frame:
            self.viewer.set_pixmap_first(pixmap)
            self.first_frame = False
        else:
            self.viewer.update_pixmap(pixmap)
        self.add_profile_time("display", time.perf_counter() - display_timer)
        self.add_profile_time("total", time.perf_counter() - total_timer)

        self.processed_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            processed_fps = self.processed_frame_count / elapsed
            self.fps_start_time = time.time()
            self.processed_frame_count = 0
            target_fps = 1000.0 / self.frame_interval_ms
            self.status_label.setText(
                f"Processing: {processed_fps:.1f} FPS | Target: {target_fps:.1f} FPS | "
                f"Drone: {counts.get('drone', 0)} | Bird: {counts.get('bird', 0)} | "
                f"Rejected: {rejected_count} | Frame: {self.frame_index}\n"
                f"read:{self.profile_ms('read'):.1f}ms  "
                f"inference:{self.profile_ms('inference'):.1f}ms  "
                f"render:{self.profile_ms('render'):.1f}ms  "
                f"display:{self.profile_ms('display'):.1f}ms  "
                f"write:{self.profile_ms('write'):.1f}ms  "
                f"total:{self.profile_ms('total'):.1f}ms"
            )

        elapsed_ms = (time.time() - self.video_start_wall_time) * 1000.0
        next_due_ms = self.frame_index * self.frame_interval_ms
        self.timer.start(max(1, int(next_due_ms - elapsed_ms)))

class MainWindow(QMainWindow):
    def __init__(self, config: GuiConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("Two-Stage Drone/Bird GUI")
        self.resize(1450, 900)
        self.setAcceptDrops(True)

        self.engine = TwoStageEngine(config)
        self.show_confidence = config.show_confidence
        self.last_image_bgr = None
        self.last_results: list[InferenceResult] = []
        self.video_dialog: VideoDialog | None = None

        self.viewer = ImageViewer()
        self.select_image_button = QPushButton("Select Image")
        self.select_image_button.clicked.connect(self.select_images)
        self.select_video_button = QPushButton("Select Video")
        self.select_video_button.clicked.connect(self.select_video)

        self.confidence_checkbox = QCheckBox("Show confidence")
        self.confidence_checkbox.setChecked(self.show_confidence)
        self.confidence_checkbox.stateChanged.connect(self.toggle_confidence)

        self.counts_label = QLabel("Drone: 0\nBird: 0\nRejected: 0")
        self.counts_label.setStyleSheet("font-size: 18px;")
        self.counts_label.setAlignment(Qt.AlignTop)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(170)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.select_image_button)
        controls_layout.addWidget(self.select_video_button)
        controls_layout.addWidget(self.confidence_checkbox)
        controls_layout.addStretch(1)
        controls_group.setLayout(controls_layout)

        counts_group = QGroupBox("Analysis Result")
        counts_layout = QVBoxLayout()
        counts_layout.addWidget(self.counts_label)
        counts_group.setLayout(counts_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(controls_group)
        right_layout.addWidget(counts_group)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.info_text)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMinimumWidth(380)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.viewer, stretch=4)
        main_layout.addWidget(right_widget, stretch=1)

        root_widget = QWidget()
        root_widget.setLayout(main_layout)
        self.setCentralWidget(root_widget)

        self.log_startup()

    def log_startup(self) -> None:
        self.info_text.append("Models loaded successfully.")
        self.info_text.append(f"Detector: {self.config.detector_path}")
        self.info_text.append(f"Classifier: {self.config.classifier_path}")
        self.info_text.append(f"Save dir: {self.config.save_dir}")
        self.info_text.append(f"CUDA: {self.engine.cuda_available} | device={self.engine.device} | half={self.engine.use_half}")
        if self.engine.cuda_available:
            self.info_text.append(f"GPU: {torch.cuda.get_device_name(0)}")
        self.info_text.append(f"Classifier names: {self.engine.classifier_names}")
        self.info_text.append("Zoom: mouse wheel | Fit: double click")
        self.info_text.append("")

    def toggle_confidence(self) -> None:
        self.show_confidence = self.confidence_checkbox.isChecked()
        if self.last_image_bgr is not None:
            annotated, counts = self.engine.render(self.last_image_bgr, self.last_results, show_confidence=self.show_confidence)
            self.viewer.update_pixmap(bgr_to_qpixmap(annotated))
            self.counts_label.setText(f"Drone: {counts.get('drone', 0)}\nBird: {counts.get('bird', 0)}")

        if self.video_dialog is not None and self.video_dialog.isVisible():
            self.video_dialog.set_show_confidence(self.show_confidence)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        image_paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

        if not image_paths:
            QMessageBox.information(self, "Info", "No supported image file found.")
            return

        for image_path in image_paths:
            self.run_on_image(image_path)

    def select_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select image",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        for file in files:
            self.run_on_image(Path(file))

    def run_on_image(self, image_path: Path) -> None:
        try:
            raw_image, results, rejected_count = self.engine.run_two_stage_on_image(image_path)
            self.last_image_bgr = raw_image
            self.last_results = results

            annotated, counts = self.engine.render(raw_image, results, show_confidence=self.show_confidence)
            self.viewer.set_pixmap_first(bgr_to_qpixmap(annotated))
            self.counts_label.setText(
                f"Drone: {counts.get('drone', 0)}\n"
                f"Bird: {counts.get('bird', 0)}\n"
                f"Rejected: {rejected_count}"
            )

            annotated_save, _ = self.engine.render(raw_image, results, show_confidence=False)
            saved_path = self.engine.save_image(image_path, annotated_save)
            self.info_text.append(f"Image: {image_path.name}")
            self.info_text.append(f"Saved: {saved_path}")
            self.info_text.append(f"drone={counts.get('drone', 0)} bird={counts.get('bird', 0)} rejected={rejected_count}")
            self.info_text.append("")

        except Exception as error:
            self.info_text.append("Error:")
            self.info_text.append(str(error))
            self.info_text.append(traceback.format_exc())
            self.info_text.append("")

    def select_video(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Select video", str(Path.home()), VIDEO_FILTER)
        if not file:
            return
        self.video_dialog = VideoDialog(self.engine, parent=self)
        self.video_dialog.show()
        self.video_dialog.start(file, show_confidence=self.show_confidence)


def run_app(config: GuiConfig) -> None:
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())
