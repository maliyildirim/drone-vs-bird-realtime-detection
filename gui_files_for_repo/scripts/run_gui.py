"""
Command-line entry point for the PySide6 two-stage drone-vs-bird GUI.

Example:
    python scripts/run_gui.py \
        --detector models/detector_yolov26m_target.pt \
        --classifier models/classifier_yolo11m_wbg.pt \
        --save-dir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.two_stage_gui import GuiConfig, run_app  # noqa: E402


def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration format: {path}")

    return data


def build_config(args: argparse.Namespace) -> GuiConfig:
    config_data = load_yaml_config(args.config)

    detector_cfg = config_data.get("detector", {})
    classifier_cfg = config_data.get("classifier", {})
    video_cfg = config_data.get("video", {})
    gui_cfg = config_data.get("gui", {})

    return GuiConfig(
        detector_path=args.detector or detector_cfg.get("model_path", "models/detector_yolov26m_target.pt"),
        classifier_path=args.classifier or classifier_cfg.get("model_path", "models/classifier_yolo11m_wbg.pt"),
        save_dir=args.save_dir or gui_cfg.get("save_dir", "outputs"),
        detector_conf=float(args.det_conf if args.det_conf is not None else detector_cfg.get("confidence_threshold", 0.30)),
        detector_iou=float(args.det_iou if args.det_iou is not None else detector_cfg.get("iou_threshold", 0.60)),
        detector_max_det=int(args.max_det if args.max_det is not None else detector_cfg.get("max_detections", 80)),
        detector_imgsz_image=int(detector_cfg.get("image_size_image", 960)),
        detector_imgsz_video=int(detector_cfg.get("image_size_video", 640)),
        min_box_size_px=int(detector_cfg.get("min_box_size_px", 7)),
        classifier_imgsz=int(classifier_cfg.get("image_size", 224)),
        classifier_min_conf=float(args.cls_conf if args.cls_conf is not None else classifier_cfg.get("min_confidence", 0.60)),
        crop_padding=float(classifier_cfg.get("crop_padding", 0.35)),
        video_frame_skip=int(video_cfg.get("frame_skip", 1)),
        display_scale=float(video_cfg.get("display_scale", 0.75)),
        show_confidence=bool(gui_cfg.get("show_confidence", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PySide6 GUI for the two-stage drone-vs-bird detection pipeline."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "gui_config.yaml",
        help="Path to GUI configuration YAML file.",
    )
    parser.add_argument("--detector", type=str, default=None, help="Path to YOLO detector model.")
    parser.add_argument("--classifier", type=str, default=None, help="Path to YOLO classifier model.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory where outputs will be saved.")
    parser.add_argument("--det-conf", type=float, default=None, help="Detector confidence threshold.")
    parser.add_argument("--det-iou", type=float, default=None, help="Detector NMS IoU threshold.")
    parser.add_argument("--cls-conf", type=float, default=None, help="Classifier confidence threshold.")
    parser.add_argument("--max-det", type=int, default=None, help="Maximum detections per frame.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_app(config)


if __name__ == "__main__":
    main()
