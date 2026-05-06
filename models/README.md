# Models

This directory is reserved for trained model weights used by the two-stage drone-vs-bird detection pipeline.

The trained model weights are **not included** in this repository.

## Why Are the Model Weights Not Included?

The model weights are not publicly redistributed for the following reasons:

- the models were trained using a curated project-specific dataset,
- part of the data preparation process involved restricted data obtained through a formal Data Usage Agreement,
- the original dataset sources may have redistribution restrictions,
- TensorRT engine files are hardware-specific,
- public redistribution of trained weights may not be appropriate without reviewing the intended use case.

For these reasons, this repository provides the implementation, configuration files, GUI pipeline, methodology, and evaluation summaries, but does not directly include trained `.pt`, `.onnx`, `.engine`, or other model weight files.

---

## Expected Model Files

If you have access to the trained models or train your own models, place them in this directory using the following names:

```text
models/
├── detector_yolov26m_target.pt
└── classifier_yolo11m_wbg.pt
```

These filenames match the default paths defined in:

```text
configs/gui_config.yaml
```

---

## Model Descriptions

| File | Task | Description |
|---|---|---|
| `detector_yolov26m_target.pt` | Object Detection | YOLOv26m detector trained with a single `target` class |
| `classifier_yolo11m_wbg.pt` | Image Classification | YOLO11m-cls classifier trained with `background`, `bird`, and `drone` classes |

---

## Detector Model

The detector model is trained as a single-class object detector.

```text
0: target
```

Both drones and birds are represented as generic flying targets during detector training. The detector is responsible for localizing candidate flying objects, while the final semantic decision is handled by the second-stage classifier.

---

## Classifier Model

The classifier model receives cropped detector outputs and predicts one of the following classes:

```text
background
bird
drone
```

The `background` class is used to reject false-positive detector outputs such as sky regions, tree edges, grass, shadows, blurred areas, and other non-target regions.

During inference:

```text
detector output → crop extraction → classifier prediction
```

If the classifier predicts `background`, the detection is rejected and not displayed as a valid drone or bird target.

---

## Requesting Model Weights

The trained model weights may be shared upon reasonable request for academic, research, or non-commercial evaluation purposes.

To request access, please contact the author by email:

```text
<your-email@example.com>
```

Please include the following information in your request:

- your name and affiliation,
- intended use case,
- whether the models will be used for research, education, or evaluation,
- confirmation that the models will not be redistributed without permission.

Access is not guaranteed and may depend on dataset usage restrictions, project constraints, and the intended purpose of use.

---

## TensorRT Engine Files

TensorRT `.engine` files are not included in this repository.

TensorRT engines are usually hardware-specific and should be generated on the target machine. A TensorRT engine generated on one GPU or TensorRT version may not work correctly on another system.

Recommended workflow:

```text
.pt → ONNX / TensorRT export → hardware-specific .engine file
```

If TensorRT deployment is required, generate the engine files locally using the target device, CUDA version, and TensorRT version.

---

## Notes

- Do not commit trained model weights to this repository.
- Do not commit TensorRT `.engine` files.
- Do not commit large model artifacts.
- Use this directory only for local model placement.
- The repository is designed to run once the expected model files are placed in this folder.
