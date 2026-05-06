# Data

## Important Dataset Notice

The full dataset used in this project is **not included** in this repository.

This project used the WOSDETC / Drone-vs-Bird Detection Challenge dataset as one of the data sources. The WOSDETC dataset was obtained through a formal Data Usage Agreement and therefore cannot be redistributed publicly.

Users who want to access the WOSDETC dataset should refer to the official challenge repository:

```text
https://github.com/wosdetc/challenge
```

According to the official WOSDETC repository, dataset access requests should be sent to:

```text
wosdetc@googlegroups.com
```

Users are required to sign a Data Usage Agreement before using the dataset for research purposes.

This repository does **not** contain:

- raw videos,
- extracted frames,
- original dataset images,
- restricted dataset annotations,
- dataset download links,
- signed agreement documents,
- private or restricted dataset files.

Only dataset structure descriptions, methodology, training configuration, evaluation summaries, and implementation files are provided.

---

## Dataset Overview

This project uses two dataset structures:

1. A single-class object detection dataset for candidate flying target localization.
2. A crop-based image classification dataset for drone / bird / background classification.

The detector and classifier datasets are not distributed with this repository. The following sections only describe the dataset organization used during development.

---

## Detector Dataset

The detector dataset was prepared as a single-class object detection dataset.

Instead of directly separating drones and birds at the detection stage, both object types were represented as a generic flying target. This allows the detector to focus on candidate target localization, while the final drone / bird / background decision is handled by the second-stage classifier.

### Class Definition

```text
0: target
```

### Split Summary

| Split | Images |
|---|---:|
| Train | 7,044 |
| Validation | 1,244 |
| Test | 511 |

### Expected YOLO Detection Format

```text
dataset_detector/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

Example `data.yaml`:

```yaml
path: dataset_detector
train: images/train
val: images/val
test: images/test

names:
  0: target
```

---

## Classifier Dataset

The classifier dataset was created from cropped detector regions.

In addition to drone and bird crops, false-positive detector outputs were mined and added as a third `background` class. This allows the second-stage classifier to reject detector false positives during inference.

### Class Definition

```text
background
bird
drone
```

### Training Split

| Class | Images |
|---|---:|
| Background | 2,000 |
| Bird | 4,601 |
| Drone | 5,338 |
| **Total** | **11,939** |

### Validation Split

| Class | Images |
|---|---:|
| Background | 300 |
| Bird | 817 |
| Drone | 905 |
| **Total** | **2,022** |

### Separate Evaluation Set

| Class | Images |
|---|---:|
| Background | 850 |
| Bird | 810 |
| Drone | 901 |
| **Total** | **2,561** |

### Expected Classification Format

```text
dataset_classifier/
├── train/
│   ├── background/
│   ├── bird/
│   └── drone/
│
├── val/
│   ├── background/
│   ├── bird/
│   └── drone/
│
└── evaluation/
    ├── background/
    ├── bird/
    └── drone/
```

---

## Background Class Mining

The `background` class was created by collecting false-positive detector outputs.

Typical background false positives include:

- sky texture,
- tree branches,
- grass,
- shadows,
- building edges,
- blurred image regions,
- non-target background objects.

These false-positive regions were cropped and added to the classifier dataset as negative samples.

During inference, if the classifier predicts `background`, the detection is rejected and not displayed as a valid drone or bird target.

---

## Notes

- Raw datasets are not distributed with this repository.
- Dataset files should not be committed to GitHub.
- Local dataset folders are ignored by `.gitignore`.
- The repository focuses on the training, inference, GUI, methodology, and documentation pipeline.
- Users who want to reproduce the training process should obtain an appropriate dataset and organize it according to the structures described above.
