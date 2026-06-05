# Datasets for Autonomous Driving Neural Network Verification

This directory houses the driving datasets required for physical weather characterization and steering model verification: the **ACDC** dataset (for adverse weather contrast and brightness limits) and the **Udacity** dataset (for steering regression verification).

---

## 1. ACDC Dataset (Adverse Conditions Dataset with Correspondences)

ACDC is used to extract empirical contrast drop ($\epsilon_c$) and brightness bias ($\epsilon_b$) coefficients. It provides GPS-synchronized image pairs: an adverse weather image and its corresponding clear-weather reference image.

### Expected Directory Hierarchy
```text
ACDC/
├── rgb_anon/
│   ├── [fog|night|snow|rain]/
│   │   ├── train/
│   │   │   └── <sequence_folder>/ (e.g., GOPR0476/)
│   │   │       └── <sequence_frame>_rgb_anon.png
│   │   ├── val/
│   │   │   └── <sequence_folder>/
│   │   │       └── <sequence_frame>_rgb_anon.png
│   │   ├── train_ref/
│   │   │   └── <sequence_folder>/
│   │   │       └── <sequence_frame>_rgb_ref_anon.png
│   │   └── val_ref/
│   │       └── <sequence_folder>/
│   │           └── <sequence_frame>_rgb_ref_anon.png
└── gt/
    ├── [fog|night|snow|rain]/
    │   ├── train/
    │   │   └── <sequence_folder>/
    │   │       └── <sequence_frame>_gt_labelTrainIds.png
    │   └── val/
    │       └── <sequence_folder>/
    │           └── <sequence_frame>_gt_labelTrainIds.png
```

### File Mapping Syntax
For each frame file inside the adverse folder (e.g. `val/GOPR0402/GOPR0402_frame_000120_rgb_anon.png`), the characterizer maps it to:
1.  **Clear Reference Frame:** `val_ref/GOPR0402/GOPR0402_frame_000120_rgb_ref_anon.png`
2.  **Semantic Mask Frame:** `gt/rain/val/GOPR0402/GOPR0402_frame_000120_gt_labelTrainIds.png`

### Spatial Semantic Masking (Snow & Rain)
Global atmospheric changes like Fog or Night affect the entire image uniformly. Localized road conditions (like Snow on asphalt or wet Rain reflection mirrors) only affect the road surface.
*   The ground truth masks use the **Cityscapes TrainID format**.
*   **TrainID 0 represents the Road category.**
*   Our characterization script isolates pixels where `labelTrainIds == 0` to compute standard deviation and mean only on the drivable road corridor, preventing sky or building details from corrupting the coefficients.

---

## 2. Udacity Driving Dataset (Self-Driving Car Behavioral Cloning)

Used as the regression verification baseline. It contains continuous steering sequences recorded from a virtual simulator.

### Expected Directory Hierarchy
```text
Udacity/
├── self_driving_car_dataset_jungle/
│   ├── driving_log.csv
│   └── IMG/
│       ├── center_*.jpg
│       ├── left_*.jpg
│       └── right_*.jpg
└── self_driving_car_dataset_make/
    ├── driving_log.csv
    └── IMG/
        └── ...
```

### CSV File Format
The `driving_log.csv` has no header and is structured as:
```csv
center_img_path, left_img_path, right_img_path, steering_angle, throttle, brake, speed
```
*   **Column 0:** Path to the center camera frame image (loaded for verification).
*   **Column 3:** Floating-point steering angle (radians).

### Image Preprocessing and Cropping
To match NVIDIA's native PilotNet architecture, the `UdacityDataset` loader performs the following steps:
1.  **Luminance normalization:** Converts values to float range `[0.0, 1.0]`.
2.  **Corridor cropping:** Raw simulator frames are $320 \times 160$ pixels. The top $60$ pixels (containing sky and trees) and bottom $25$ pixels (containing the car hood) are cropped out to isolate only the road lanes.
3.  **Resizing:** Resizes the resulting $320 \times 75$ crop to the downsampled target resolution of $37 \times 117$ pixels, reducing the verifier VRAM/memory footprint.
