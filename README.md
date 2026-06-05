# SDP-CROWN: Neural Network Verification for Automated Driving

This repository extends the **SDP-CROWN** (Semidefinite Programming based CROWN) neural network verifier to safety-critical autonomous driving regression tasks. Specifically, it enables physical perturbation verification for end-to-end steering controllers (like NVIDIA's PilotNet) under adverse weather conditions (fog, night, snow, rain).

The physical perturbations are modeled as parameterized **Semantic Perturbation Layers** (controlling contrast scaling $\epsilon_c$ and brightness bias $\epsilon_b$), calibrated directly from real-world GPS-synchronized clear vs. adverse weather driving scenes in the **ACDC dataset**.

---

## 1. Quickstart & Walkthrough

Here is a step-by-step guide to running the automated driving verification pipeline.

### Step 1: Activate the Python Environment
The pipeline relies on PyTorch, auto_LiRPA, OpenCV, and Pandas. You can run all scripts using the pre-configured virtual environment:
```bash
# From the repository root (sdp-crown-automated-driving)
../nn_verification/venv_sdp/bin/python3 <script_name>.py [arguments]
```

### Step 2: Characterize Adverse Weather (Generate $\epsilon$ Bounds)
Run the standalone calibration script to calculate physical contrast drop and brightness bias bounds from the **ACDC dataset**.

You can run this on a single drive split sequence to test a specific image grouping:
```bash
../nn_verification/venv_sdp/bin/python3 tools/extract_physics_bounds.py \
    --condition rain \
    --sequence GOPR0402 \
    --max_images 5
```
*   **What to expect:** The script will pair the rain images from `GOPR0402` with clear-weather reference frames, isolate road pixels using ground truth semantic masks, and output the derived bounds:
    ```text
    ==================================================
    ========= RAIN CHARACTERIZATION RESULTS ==========
    Total processed image pairs: 5
    Contrast Drop (eps_c) range:    [0.3623, 1.4435]
    Brightness Bias (eps_b) range:  [-0.2614, 0.0369]
    ==================================================
    ```
*   **Output:** The recommended mathematically valid intervals (containing `0.0` as the nominal baseline) are saved to `results/physics_bounds.json`.

### Step 3: Run CROWN Steering Verification
Verify the pre-trained `MicroPilotNet` steering network on the Udacity dataset using the derived bounds.

*Because dense matrix bounds tracking requires significant memory, we provide a `--device cpu` flag to run verification on host memory and avoid GPU Out-Of-Memory (OOM) errors.*

#### Test A: Using Calibration Bounds (Sequence specific)
```bash
../nn_verification/venv_sdp/bin/python3 verify_steering.py \
    --weather rain \
    --bounds_file results/physics_bounds.json \
    --num_frames 5 \
    --device cpu
```

#### Test B: Using Paper-Calibrated Bounds (Recommended)
Verify safety under the exact bounds reported in the study (which use tighter margins for global stability):
```bash
../nn_verification/venv_sdp/bin/python3 verify_steering.py \
    --weather rain \
    --eps_c_min -0.0279 \
    --eps_c_max 0.0 \
    --eps_b_min 0.0 \
    --eps_b_max 0.1003 \
    --num_frames 5 \
    --device cpu
```
*   **What to expect:** The verifier computes worst-case steering output bounds ($\theta_{min}, \theta_{max}$) for each frame under the weather disturbance, and checks if they stay within a safety corridor of $\pm 0.1$ radians (approx. $2.5^\circ$) around the nominal clear steering path:
    ```text
    Verifying 5 frames of sequence...
    =======================================================
    ======== Starting RAIN AV Steering Verification =======
    Safety Corridor: +/- 0.1 rad deviation from nominal path
    =======================================================
    Frame 00: Nominal: -0.0100 | Bounds: [-0.0147, -0.0083] | Corridor: [-0.1100, +0.0900] | SAFE
    Frame 01: Nominal: -0.0048 | Bounds: [-0.0099, -0.0013] | Corridor: [-0.1048, +0.0952] | SAFE
    ...
    =======================================================
    ========= Final RAIN Certified Safety: 100.0% =========
    =======================================================
    ```
*   **Output:** Saves verification results to `results/verification_results.json`.

### Step 4: Plot and Visualize Results
Generate a professional graph plotting nominal steering, the safety corridor, CROWN worst-case steering bounds, and frame status:
```bash
../nn_verification/venv_sdp/bin/python3 tools/plot_verification.py
```
*   **Output:** Creates a high-resolution chart at `results/verification_plot.png`.

---

## 2. Directory Structure

*   `tools/`
    *   `extract_physics_bounds.py`: Stands alone to run ACDC contrast and brightness statistical characterizations.
    *   `plot_verification.py`: Stands alone to plot safety corridor vs. CROWN boundaries.
*   `auto_LiRPA/`: Core Linear Bound Propagation library.
*   `datasets/`: Directory containing ACDC and Udacity driving datasets.
*   `models_weights/`: Pretrained model weights (e.g., `pilotnet_udacity.pth`).
*   `models.py`: Network definitions (including classification baselines and `MicroPilotNet`).
*   `semantic_layers.py`: Implements custom PyTorch weather layers and network wrapper module.
*   `udacity_dataset.py`: Udacity image loader, crops the top (sky) and bottom (hood) before resizing.
*   `verify_steering.py`: Runs regression verifier over continuous driving frames.

---

## 3. Core Baseline Verification (MNIST & CIFAR)

To run baseline classification verification (L2-norm radius perturbations) to reproduce original academic benchmarks:

```bash
# Reproducing results in the SDP-CROWN paper (re-scales and runs SDP-CROWN)
./run_sdp_crown.sh
```
Or execute individual models:
```bash
../nn_verification/venv_sdp/bin/python3 sdp_crown.py --model mnist_mlp --radius 1.0
../nn_verification/venv_sdp/bin/python3 sdp_crown.py --model cifar10_cnn_a --radius 24/255
```

---

## References

*   **SDP-CROWN Paper:** [ICML 2025 PDF](https://arxiv.org/pdf/2506.06665)
*   **ACDC Dataset:** [ETH Zurich](https://acdc.vision.ee.ethz.ch/)
*   **auto_LiRPA Engine:** [GitHub](https://github.com/KaidiXu/auto_LiRPA)