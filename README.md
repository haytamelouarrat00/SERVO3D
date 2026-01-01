# SERVO3D

## Overview

SERVO3D is a Python framework for simulating image-based visual servoing (IBVS) of a camera inside 3D scenes. It renders views from a GLB scene with Open3D, detects and matches visual features between a desired and current view, backprojects matched points with depth, and computes a camera velocity command via the interaction matrix (image Jacobian). A control loop updates the camera pose until the pixel error converges, while plotting the camera view, error norm, velocity commands, and 3D trajectory. The code is organized into modules for scene loading, camera intrinsics/pose handling, feature matching and filtering, and control law computation, with `main.py` wiring the end-to-end simulation.

## Research Motivation

Visual servoing is a core capability for robotics, AR/VR, and autonomous inspection/navigation, yet many IBVS implementations are hard to reproduce because they depend on hardware rigs or proprietary simulators. SERVO3D focuses on a lightweight, reproducible setup that ties together rendering, feature tracking, depth-based geometry, and control. The goal is to make it easy to prototype and analyze IBVS behavior across varied 3D scenes, test sensitivity to feature quality and camera pose, and iterate on control laws before moving to real systems. The motivation behind this framework is the need to find a proper way to combine correspondence filtering with navigating in simulated environments and.

## Problem Statement

The core problem is selecting the best feature correspondences between a desired view and the current view so that the target set used by the IBVS controller is high quality, geometrically consistent, and stable. Poor or ambiguous matches lead to noisy depth backprojections and unreliable image Jacobians, which degrade servoing accuracy and make comparisons across runs untrustworthy. This framework focuses on improving correspondence quality to produce robust target features for control and fair evaluation.

## Background and Related Work

IBVS (Image-Based Visual Servoing) is a control method that moves a camera by directly minimizing errors between current and the desired image features, using only image-space measurements without explicitly reconstructing 3D pose.

## Methodology

#### Feature Detection & Matching

- **SIFT features** extracted from desired and current camera views
- **FLANN-based matching** with Lowe's ratio test (threshold = 0.75)
- **RANSAC filtering** to reject outliers via homography estimation
- **Non-collinearity check** ensures the selected 4 points form a valid quadrilateral
- **Reprojection error filtering** validates geometric consistency using depth maps

#### Visual Servoing Control

- **Image-based control**: Error computed in normalized image coordinates (not 3D space)
- **Interaction matrix $L$**: Built from 3D point positions $(X, Y, Z)$ in the camera frame
- $L = f(x, y, Z) \in \mathbb{R}^{2N \times 6}$, relating pixel velocity to camera velocity
- **Velocity computation**:

$$v = -\lambda \, L^{+} \, e$$

using a damped least-squares pseudoinverse

- **6-DOF control output**:

$$[v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]$$

expressed in the camera frame

#### Depth Handling

- **Depth maps** rendered from the 3D scene provide ground-truth $Z$ values
- **Per-iteration depth update**: $Z$ refreshed from the current view's depth map
- **Backprojection**: 2D pixels + depth → 3D points in the camera frame

#### Pose Update

- **Exponential integration**: Velocity applied over timestep $dt$
- **Frame transformation**: Camera-frame velocities converted to world frame for translation update
- **Rotation update**: Incremental rotation composed with the current orientation

#### Convergence

- **Error metric**: Norm of pixel error in normalized image coordinates
- **Termination criterion**:
  - Stop if error < threshold, or
  - Maximum number of iterations reached

## Correspondence Filtering Methodology for Image-Based Visual Servoing

### 1. Initial Feature Detection and Matching

We employ **SIFT** for keypoint detection and description due to its robustness to scale, rotation, and illumination changes. Feature matching is performed using **FLANN** with a KD-tree index structure (`algorithm=1`, `trees=5`). To reject ambiguous matches, we apply **Lowe's ratio test** with threshold $\tau = 0.75$:

$$\text{match accepted} \iff d(m_1) < \tau \cdot d(m_2)$$

where $d(m_1)$ and $d(m_2)$ are distances to the first and second nearest neighbors, respectively. This eliminates matches where the best match is not significantly better than the second-best, indicating potential ambiguity in textureless or repetitive regions.

**Rationale:** Raw SIFT matching produces many false correspondences, especially in repetitive textures or similar local structures. The ratio test rejects matches that lack discriminative power.

---

### 2. Geometric Verification via Reprojection Error

Since depth is available from rendered depth maps, we enforce geometric consistency using **3D reprojection error filtering**. For each candidate match $(p_1, p_2)$ between reference and target images:

#### Step 2.1: Backprojection to 3D

Using the depth map $D_1$ from the reference view and intrinsics $(f_x, f_y, c_x, c_y)$, backproject the 2D keypoint to 3D in the reference camera frame:

$$X = (u_1 - c_x)\frac{Z}{f_x},\quad Y = (v_1 - c_y)\frac{Z}{f_y},\quad Z = D_1[v_1, u_1]$$

#### Step 2.2: Coordinate Transformation

Compute the relative transformation $(R_{\text{rel}}, t_{\text{rel}})$ between known camera extrinsics:

$$R_{\text{rel}} = R_{\text{target}}^T R_{\text{reference}},\quad t_{\text{rel}} = R_{\text{target}}^T (t_{\text{reference}} - t_{\text{target}})$$

Transform the 3D point into the target camera frame:

$$P_{\text{target}} = R_{\text{rel}} P_{\text{reference}} + t_{\text{rel}}$$

#### Step 2.3: Reprojection and Error Computation

Project onto the target image plane:

$$u' = f_x \frac{X_{\text{target}}}{Z_{\text{target}}} + c_x,\quad v' = f_y \frac{Y_{\text{target}}}{Z_{\text{target}}} + c_y$$

Compute reprojection error:

$$e_{\text{reproj}} = \left\| \begin{bmatrix}u'\\v'\end{bmatrix} - \begin{bmatrix}u_2\\v_2\end{bmatrix} \right\|_2$$

Keep matches only if:

$$e_{\text{reproj}} < \text{threshold} \quad (\text{default: } 200\text{ px})$$

**Rationale:** With known poses and depth, a correct match should reproject near its counterpart. This rejects:

- mismatches passing the ratio test,
- correspondences with erroneous depth,
- points on dynamic/incorrectly reconstructed surfaces.

---

### 3. RANSAC-Based Homography Filtering

After reprojection filtering, apply **RANSAC** with a homography model.

#### Step 3.1: Hypothesis Generation

Randomly sample 4 correspondences and compute a homography $H$ (DLT).

#### Step 3.2: Consensus Scoring

For all correspondences, compute transfer error:

$$e_H = \|p_2 - H p_1\|_2$$

Classify inliers if:

$$e_H < 5.0\text{ px}$$

#### Step 3.3: Iteration

Repeat for $N$ iterations, keep the homography with the largest inlier set.

**Rationale:** This 2D filter catches outliers that may survive reprojection checks due to:

- depth noise/quantization,
- near-degenerate geometry,
- numerical instability in pose/transform computations.

---

### 4. Non-Collinearity Constraint for Point Selection

IBVS requires 4 correspondences in a non-degenerate configuration. Collinear points lead to a rank-deficient interaction matrix and control singularities.

#### Collinearity Test

For any triplet $(p_1,p_2,p_3)$, compute signed triangle area:

$$A = \frac{1}{2}\left|x_1(y_2-y_3) + x_2(y_3-y_1) + x_3(y_1-y_2)\right|$$

A set is considered collinear if any triplet satisfies:

$$A < \varepsilon \quad (\text{default: } 10^{-3})$$

#### Selection Strategy

1. Sort matches by descriptor distance (lower is better).
2. Iterate over combinations of 4 points from top candidates.
3. Select the first combination where both reference and target sets are non-collinear.
4. Re-index selected matches to $[0,1,2,3]$.

**Rationale:** The IBVS interaction matrix for a point $(x,y,Z)$ has the form:

$$L = \begin{bmatrix} -\frac{1}{Z} & 0 & \frac{x}{Z} & xy & -(1+x^2) & y \\ 0 & -\frac{1}{Z} & \frac{y}{Z} & 1+y^2 & -xy & -x \end{bmatrix}$$

Collinearity makes rows of $L$ linearly dependent, causing $\text{rank}(L) < 6$ and an ill-conditioned $L^+$, leading to:

- unstable/unbounded velocity commands,
- loss of controllability,
- numerical instability.

---

## Summary of Filtering Pipeline

| Stage | Method | Rejects |
|------:|-----------------------------|-----------------------------------------|
| 1 | Ratio Test ($\tau=0.75$) | Ambiguous matches in repetitive regions |
| 2 | Reprojection Error (<200 px)| Geometrically inconsistent matches |
| 3 | RANSAC Homography | 2D outliers, depth-related errors |
| 4 | Non-collinearity | Degenerate point configurations |

This multi-stage filtering ensures the final 4 correspondences are:

- **Discriminative** (good SIFT matches),
- **Geometrically consistent** (valid under known geometry),
- **Robust** (inliers to dominant 2D model),
- **Well-conditioned** (stable IBVS control).

## Repository Structure

```
nservo/
├── main.py                     # Entry point: end-to-end IBVS simulation loop
├── README.md                   # Project documentation
├── .gitignore
│
├── src/                        # Core modules
│   ├── camera.py               # Camera model: intrinsics, pose, projection/backprojection
│   ├── control.py              # Control laws: interaction matrix, velocity computation, 2½D VS
│   ├── features.py             # Feature detection, matching, filtering pipeline
│   ├── features_special.py     # Alternative Matcher class with extended filtering
│   ├── scene.py                # GLB scene loading via Open3D
│   └── utils.py                # Utility functions
│
├── data/                       # 3D scene assets
│   ├── modular_environment.glb
│   └── japanese_street_at_night.glb
│
└── tests/                      # Unit tests
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `main.py` | Orchestrates the simulation: renders views, extracts features, runs control loop, visualizes results |
| `camera.py` | Pinhole camera model with pose updates, view/depth rendering, projection and backprojection |
| `control.py` | Interaction matrix computation, IBVS velocity law, 2½D hybrid control, SO(3) utilities |
| `features.py` | SIFT+FLANN matching, ratio test, reprojection filtering, RANSAC, non-collinearity selection |
| `features_special.py` | Object-oriented `Matcher` class with depth querying and geometric filtering |
| `scene.py` | Loads GLB/GLTF 3D models into Open3D for rendering |

## Data Description

Files should be loaded as `.ply`, `.glb` or another compatible format.

## Experimental Setup

First start by setting up your conda env:

```bash
conda env create -f environment.yml
conda activate servo3d
```

To run the simulation:

```bash
python main.py
```

In order to run the simulation with CLI debugging use the `--debug` or the `-d` flag:

```bash
python main.py -d #or --debug
```

This will output a line with the following details:

**Iteration X:** Error = $e$, $\lVert v \rVert = v$

## Results

Results are displayed in a real-time visualization widget with 4 panels:

#### 1. Current Camera View

Displays the rendered image from the current camera pose with overlaid feature markers:

- **Green X markers**: Current feature positions in the image
- **Red circle markers**: Desired/target feature positions

As the servo converges, the green markers move toward the red circles.

#### 2. Error Norm Evolution

Plots the pixel error norm (in normalized image coordinates) over iterations:

- **Blue line**: Error trajectory
- **Red dashed line**: Convergence threshold

The error should decrease monotonically toward the threshold.

#### 3. Velocity Commands Evolution

Displays all 6 velocity components over time:

- **vx, vy, vz**: Linear velocities (camera frame)
- **ωx, ωy, ωz**: Angular velocities (camera frame)

Useful for diagnosing control behavior—oscillations or large spikes indicate potential issues.

#### 4. 3D Camera Trajectory

Visualizes the camera path in world coordinates:

- **Blue line with dots**: Camera trajectory
- **Green star**: Initial pose
- **Red star**: Desired pose
- **Blue dot**: Current pose

Shows the characteristic arc-like path of IBVS (due to image-space optimization) versus the straight-line path of position-based methods.

![Demo](simulation.gif)

### Console Output

Iteration 1: Error = 0.73618, ||v|| = 42.8159
Iteration 2: Error = 0.69937, ||v|| = 40.6751
Iteration 3: Error = 0.66440, ||v|| = 38.6414
Iteration 4: Error = 0.63118, ||v|| = 36.7093
Iteration 5: Error = 0.59962, ||v|| = 34.8738
Iteration 6: Error = 0.56964, ||v|| = 33.1301
Iteration 7: Error = 0.54116, ||v|| = 31.4736
Iteration 8: Error = 0.51410, ||v|| = 29.9000
Iteration 9: Error = 0.48840, ||v|| = 24.7860
Iteration 10: Error = 0.46405, ||v|| = 23.8162
Iteration 11: Error = 0.44093, ||v|| = 22.8499
Iteration 12: Error = 0.41898, ||v|| = 21.9278
Iteration 13: Error = 0.39812, ||v|| = 21.0232
Iteration 14: Error = 0.37832, ||v|| = 20.1342
Iteration 15: Error = 0.35951, ||v|| = 19.2365
Iteration 16: Error = 0.34164, ||v|| = 18.4487
Iteration 17: Error = 0.32466, ||v|| = 20.2241
Iteration 18: Error = 0.30848, ||v|| = 19.1472
Iteration 19: Error = 0.29311, ||v|| = 18.1486
Iteration 20: Error = 0.27850, ||v|| = 17.1974
...

Upon convergence or termination:

```
✓ Converged in N iterations!
Final error: X.XXXXX pixels
Final pose: [x, y, z, rx, ry, rz]
Desired pose: [x, y, z, rx, ry, rz]
```

Or if max iterations reached:

```
✗ Did not converge after N iterations
Final error: X.XX pixels
```

## Limitations

## Future Work

## Contact

For any questions, please contact me on <eo.haytam@gmail.com>
