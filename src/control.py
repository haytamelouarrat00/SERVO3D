import numpy as np

from src.camera import Camera


def normalize_pixels(cam, pixels_uv):
    """pixels_uv: (N,2) in pixels -> (N,2) normalized"""
    pixels_uv = np.asarray(pixels_uv, dtype=np.float64)
    x = (pixels_uv[:, 0] - cam.cx) / cam.fx
    y = (pixels_uv[:, 1] - cam.cy) / cam.fy
    return np.stack([x, y], axis=1)


def compute_error(current_features, desired_features):
    current_features = np.asarray(current_features, dtype=np.float64)
    desired_features = np.asarray(desired_features, dtype=np.float64)
    assert current_features.shape == desired_features.shape

    e = (current_features - desired_features).reshape(-1, 1)
    return float(np.linalg.norm(e)), e


def compute_current_L(points: np.ndarray):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    Z_inv = 1.0 / Z
    x = X * Z_inv
    y = Y * Z_inv

    n = len(Z)
    L = np.zeros((2 * n, 6))

    L[0::2, 0] = -Z_inv
    L[0::2, 2] = x * Z_inv
    L[0::2, 3] = x * y
    L[0::2, 4] = -(1 + x**2)
    L[0::2, 5] = y

    L[1::2, 1] = -Z_inv
    L[1::2, 2] = y * Z_inv
    L[1::2, 3] = 1 + y**2
    L[1::2, 4] = -x * y
    L[1::2, 5] = -x

    return L


def compute_desired_L(target_points: np.ndarray):
    return compute_current_L(target_points)


def compute_normalized_L(
    cam: Camera, current_points: np.ndarray, desired_points: np.ndarray
):
    L_current = compute_current_L(current_points)
    L_desired = compute_desired_L(desired_points)
    L_normalized = 0.5 * (L_current + L_desired)
    return L_normalized


def velocity(current_features, desired_features, L, cam, lambda_gain=0.5):
    current_features = normalize_pixels(cam, current_features)
    desired_features = normalize_pixels(cam, desired_features)
    err_norm, error = compute_error(current_features, desired_features)
    L_pseudo_inv = np.linalg.pinv(L)
    v = -lambda_gain * L_pseudo_inv @ error
    return v.flatten(), err_norm
