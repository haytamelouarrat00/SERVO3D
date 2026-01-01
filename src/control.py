import numpy as np

from src.camera import Camera


def compute_error(current_features, desired_features):
    assert current_features.shape == desired_features.shape, (
        "Feature arrays must have the same shape."
    )
    current_features = current_features.reshape(-1, 1)
    desired_features = desired_features.reshape(-1, 1)
    error = current_features - desired_features
    error_norm = np.linalg.norm(error)
    return error_norm, error


def compute_current_L(cam: Camera, points: np.ndarray):
    """
    Computes the interaction matrix L for the current features.

    Args:
        cam: Camera object with intrinsic parameters
        points: Nx3 array of 3D points in camera coordinates
    Returns:
        L: Interaction matrix of shape (2N, 6)
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    fx = float(cam.fx)
    fy = float(cam.fy)

    Z_inv = 1.0 / Z
    X_over_Z = X * Z_inv
    Y_over_Z = Y * Z_inv

    n_points = points.shape[0]
    L = np.zeros((2 * n_points, 6), dtype=np.float64)

    L[0::2, 0] = -fx * Z_inv
    L[0::2, 1] = 0.0
    L[0::2, 2] = fx * X_over_Z * Z_inv
    L[0::2, 3] = fx * X_over_Z * Y_over_Z
    L[0::2, 4] = -fx * (1.0 + X_over_Z**2)
    L[0::2, 5] = fx * Y_over_Z

    L[1::2, 0] = 0.0
    L[1::2, 1] = -fy * Z_inv
    L[1::2, 2] = fy * Y_over_Z * Z_inv
    L[1::2, 3] = fy * (1.0 + Y_over_Z**2)
    L[1::2, 4] = -fy * X_over_Z * Y_over_Z
    L[1::2, 5] = -fy * X_over_Z

    return L


def compute_desired_L(cam: Camera, target_points: np.ndarray):
    return compute_current_L(cam, target_points)


def compute_normalized_L(
    cam: Camera, current_points: np.ndarray, desired_points: np.ndarray
):
    L_current = compute_current_L(cam, current_points)
    L_desired = compute_desired_L(cam, desired_points)
    L_normalized = 0.5 * (L_current + L_desired)
    return L_normalized


def velocity(current_features, desired_features, L, lambda_gain=0.5):
    err_norm, error = compute_error(current_features, desired_features)
    L_pseudo_inv = np.linalg.pinv(L)
    v = -lambda_gain * L_pseudo_inv @ error
    return v.flatten(), err_norm
