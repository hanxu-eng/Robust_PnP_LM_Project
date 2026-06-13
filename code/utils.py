"""Utility functions for synthetic PnP experiments."""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3D vector."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Map a rotation vector to a rotation matrix using Rodrigues' formula."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-12:
        return np.eye(3) + W + 0.5 * (W @ W)

    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * W + b * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Map a rotation matrix to a rotation vector."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    theta = float(np.arccos(cos_angle))

    if theta < 1e-12:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ],
            dtype=float,
        )

    if np.pi - theta < 1e-6:
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] = -axis[2]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = axis / axis_norm
        return theta * axis

    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    )


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Compute geodesic rotation error in degrees."""
    R_est = np.asarray(R_est, dtype=float).reshape(3, 3)
    R_gt = np.asarray(R_gt, dtype=float).reshape(3, 3)
    cos_angle = (np.trace(R_gt.T @ R_est) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def project_points(
    X: np.ndarray,
    rvec: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project 3D points to image coordinates."""
    X = np.asarray(X, dtype=float)
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    t = np.asarray(t, dtype=float).reshape(3)
    K = np.asarray(K, dtype=float).reshape(3, 3)

    R = so3_exp(rvec)
    X_cam = (R @ X.T).T + t.reshape(1, 3)
    z = X_cam[:, 2]
    z_safe = np.where(np.abs(z) < 1e-9, np.where(z >= 0.0, 1e-9, -1e-9), z)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * X_cam[:, 0] / z_safe + cx
    v = fy * X_cam[:, 1] / z_safe + cy
    return np.column_stack([u, v])


def generate_synthetic_pnp_data(
    n_points: int = 100,
    noise_sigma: float = 1.0,
    outlier_ratio: float = 0.1,
    image_width: int = 640,
    image_height: int = 480,
    seed: int = 0,
) -> dict:
    """Generate reproducible synthetic PnP data with Gaussian noise and outliers."""
    rng = np.random.default_rng(seed)

    K = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    X = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, size=n_points),
            rng.uniform(-2.0, 2.0, size=n_points),
            rng.uniform(4.0, 8.0, size=n_points),
        ]
    )

    rvec_gt = np.array([0.15, -0.10, 0.08], dtype=float)
    t_gt = np.array([0.30, -0.20, 0.50], dtype=float)
    R_gt = so3_exp(rvec_gt)

    clean_uv = project_points(X, rvec_gt, t_gt, K)
    observed_uv = clean_uv + rng.normal(0.0, noise_sigma, size=clean_uv.shape)
    inlier_mask = np.ones(n_points, dtype=bool)

    n_outliers = int(round(n_points * outlier_ratio))
    if n_outliers > 0:
        outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
        random_uv = np.column_stack(
            [
                rng.uniform(0.0, float(image_width), size=n_outliers),
                rng.uniform(0.0, float(image_height), size=n_outliers),
            ]
        )
        large_shift = rng.normal(0.0, 120.0, size=(n_outliers, 2))
        observed_uv[outlier_idx] = 0.5 * random_uv + 0.5 * (clean_uv[outlier_idx] + large_shift)
        inlier_mask[outlier_idx] = False

    return {
        "X": X,
        "K": K,
        "clean_uv": clean_uv,
        "observed_uv": observed_uv,
        "rvec_gt": rvec_gt,
        "t_gt": t_gt,
        "R_gt": R_gt,
        "inlier_mask": inlier_mask,
    }


def reprojection_rmse_clean(
    X: np.ndarray,
    clean_uv: np.ndarray,
    rvec_est: np.ndarray,
    t_est: np.ndarray,
    K: np.ndarray,
) -> float:
    """Compute RMSE against clean projected points."""
    pred_uv = project_points(X, rvec_est, t_est, K)
    errors = pred_uv - clean_uv
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def reprojection_rmse_observed(
    X: np.ndarray,
    observed_uv: np.ndarray,
    rvec_est: np.ndarray,
    t_est: np.ndarray,
    K: np.ndarray,
) -> float:
    """Compute RMSE against noisy/outlier-contaminated observations."""
    pred_uv = project_points(X, rvec_est, t_est, K)
    errors = pred_uv - observed_uv
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Compute Euclidean translation error."""
    return float(np.linalg.norm(np.asarray(t_est, dtype=float) - np.asarray(t_gt, dtype=float)))
