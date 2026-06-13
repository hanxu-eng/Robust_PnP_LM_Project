"""Load real 2D-3D PnP problems from COLMAP text models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .utils import project_points, so3_log
except ImportError:
    from utils import project_points, so3_log


SUPPORTED_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP qvec [qw, qx, qy, qz] to a rotation matrix."""
    qvec = np.asarray(qvec, dtype=float).reshape(4)
    qvec = qvec / max(np.linalg.norm(qvec), 1e-12)
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qw * qz,
                2.0 * qz * qx + 2.0 * qw * qy,
            ],
            [
                2.0 * qx * qy + 2.0 * qw * qz,
                1.0 - 2.0 * qz * qz - 2.0 * qx * qx,
                2.0 * qy * qz - 2.0 * qw * qx,
            ],
            [
                2.0 * qz * qx - 2.0 * qw * qy,
                2.0 * qy * qz + 2.0 * qw * qx,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=float,
    )


def camera_params_to_K(model: str, params: list[float]) -> tuple[np.ndarray, str]:
    """Build a pinhole intrinsic matrix from common COLMAP camera models."""
    model = model.upper()
    if model not in SUPPORTED_CAMERA_MODELS:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    note = "No distortion correction is applied; use undistorted COLMAP models when possible."
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        note = "PINHOLE model has no distortion parameters."
    elif model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        fx, fy, cx, cy = params[:4]

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return K, note


def read_cameras_text(path: Path) -> dict[int, dict]:
    """Read COLMAP cameras.txt."""
    cameras: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            camera_id = int(tokens[0])
            model = tokens[1]
            width = int(tokens[2])
            height = int(tokens[3])
            params = [float(v) for v in tokens[4:]]
            K, note = camera_params_to_K(model, params)
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "K": K,
                "note": note,
            }
    return cameras


def read_points3d_text(path: Path) -> dict[int, np.ndarray]:
    """Read COLMAP points3D.txt and return 3D coordinates."""
    points: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            point_id = int(tokens[0])
            points[point_id] = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    return points


def read_images_text(path: Path) -> dict[int, dict]:
    """Read COLMAP images.txt with image poses and 2D-3D observations."""
    images: dict[int, dict] = {}
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        image_tokens = lines[i].split()
        if len(image_tokens) < 10:
            raise ValueError(f"Invalid image line in images.txt: {lines[i]}")

        image_id = int(image_tokens[0])
        qvec = np.array([float(v) for v in image_tokens[1:5]], dtype=float)
        tvec = np.array([float(v) for v in image_tokens[5:8]], dtype=float)
        camera_id = int(image_tokens[8])
        name = " ".join(image_tokens[9:])

        xy = []
        point_ids = []
        if i + 1 < len(lines):
            point_tokens = lines[i + 1].split()
            for j in range(0, len(point_tokens), 3):
                if j + 2 >= len(point_tokens):
                    break
                xy.append([float(point_tokens[j]), float(point_tokens[j + 1])])
                point_ids.append(int(point_tokens[j + 2]))

        images[image_id] = {
            "image_id": image_id,
            "qvec": qvec,
            "R": qvec_to_rotmat(qvec),
            "t": tvec,
            "camera_id": camera_id,
            "name": name,
            "xy": np.asarray(xy, dtype=float),
            "point3d_ids": np.asarray(point_ids, dtype=np.int64),
        }
        i += 2
    return images


def load_colmap_model(colmap_dir: str | Path) -> tuple[dict[int, dict], dict[int, dict], dict[int, np.ndarray]]:
    """Load cameras, images and 3D points from a COLMAP text model directory."""
    colmap_dir = Path(colmap_dir)
    cameras_path = colmap_dir / "cameras.txt"
    images_path = colmap_dir / "images.txt"
    points_path = colmap_dir / "points3D.txt"
    for path in [cameras_path, images_path, points_path]:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}")
    cameras = read_cameras_text(cameras_path)
    images = read_images_text(images_path)
    points3d = read_points3d_text(points_path)
    return cameras, images, points3d


def image_observation_count(image: dict, points3d: dict[int, np.ndarray]) -> int:
    """Count valid 2D-3D correspondences in one COLMAP image."""
    return int(sum(pid >= 0 and int(pid) in points3d for pid in image["point3d_ids"]))


def select_image(
    images: dict[int, dict],
    points3d: dict[int, np.ndarray],
    image_id: int | None = None,
    image_name: str | None = None,
    min_points: int = 30,
) -> dict:
    """Select a COLMAP image by id/name, or pick the one with most observations."""
    if image_id is not None:
        if image_id not in images:
            raise KeyError(f"Image id {image_id} not found.")
        image = images[image_id]
    elif image_name is not None:
        matches = [img for img in images.values() if img["name"] == image_name]
        if not matches:
            raise KeyError(f"Image name {image_name!r} not found.")
        image = matches[0]
    else:
        image = max(images.values(), key=lambda img: image_observation_count(img, points3d))

    count = image_observation_count(image, points3d)
    if count < min_points:
        raise ValueError(
            f"Selected image has only {count} valid correspondences; need at least {min_points}."
        )
    return image


def build_pnp_problem_from_colmap(
    colmap_dir: str | Path,
    image_id: int | None = None,
    image_name: str | None = None,
    min_points: int = 30,
    max_points: int | None = None,
    seed: int = 0,
) -> dict:
    """Build a real PnP problem from one image in a COLMAP sparse text model."""
    cameras, images, points3d = load_colmap_model(colmap_dir)
    image = select_image(images, points3d, image_id=image_id, image_name=image_name, min_points=min_points)
    camera = cameras[image["camera_id"]]

    X_list = []
    uv_list = []
    for xy, point_id in zip(image["xy"], image["point3d_ids"]):
        point_id = int(point_id)
        if point_id < 0 or point_id not in points3d:
            continue
        X_list.append(points3d[point_id])
        uv_list.append(xy)

    X = np.asarray(X_list, dtype=float)
    observed_uv = np.asarray(uv_list, dtype=float)
    if max_points is not None and X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        idx.sort()
        X = X[idx]
        observed_uv = observed_uv[idx]

    R_gt = image["R"]
    t_gt = image["t"]
    rvec_gt = so3_log(R_gt)
    reference_uv = project_points(X, rvec_gt, t_gt, camera["K"])
    depth = ((R_gt @ X.T).T + t_gt.reshape(1, 3))[:, 2]
    valid_depth = depth > 1e-6
    if not np.all(valid_depth):
        X = X[valid_depth]
        observed_uv = observed_uv[valid_depth]
        reference_uv = reference_uv[valid_depth]

    return {
        "X": X,
        "K": camera["K"],
        "observed_uv": observed_uv,
        "reference_uv": reference_uv,
        "rvec_gt": rvec_gt,
        "t_gt": t_gt,
        "R_gt": R_gt,
        "image_width": camera["width"],
        "image_height": camera["height"],
        "image_id": image["image_id"],
        "image_name": image["name"],
        "camera_model": camera["model"],
        "camera_note": camera["note"],
        "num_correspondences": int(X.shape[0]),
    }


def corrupt_real_observations_by_shuffling(
    observed_uv: np.ndarray,
    outlier_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create wrong 2D-3D matches by shuffling real 2D observations."""
    observed_uv = np.asarray(observed_uv, dtype=float)
    n_points = observed_uv.shape[0]
    rng = np.random.default_rng(seed)
    corrupted = observed_uv.copy()
    inlier_mask = np.ones(n_points, dtype=bool)
    n_outliers = int(round(n_points * outlier_ratio))
    if n_outliers <= 0:
        return corrupted, inlier_mask

    outlier_idx = rng.choice(n_points, size=n_outliers, replace=False)
    source_idx = rng.choice(n_points, size=n_outliers, replace=True)
    same = source_idx == outlier_idx
    source_idx[same] = (source_idx[same] + 1) % n_points
    corrupted[outlier_idx] = observed_uv[source_idx]
    inlier_mask[outlier_idx] = False
    return corrupted, inlier_mask
