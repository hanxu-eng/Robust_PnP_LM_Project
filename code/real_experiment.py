"""Run LM-PnP on real 2D-3D correspondences from a COLMAP text model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from .pnp_lm import lm_pnp
    from .plot_results import save_line_plot
    from .real_data import build_multiple_pnp_problems_from_colmap, corrupt_real_observations_by_shuffling
    from .utils import (
        project_points,
        reprojection_rmse_observed,
        rotation_error_deg,
        translation_error,
    )
except ImportError:
    from pnp_lm import lm_pnp
    from plot_results import save_line_plot
    from real_data import build_multiple_pnp_problems_from_colmap, corrupt_real_observations_by_shuffling
    from utils import (
        project_points,
        reprojection_rmse_observed,
        rotation_error_deg,
        translation_error,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

DETAIL_FIELDS = [
    "dataset_type",
    "image_id",
    "image_name",
    "camera_model",
    "num_correspondences",
    "outlier_ratio",
    "trial",
    "method",
    "success",
    "reference_reprojection_rmse",
    "observed_reprojection_rmse",
    "rotation_error_deg",
    "translation_error",
    "num_iters",
    "final_cost",
]

SUMMARY_FIELDS = [
    "dataset_type",
    "image_id",
    "image_name",
    "camera_model",
    "num_correspondences",
    "outlier_ratio",
    "method",
    "success_rate",
    "reference_reprojection_rmse_mean",
    "reference_reprojection_rmse_std",
    "observed_reprojection_rmse_mean",
    "observed_reprojection_rmse_std",
    "rotation_error_deg_mean",
    "rotation_error_deg_std",
    "translation_error_mean",
    "translation_error_std",
    "num_iters_mean",
    "final_cost_mean",
]


def mode_config(mode: str) -> dict:
    """Return real-data experiment settings."""
    if mode == "quick":
        return {
            "n_trials": 3,
            "outlier_ratio_list": [0.0, 0.1, 0.2],
            "max_iters": 30,
            "max_points": 200,
        }
    if mode == "full":
        return {
            "n_trials": 10,
            "outlier_ratio_list": [0.0, 0.1, 0.2, 0.3],
            "max_iters": 40,
            "max_points": 500,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def make_initial_params(problem: dict, seed: int) -> np.ndarray:
    """Create a fixed-size perturbation around the reference COLMAP pose."""
    rng = np.random.default_rng(seed)
    rot_step = rng.normal(0.0, np.radians(2.0), size=3)
    trans_scale = max(0.02, 0.03 * np.linalg.norm(problem["t_gt"]))
    trans_step = rng.normal(0.0, trans_scale, size=3)
    init_rvec = problem["rvec_gt"] + rot_step
    init_t = problem["t_gt"] + trans_step
    return np.r_[init_rvec, init_t]


def reference_reprojection_rmse(problem: dict, rvec: np.ndarray, t: np.ndarray) -> float:
    """Evaluate pose against the COLMAP reference pose projection."""
    return reprojection_rmse_observed(problem["X"], problem["reference_uv"], rvec, t, problem["K"])


def run_one_method(problem: dict, observed_uv: np.ndarray, init_params: np.ndarray, max_iters: int, method: str) -> dict:
    """Run one LM-PnP variant and compute real-data metrics."""
    robust = method == "Huber-LM"
    result = lm_pnp(
        problem["X"],
        observed_uv,
        problem["K"],
        init_params=init_params,
        max_iters=max_iters,
        robust=robust,
        huber_delta=5.0,
    )
    return {
        "result": result,
        "success": int(bool(result["success"])),
        "reference_reprojection_rmse": reference_reprojection_rmse(problem, result["rvec"], result["t"]),
        "observed_reprojection_rmse": reprojection_rmse_observed(
            problem["X"], observed_uv, result["rvec"], result["t"], problem["K"]
        ),
        "rotation_error_deg": rotation_error_deg(result["R"], problem["R_gt"]),
        "translation_error": translation_error(result["t"], problem["t_gt"]),
        "num_iters": result["num_iters"],
        "final_cost": result["cost_history"][-1],
    }


def run_real_experiment(problem: dict, mode: str, seed: int) -> list[dict]:
    """Run real-data baseline and wrong-match stress tests."""
    cfg = mode_config(mode)
    rows: list[dict] = []
    for outlier_ratio in cfg["outlier_ratio_list"]:
        for trial in range(cfg["n_trials"]):
            trial_seed = seed + trial + int(outlier_ratio * 1000)
            observed_uv, _ = corrupt_real_observations_by_shuffling(
                problem["observed_uv"], outlier_ratio=outlier_ratio, seed=trial_seed
            )
            init_params = make_initial_params(problem, seed=trial_seed + 77)
            for method in ["Ordinary-LM", "Huber-LM"]:
                metrics = run_one_method(problem, observed_uv, init_params, cfg["max_iters"], method)
                row = {
                    "dataset_type": "COLMAP-real",
                    "image_id": problem["image_id"],
                    "image_name": problem["image_name"],
                    "camera_model": problem["camera_model"],
                    "num_correspondences": problem["num_correspondences"],
                    "outlier_ratio": outlier_ratio,
                    "trial": trial,
                    "method": method,
                    **{key: value for key, value in metrics.items() if key != "result"},
                }
                rows.append(row)
    return rows


def run_real_experiments(problems: list[dict], mode: str, seed: int) -> list[dict]:
    """Run real-data experiments for multiple selected COLMAP images."""
    rows: list[dict] = []
    for index, problem in enumerate(problems):
        rows.extend(run_real_experiment(problem, mode=mode, seed=seed + index * 10000))
    return rows


def aggregate_rows(rows: list[dict], by_image: bool = False) -> list[dict]:
    """Aggregate real-data rows without pandas."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if by_image:
            key = (int(row["image_id"]), float(row["outlier_ratio"]), row["method"])
        else:
            key = (float(row["outlier_ratio"]), row["method"])
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        if by_image:
            _, outlier_ratio, method = key
        else:
            outlier_ratio, method = key
        ref = np.array([float(r["reference_reprojection_rmse"]) for r in group])
        obs = np.array([float(r["observed_reprojection_rmse"]) for r in group])
        rot = np.array([float(r["rotation_error_deg"]) for r in group])
        trans = np.array([float(r["translation_error"]) for r in group])
        iters = np.array([float(r["num_iters"]) for r in group])
        cost = np.array([float(r["final_cost"]) for r in group])
        success = np.array([float(r["success"]) for r in group])
        first = group[0]
        summary_rows.append(
            {
                "dataset_type": first["dataset_type"],
                "image_id": first["image_id"],
                "image_name": first["image_name"],
                "camera_model": first["camera_model"],
                "num_correspondences": first["num_correspondences"],
                "outlier_ratio": outlier_ratio,
                "method": method,
                "success_rate": float(np.mean(success)),
                "reference_reprojection_rmse_mean": float(np.mean(ref)),
                "reference_reprojection_rmse_std": float(np.std(ref, ddof=0)),
                "observed_reprojection_rmse_mean": float(np.mean(obs)),
                "observed_reprojection_rmse_std": float(np.std(obs, ddof=0)),
                "rotation_error_deg_mean": float(np.mean(rot)),
                "rotation_error_deg_std": float(np.std(rot, ddof=0)),
                "translation_error_mean": float(np.mean(trans)),
                "translation_error_std": float(np.std(trans, ddof=0)),
                "num_iters_mean": float(np.mean(iters)),
                "final_cost_mean": float(np.mean(cost)),
            }
        )
    return summary_rows


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_real_summary(summary_rows: list[dict], fig_format: str) -> list[Path]:
    """Generate real-data SVG/PDF figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["Ordinary-LM", "Huber-LM"]
    paths: list[Path] = []

    for metric, ylabel, filename, title in [
        (
            "reference_reprojection_rmse_mean",
            "Reference reprojection RMSE (px)",
            "real_reference_rmse",
            "Real COLMAP correspondences: pose accuracy",
        ),
        (
            "rotation_error_deg_mean",
            "Rotation error (deg)",
            "real_rotation_error",
            "Real COLMAP correspondences: rotation error",
        ),
        (
            "observed_reprojection_rmse_mean",
            "Observed reprojection RMSE (px)",
            "real_observed_rmse",
            "Real COLMAP correspondences: observed fit",
        ),
        (
            "translation_error_mean",
            "Translation error",
            "real_translation_error",
            "Real COLMAP correspondences: translation error",
        ),
    ]:
        series = []
        for method in methods:
            rows = [r for r in summary_rows if r["method"] == method]
            rows.sort(key=lambda r: float(r["outlier_ratio"]))
            series.append(
                (
                    method,
                    np.array([float(r["outlier_ratio"]) for r in rows]),
                    np.array([float(r[metric]) for r in rows]),
                )
            )
        path = FIGURE_DIR / f"{filename}.{fig_format}"
        paths.append(
            save_line_plot(
                series,
                "Wrong-match ratio on real correspondences",
                ylabel,
                title,
                path,
            )
        )
    return paths


def plot_per_image_summary(per_image_rows: list[dict], fig_format: str) -> list[Path]:
    """Plot per-image performance at a representative wrong-match ratio."""
    if plt is None:
        return []
    target_ratio = 0.2
    available = sorted({float(r["outlier_ratio"]) for r in per_image_rows})
    if available:
        target_ratio = min(available, key=lambda value: abs(value - target_ratio))
    rows = [r for r in per_image_rows if float(r["outlier_ratio"]) == target_ratio]
    image_ids = sorted({int(r["image_id"]) for r in rows})
    methods = ["Ordinary-LM", "Huber-LM"]
    colors = {"Ordinary-LM": "#546A7B", "Huber-LM": "#D45113"}

    paths: list[Path] = []
    for metric, ylabel, filename, title in [
        (
            "reference_reprojection_rmse_mean",
            "Reference RMSE (px)",
            "real_per_image_reference_rmse",
            "Per-image real-data pose accuracy",
        ),
        (
            "rotation_error_deg_mean",
            "Rotation error (deg)",
            "real_per_image_rotation_error",
            "Per-image real-data rotation error",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        x = np.arange(len(image_ids))
        width = 0.34
        for offset, method in [(-width / 2, "Ordinary-LM"), (width / 2, "Huber-LM")]:
            values = []
            labels = []
            for image_id in image_ids:
                match = [
                    r
                    for r in rows
                    if int(r["image_id"]) == image_id and r["method"] == method
                ]
                values.append(float(match[0][metric]) if match else np.nan)
                labels.append(match[0]["image_name"] if match else str(image_id))
            ax.bar(x + offset, values, width=width, color=colors[method], alpha=0.82, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels([str(image_id) for image_id in image_ids])
        ax.set_xlabel("COLMAP image id")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}, wrong-match ratio={target_ratio:g}", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = FIGURE_DIR / f"{filename}.{fig_format}"
        paths.append(_save_figure(fig, path, fig_format))
    return paths


def find_image_path(image_dir: str | Path | None, image_name: str) -> Path | None:
    """Find the real source image by name."""
    if image_dir is None:
        return None
    root = Path(image_dir)
    direct = root / image_name
    if direct.exists():
        return direct
    matches = list(root.rglob(image_name))
    if matches:
        return matches[0]
    return None


def _limit_points_for_plot(uv: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Return stable point indices for readable image overlays."""
    n_points = uv.shape[0]
    if n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_points, size=max_points, replace=False)
    idx.sort()
    return idx


def _read_image(image_path: Path) -> np.ndarray:
    """Read an image for Matplotlib overlays."""
    if plt is None:
        raise RuntimeError("Matplotlib is required for image overlays.")
    return plt.imread(str(image_path))


def _save_figure(fig, path: Path, fig_format: str) -> Path:
    """Save and close a Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=fig_format, bbox_inches="tight")
    plt.close(fig)
    return path


def image_safe_stem(problem: dict) -> str:
    """Return a short file-safe image label."""
    stem = Path(problem["image_name"]).stem
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"img{int(problem['image_id']):03d}_{safe}"


def fit_demo_results(problem: dict, outlier_ratio: float, seed: int, max_iters: int) -> dict:
    """Run one representative real-data stress trial for visual overlays."""
    observed_uv, inlier_mask = corrupt_real_observations_by_shuffling(
        problem["observed_uv"],
        outlier_ratio=outlier_ratio,
        seed=seed,
    )
    init_params = make_initial_params(problem, seed=seed + 77)
    ordinary = run_one_method(problem, observed_uv, init_params, max_iters, "Ordinary-LM")["result"]
    huber = run_one_method(problem, observed_uv, init_params, max_iters, "Huber-LM")["result"]
    return {
        "observed_uv": observed_uv,
        "inlier_mask": inlier_mask,
        "ordinary": ordinary,
        "huber": huber,
    }


def plot_real_dashboard(summary_rows: list[dict], fig_format: str) -> Path | None:
    """Create a compact dashboard for the real-data experiment."""
    if plt is None:
        return None
    path = FIGURE_DIR / f"real_pose_dashboard.{fig_format}"
    metrics = [
        ("reference_reprojection_rmse_mean", "Reference RMSE (px)", "A. Reference projection"),
        ("observed_reprojection_rmse_mean", "Observed RMSE (px)", "B. Observed fit"),
        ("rotation_error_deg_mean", "Rotation error (deg)", "C. Rotation error"),
        ("translation_error_mean", "Translation error", "D. Translation error"),
    ]
    methods = ["Ordinary-LM", "Huber-LM"]
    colors = {"Ordinary-LM": "#546A7B", "Huber-LM": "#D45113"}
    markers = {"Ordinary-LM": "o", "Huber-LM": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for ax, (metric, ylabel, title) in zip(axes.ravel(), metrics):
        for method in methods:
            rows = [r for r in summary_rows if r["method"] == method]
            rows.sort(key=lambda r: float(r["outlier_ratio"]))
            x = np.array([float(r["outlier_ratio"]) for r in rows])
            y = np.array([float(r[metric]) for r in rows])
            ax.plot(x, y, marker=markers[method], color=colors[method], linewidth=2.0, label=method)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Wrong-match ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Real ETH3D/COLMAP PnP experiment summary", x=0.02, y=1.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    return _save_figure(fig, path, fig_format)


def plot_keypoint_overlay(problem: dict, image_path: Path, fig_format: str, seed: int) -> Path | None:
    """Plot real observed 2D points and COLMAP reference projections on the image."""
    if plt is None:
        return None
    image = _read_image(image_path)
    idx = _limit_points_for_plot(problem["observed_uv"], max_points=220, seed=seed)
    path = FIGURE_DIR / f"real_keypoints_overlay_{image_safe_stem(problem)}.{fig_format}"
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.imshow(image)
    ax.scatter(
        problem["observed_uv"][idx, 0],
        problem["observed_uv"][idx, 1],
        s=18,
        facecolors="none",
        edgecolors="#2A9D8F",
        linewidths=0.9,
        label="COLMAP 2D observations",
    )
    ax.scatter(
        problem["reference_uv"][idx, 0],
        problem["reference_uv"][idx, 1],
        s=9,
        c="#D45113",
        alpha=0.85,
        label="Reference pose projections",
    )
    ax.set_title(f"Real image correspondences: {problem['image_name']}", loc="left", fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True)
    return _save_figure(fig, path, fig_format)


def plot_reprojection_overlay(
    problem: dict,
    image_path: Path,
    demo: dict,
    fig_format: str,
    outlier_ratio: float,
    seed: int,
) -> Path | None:
    """Plot corrupted observations and optimized Huber projections on the image."""
    if plt is None:
        return None
    image = _read_image(image_path)
    observed_uv = demo["observed_uv"]
    inlier_mask = demo["inlier_mask"]
    huber_proj = project_points(problem["X"], demo["huber"]["rvec"], demo["huber"]["t"], problem["K"])
    idx = _limit_points_for_plot(observed_uv, max_points=180, seed=seed)
    path = FIGURE_DIR / f"real_reprojection_overlay_{image_safe_stem(problem)}.{fig_format}"

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.imshow(image)
    inlier_idx = idx[inlier_mask[idx]]
    outlier_idx = idx[~inlier_mask[idx]]
    ax.scatter(observed_uv[inlier_idx, 0], observed_uv[inlier_idx, 1], s=16, c="#2A9D8F", alpha=0.75, label="Kept matches")
    if outlier_idx.size > 0:
        ax.scatter(
            observed_uv[outlier_idx, 0],
            observed_uv[outlier_idx, 1],
            s=24,
            marker="x",
            c="#C1121F",
            linewidths=1.0,
            label="Simulated wrong matches",
        )
    ax.scatter(
        huber_proj[idx, 0],
        huber_proj[idx, 1],
        s=10,
        facecolors="none",
        edgecolors="#F77F00",
        linewidths=0.8,
        label="Huber-LM projections",
    )
    ax.set_title(f"Real image robust PnP overlay, wrong-match ratio={outlier_ratio:g}", loc="left", fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True)
    return _save_figure(fig, path, fig_format)


def plot_residual_vectors(
    problem: dict,
    image_path: Path,
    demo: dict,
    fig_format: str,
    outlier_ratio: float,
    seed: int,
) -> Path | None:
    """Plot reprojection residual vectors for Huber-LM on the real image."""
    if plt is None:
        return None
    image = _read_image(image_path)
    observed_uv = demo["observed_uv"]
    huber_proj = project_points(problem["X"], demo["huber"]["rvec"], demo["huber"]["t"], problem["K"])
    idx = _limit_points_for_plot(observed_uv, max_points=100, seed=seed)
    delta = huber_proj[idx] - observed_uv[idx]
    path = FIGURE_DIR / f"real_residual_vectors_{image_safe_stem(problem)}.{fig_format}"

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.imshow(image)
    ax.quiver(
        observed_uv[idx, 0],
        observed_uv[idx, 1],
        delta[:, 0],
        delta[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0022,
        color="#D45113",
        alpha=0.82,
    )
    ax.scatter(observed_uv[idx, 0], observed_uv[idx, 1], s=10, c="#264653", alpha=0.8)
    ax.set_title(f"Huber-LM reprojection residual vectors, wrong-match ratio={outlier_ratio:g}", loc="left", fontweight="bold")
    ax.set_axis_off()
    return _save_figure(fig, path, fig_format)


def plot_residual_histogram(problem: dict, demo: dict, fig_format: str, outlier_ratio: float) -> Path | None:
    """Plot residual norm distributions for Ordinary-LM and Huber-LM."""
    if plt is None:
        return None
    observed_uv = demo["observed_uv"]
    ordinary_proj = project_points(problem["X"], demo["ordinary"]["rvec"], demo["ordinary"]["t"], problem["K"])
    huber_proj = project_points(problem["X"], demo["huber"]["rvec"], demo["huber"]["t"], problem["K"])
    ordinary_res = np.linalg.norm(ordinary_proj - observed_uv, axis=1)
    huber_res = np.linalg.norm(huber_proj - observed_uv, axis=1)
    path = FIGURE_DIR / f"real_residual_histogram_{image_safe_stem(problem)}.{fig_format}"

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    upper = float(np.percentile(np.r_[ordinary_res, huber_res], 95))
    if upper <= 1e-12:
        upper = 1.0
    bins = np.linspace(0.0, upper, 28)
    ax.hist(ordinary_res, bins=bins, alpha=0.55, color="#546A7B", label="Ordinary-LM")
    ax.hist(huber_res, bins=bins, alpha=0.55, color="#D45113", label="Huber-LM")
    ax.set_title(f"Residual distribution on real correspondences, wrong-match ratio={outlier_ratio:g}", loc="left", fontweight="bold")
    ax.set_xlabel("Reprojection residual norm (px)")
    ax.set_ylabel("Point count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, path, fig_format)


def plot_real_visuals(
    problem: dict,
    summary_rows: list[dict],
    image_dir: str | Path | None,
    fig_format: str,
    seed: int,
    max_iters: int,
) -> list[Path]:
    """Generate richer real-data visualizations."""
    paths: list[Path] = []
    dashboard = plot_real_dashboard(summary_rows, fig_format)
    if dashboard is not None:
        paths.append(dashboard)

    image_path = find_image_path(image_dir, problem["image_name"])
    if image_path is None:
        return paths

    overlay = plot_keypoint_overlay(problem, image_path, fig_format, seed)
    if overlay is not None:
        paths.append(overlay)

    outlier_ratio = 0.2
    demo = fit_demo_results(problem, outlier_ratio=outlier_ratio, seed=seed + 3000, max_iters=max_iters)
    for path in [
        plot_reprojection_overlay(problem, image_path, demo, fig_format, outlier_ratio, seed),
        plot_residual_vectors(problem, image_path, demo, fig_format, outlier_ratio, seed),
        plot_residual_histogram(problem, demo, fig_format, outlier_ratio),
    ]:
        if path is not None:
            paths.append(path)
    return paths


def plot_multi_image_montage(
    problems: list[dict],
    image_dir: str | Path | None,
    fig_format: str,
    seed: int,
) -> Path | None:
    """Create a montage showing selected real images with observed 2D points."""
    if plt is None or image_dir is None or not problems:
        return None
    items = []
    for problem in problems[:6]:
        image_path = find_image_path(image_dir, problem["image_name"])
        if image_path is None:
            continue
        items.append((problem, image_path))
    if not items:
        return None

    n_items = len(items)
    n_cols = min(3, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.2 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, (problem, image_path) in zip(axes_arr, items):
        image = _read_image(image_path)
        idx = _limit_points_for_plot(problem["observed_uv"], max_points=90, seed=seed + int(problem["image_id"]))
        ax.imshow(image)
        ax.scatter(
            problem["observed_uv"][idx, 0],
            problem["observed_uv"][idx, 1],
            s=8,
            facecolors="none",
            edgecolors="#2A9D8F",
            linewidths=0.65,
        )
        ax.set_title(
            f"Image {problem['image_id']}: {problem['num_correspondences']} matches",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_axis_off()
    for ax in axes_arr[len(items):]:
        ax.set_axis_off()
    fig.suptitle("Selected real COLMAP images and 2D-3D observations", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    path = FIGURE_DIR / f"real_multi_image_montage.{fig_format}"
    return _save_figure(fig, path, fig_format)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run real-data PnP LM experiments from COLMAP.")
    parser.add_argument("--colmap-dir", required=True, help="Directory containing cameras.txt/images.txt/points3D.txt.")
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--num-images", type=int, default=3, help="Auto-select this many top images when image id/name is not specified.")
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--format", choices=["svg", "pdf", "both"], default="svg")
    parser.add_argument("--image-dir", default=None, help="Optional directory with the source images for overlay figures.")
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = mode_config(args.mode)
    max_points = args.max_points if args.max_points is not None else cfg["max_points"]
    problems = build_multiple_pnp_problems_from_colmap(
        args.colmap_dir,
        num_images=args.num_images,
        image_id=args.image_id,
        image_name=args.image_name,
        min_points=args.min_points,
        max_points=max_points,
        seed=args.seed,
    )
    print(f"Loaded {len(problems)} real COLMAP image problem(s).")
    for problem in problems:
        print(
            f"  image {problem['image_id']} ({problem['image_name']}): "
            f"{problem['num_correspondences']} correspondences, camera={problem['camera_model']}"
        )
    print(f"Camera note: {problems[0]['camera_note']}")

    rows = run_real_experiments(problems, mode=args.mode, seed=args.seed)
    summary_rows = aggregate_rows(rows)
    per_image_summary_rows = aggregate_rows(rows, by_image=True)
    detail_path = RESULTS_DIR / "real_experiment_results.csv"
    summary_path = RESULTS_DIR / "real_summary_results.csv"
    per_image_summary_path = RESULTS_DIR / "real_per_image_summary_results.csv"
    write_csv(rows, detail_path, DETAIL_FIELDS)
    write_csv(summary_rows, summary_path, SUMMARY_FIELDS)
    write_csv(per_image_summary_rows, per_image_summary_path, SUMMARY_FIELDS)
    print(f"Wrote {len(rows)} rows to {detail_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")
    print(f"Wrote {len(per_image_summary_rows)} per-image summary rows to {per_image_summary_path}")

    formats = ["svg", "pdf"] if args.format == "both" else [args.format]
    figure_paths: list[Path] = []
    for fig_format in formats:
        figure_paths.extend(plot_real_summary(summary_rows, fig_format))
        figure_paths.extend(plot_per_image_summary(per_image_summary_rows, fig_format))
        montage = plot_multi_image_montage(problems, args.image_dir, fig_format, args.seed)
        if montage is not None:
            figure_paths.append(montage)
        for problem in problems:
            figure_paths.extend(
                plot_real_visuals(
                    problem,
                    per_image_summary_rows,
                    image_dir=args.image_dir,
                    fig_format=fig_format,
                    seed=args.seed + int(problem["image_id"]),
                    max_iters=cfg["max_iters"],
                )
            )
    print("Generated real-data figures:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
