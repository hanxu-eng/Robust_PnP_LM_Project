"""Run LM-PnP on real 2D-3D correspondences from a COLMAP text model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from .pnp_lm import lm_pnp
    from .plot_results import save_line_plot
    from .real_data import build_pnp_problem_from_colmap, corrupt_real_observations_by_shuffling
    from .utils import (
        reprojection_rmse_observed,
        rotation_error_deg,
        so3_exp,
        translation_error,
    )
except ImportError:
    from pnp_lm import lm_pnp
    from plot_results import save_line_plot
    from real_data import build_pnp_problem_from_colmap, corrupt_real_observations_by_shuffling
    from utils import (
        reprojection_rmse_observed,
        rotation_error_deg,
        so3_exp,
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
                    **metrics,
                }
                rows.append(row)
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Aggregate real-data rows without pandas."""
    groups: dict[tuple[float, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((float(row["outlier_ratio"]), row["method"]), []).append(row)

    summary_rows: list[dict] = []
    for (outlier_ratio, method), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run real-data PnP LM experiments from COLMAP.")
    parser.add_argument("--colmap-dir", required=True, help="Directory containing cameras.txt/images.txt/points3D.txt.")
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--format", choices=["svg", "pdf", "both"], default="svg")
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = mode_config(args.mode)
    max_points = args.max_points if args.max_points is not None else cfg["max_points"]
    problem = build_pnp_problem_from_colmap(
        args.colmap_dir,
        image_id=args.image_id,
        image_name=args.image_name,
        min_points=args.min_points,
        max_points=max_points,
        seed=args.seed,
    )
    print(
        f"Loaded real COLMAP image {problem['image_id']} ({problem['image_name']}) "
        f"with {problem['num_correspondences']} correspondences."
    )
    print(f"Camera model: {problem['camera_model']}. {problem['camera_note']}")

    rows = run_real_experiment(problem, mode=args.mode, seed=args.seed)
    summary_rows = aggregate_rows(rows)
    detail_path = RESULTS_DIR / "real_experiment_results.csv"
    summary_path = RESULTS_DIR / "real_summary_results.csv"
    write_csv(rows, detail_path, DETAIL_FIELDS)
    write_csv(summary_rows, summary_path, SUMMARY_FIELDS)
    print(f"Wrote {len(rows)} rows to {detail_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")

    formats = ["svg", "pdf"] if args.format == "both" else [args.format]
    figure_paths: list[Path] = []
    for fig_format in formats:
        figure_paths.extend(plot_real_summary(summary_rows, fig_format))
    print("Generated real-data figures:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
