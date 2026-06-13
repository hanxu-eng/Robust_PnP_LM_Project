"""Run reproducible synthetic PnP experiments and save CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from .pnp_lm import lm_pnp
    from .utils import (
        generate_synthetic_pnp_data,
        reprojection_rmse_clean,
        reprojection_rmse_observed,
        rotation_error_deg,
        translation_error,
    )
except ImportError:
    from pnp_lm import lm_pnp
    from utils import (
        generate_synthetic_pnp_data,
        reprojection_rmse_clean,
        reprojection_rmse_observed,
        rotation_error_deg,
        translation_error,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DETAIL_FIELDS = [
    "experiment_type",
    "setting_value",
    "trial",
    "method",
    "clean_reprojection_rmse",
    "observed_reprojection_rmse",
    "rotation_error_deg",
    "translation_error",
    "num_iters",
    "final_cost",
]
SUMMARY_FIELDS = [
    "experiment_type",
    "setting_value",
    "method",
    "clean_reprojection_rmse_mean",
    "clean_reprojection_rmse_std",
    "observed_reprojection_rmse_mean",
    "observed_reprojection_rmse_std",
    "rotation_error_deg_mean",
    "rotation_error_deg_std",
    "translation_error_mean",
    "translation_error_std",
    "num_iters_mean",
]


def mode_config(mode: str) -> dict:
    """Return experiment configuration for quick or full mode."""
    if mode == "quick":
        return {
            "n_trials": 3,
            "n_points": 80,
            "max_iters": 20,
            "noise_sigma_list": [0.5, 2.0],
            "outlier_ratio_list": [0.0, 0.2],
            "noise_fixed_outlier_ratio": 0.1,
            "outlier_fixed_noise_sigma": 1.0,
        }
    if mode == "full":
        return {
            "n_trials": 10,
            "n_points": 100,
            "max_iters": 30,
            "noise_sigma_list": [0.5, 1.0, 2.0, 3.0],
            "outlier_ratio_list": [0.0, 0.1, 0.2, 0.3],
            "noise_fixed_outlier_ratio": 0.1,
            "outlier_fixed_noise_sigma": 1.0,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_trial(
    experiment_type: str,
    setting_value: float,
    trial: int,
    data: dict,
    init_params: np.ndarray,
    max_iters: int,
) -> list[dict]:
    """Run ordinary and Huber LM for one data set and return metric rows."""
    rows: list[dict] = []
    method_specs = [
        ("Ordinary-LM", False),
        ("Huber-LM", True),
    ]
    for method, robust in method_specs:
        result = lm_pnp(
            data["X"],
            data["observed_uv"],
            data["K"],
            init_params=init_params,
            max_iters=max_iters,
            robust=robust,
            huber_delta=5.0,
        )
        clean_rmse = reprojection_rmse_clean(
            data["X"], data["clean_uv"], result["rvec"], result["t"], data["K"]
        )
        observed_rmse = reprojection_rmse_observed(
            data["X"], data["observed_uv"], result["rvec"], result["t"], data["K"]
        )
        rows.append(
            {
                "experiment_type": experiment_type,
                "setting_value": setting_value,
                "trial": trial,
                "method": method,
                "clean_reprojection_rmse": clean_rmse,
                "observed_reprojection_rmse": observed_rmse,
                "rotation_error_deg": rotation_error_deg(result["R"], data["R_gt"]),
                "translation_error": translation_error(result["t"], data["t_gt"]),
                "num_iters": result["num_iters"],
                "final_cost": result["cost_history"][-1],
            }
        )
    return rows


def run_experiments(mode: str) -> list[dict]:
    """Run all configured experiments."""
    cfg = mode_config(mode)
    rows: list[dict] = []
    base_seed = 20260613

    for noise_sigma in cfg["noise_sigma_list"]:
        for trial in range(cfg["n_trials"]):
            seed = base_seed + 1000 + trial + int(noise_sigma * 100)
            data = generate_synthetic_pnp_data(
                n_points=cfg["n_points"],
                noise_sigma=noise_sigma,
                outlier_ratio=cfg["noise_fixed_outlier_ratio"],
                seed=seed,
            )
            rng = np.random.default_rng(seed + 77)
            init_rvec = data["rvec_gt"] + rng.normal(0.0, 0.05, size=3)
            init_t = data["t_gt"] + rng.normal(0.0, 0.08, size=3)
            init_params = np.r_[init_rvec, init_t]
            rows.extend(
                evaluate_trial(
                    "noise",
                    float(noise_sigma),
                    trial,
                    data,
                    init_params,
                    cfg["max_iters"],
                )
            )

    for outlier_ratio in cfg["outlier_ratio_list"]:
        for trial in range(cfg["n_trials"]):
            seed = base_seed + 5000 + trial + int(outlier_ratio * 1000)
            data = generate_synthetic_pnp_data(
                n_points=cfg["n_points"],
                noise_sigma=cfg["outlier_fixed_noise_sigma"],
                outlier_ratio=outlier_ratio,
                seed=seed,
            )
            rng = np.random.default_rng(seed + 77)
            init_rvec = data["rvec_gt"] + rng.normal(0.0, 0.05, size=3)
            init_t = data["t_gt"] + rng.normal(0.0, 0.08, size=3)
            init_params = np.r_[init_rvec, init_t]
            rows.extend(
                evaluate_trial(
                    "outlier",
                    float(outlier_ratio),
                    trial,
                    data,
                    init_params,
                    cfg["max_iters"],
                )
            )
    return rows


def write_detail_csv(rows: list[dict], path: Path) -> None:
    """Write per-trial metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Aggregate per-trial rows without pandas."""
    groups: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        key = (row["experiment_type"], float(row["setting_value"]), row["method"])
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for key in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2])):
        experiment_type, setting_value, method = key
        group = groups[key]
        clean = np.array([float(r["clean_reprojection_rmse"]) for r in group])
        observed = np.array([float(r["observed_reprojection_rmse"]) for r in group])
        rot = np.array([float(r["rotation_error_deg"]) for r in group])
        trans = np.array([float(r["translation_error"]) for r in group])
        iters = np.array([float(r["num_iters"]) for r in group])
        summary_rows.append(
            {
                "experiment_type": experiment_type,
                "setting_value": setting_value,
                "method": method,
                "clean_reprojection_rmse_mean": float(np.mean(clean)),
                "clean_reprojection_rmse_std": float(np.std(clean, ddof=0)),
                "observed_reprojection_rmse_mean": float(np.mean(observed)),
                "observed_reprojection_rmse_std": float(np.std(observed, ddof=0)),
                "rotation_error_deg_mean": float(np.mean(rot)),
                "rotation_error_deg_std": float(np.std(rot, ddof=0)),
                "translation_error_mean": float(np.mean(trans)),
                "translation_error_std": float(np.std(trans, ddof=0)),
                "num_iters_mean": float(np.mean(iters)),
            }
        )
    return summary_rows


def write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write aggregated metrics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Robust PnP LM experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    rows = run_experiments(args.mode)
    detail_path = RESULTS_DIR / "experiment_results.csv"
    summary_path = RESULTS_DIR / "summary_results.csv"
    write_detail_csv(rows, detail_path)
    summary_rows = aggregate_rows(rows)
    write_summary_csv(summary_rows, summary_path)
    print(f"Mode: {args.mode}")
    print(f"Wrote {len(rows)} detailed rows to {detail_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
