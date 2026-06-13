"""Minimal runnable example for ordinary and robust LM-PnP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .pnp_lm import lm_pnp
    from .plot_results import save_line_plot
    from .utils import (
        generate_synthetic_pnp_data,
        reprojection_rmse_clean,
        reprojection_rmse_observed,
        rotation_error_deg,
        translation_error,
    )
except ImportError:
    from pnp_lm import lm_pnp
    from plot_results import save_line_plot
    from utils import (
        generate_synthetic_pnp_data,
        reprojection_rmse_clean,
        reprojection_rmse_observed,
        rotation_error_deg,
        translation_error,
    )


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"


def evaluate_result(method: str, result: dict, data: dict) -> dict:
    """Compute and print metrics for one optimization result."""
    clean_rmse = reprojection_rmse_clean(
        data["X"], data["clean_uv"], result["rvec"], result["t"], data["K"]
    )
    observed_rmse = reprojection_rmse_observed(
        data["X"], data["observed_uv"], result["rvec"], result["t"], data["K"]
    )
    rot_err = rotation_error_deg(result["R"], data["R_gt"])
    trans_err = translation_error(result["t"], data["t_gt"])
    metrics = {
        "method": method,
        "clean_rmse": clean_rmse,
        "observed_rmse": observed_rmse,
        "rotation_error_deg": rot_err,
        "translation_error": trans_err,
        "iterations": result["num_iters"],
    }
    print(
        f"{method:12s} | clean RMSE: {clean_rmse:8.4f} | "
        f"observed RMSE: {observed_rmse:8.4f} | "
        f"rot err: {rot_err:8.4f} deg | trans err: {trans_err:8.4f} | "
        f"iters: {result['num_iters']:2d}"
    )
    return metrics


def plot_convergence(ordinary: dict, huber: dict) -> Path:
    """Save convergence curves as a PDF vector figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "convergence.pdf"
    series = [
        (
            "Ordinary-LM",
            np.arange(len(ordinary["cost_history"])),
            np.asarray(ordinary["cost_history"], dtype=float),
        ),
        (
            "Huber-LM",
            np.arange(len(huber["cost_history"])),
            np.asarray(huber["cost_history"], dtype=float),
        ),
    ]
    return save_line_plot(
        series,
        "Accepted iteration",
        "Weighted cost",
        "LM-PnP convergence",
        out_path,
        yscale_log=True,
    )


def main() -> None:
    """Run one reproducible PnP example."""
    data = generate_synthetic_pnp_data(
        n_points=100,
        noise_sigma=1.0,
        outlier_ratio=0.1,
        seed=42,
    )
    init_rvec = data["rvec_gt"] + np.array([0.08, -0.06, 0.05])
    init_t = data["t_gt"] + np.array([0.15, -0.10, 0.12])
    init_params = np.r_[init_rvec, init_t]

    ordinary = lm_pnp(
        data["X"],
        data["observed_uv"],
        data["K"],
        init_params=init_params,
        max_iters=30,
        robust=False,
    )
    huber = lm_pnp(
        data["X"],
        data["observed_uv"],
        data["K"],
        init_params=init_params,
        max_iters=30,
        robust=True,
        huber_delta=5.0,
    )

    print("Method metrics")
    print("-" * 96)
    evaluate_result("Ordinary-LM", ordinary, data)
    evaluate_result("Huber-LM", huber, data)
    figure_path = plot_convergence(ordinary, huber)
    print(f"Saved convergence figure: {figure_path}")


if __name__ == "__main__":
    main()
