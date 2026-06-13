"""Levenberg-Marquardt optimization for PnP without ready-made solvers."""

from __future__ import annotations

import numpy as np

try:
    from .utils import project_points, so3_exp
except ImportError:  # Allows running files directly from the code directory.
    from utils import project_points, so3_exp


def residual_vector(params: np.ndarray, X: np.ndarray, uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return flattened reprojection residuals with layout [du1, dv1, ...]."""
    params = np.asarray(params, dtype=float).reshape(6)
    pred_uv = project_points(X, params[:3], params[3:], K)
    return (pred_uv - uv).reshape(-1)


def compute_numeric_jacobian(
    params: np.ndarray,
    X: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a central-difference numeric Jacobian."""
    params = np.asarray(params, dtype=float).reshape(6)
    base_size = residual_vector(params, X, uv, K).size
    J = np.zeros((base_size, 6), dtype=float)

    for j in range(6):
        step = np.zeros(6, dtype=float)
        step[j] = eps
        r_plus = residual_vector(params + step, X, uv, K)
        r_minus = residual_vector(params - step, X, uv, K)
        J[:, j] = (r_plus - r_minus) / (2.0 * eps)
    return J


def huber_weights(residuals_2d: np.ndarray, delta: float = 5.0) -> np.ndarray:
    """Return one Huber IRLS weight per image point."""
    residuals_2d = np.asarray(residuals_2d, dtype=float)
    norms = np.linalg.norm(residuals_2d, axis=1)
    weights = np.ones_like(norms)
    mask = norms > delta
    weights[mask] = delta / np.maximum(norms[mask], 1e-12)
    return weights


def expand_point_weights(point_weights: np.ndarray) -> np.ndarray:
    """Expand N point weights to 2N scalar residual weights."""
    return np.repeat(np.asarray(point_weights, dtype=float), 2)


def _weighted_cost(residuals: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted squared residual cost."""
    return float(np.sum(weights * residuals * residuals))


def _weights_for_params(
    params: np.ndarray,
    X: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
    robust: bool,
    huber_delta: float,
) -> np.ndarray:
    """Compute scalar residual weights for ordinary or Huber LM."""
    r = residual_vector(params, X, uv, K)
    if not robust:
        return np.ones_like(r)
    point_weights = huber_weights(r.reshape(-1, 2), delta=huber_delta)
    return expand_point_weights(point_weights)


def lm_pnp(
    X: np.ndarray,
    uv: np.ndarray,
    K: np.ndarray,
    init_params: np.ndarray | None = None,
    max_iters: int = 30,
    lambda_init: float = 1e-3,
    robust: bool = False,
    huber_delta: float = 5.0,
    verbose: bool = False,
) -> dict:
    """Estimate camera pose with a hand-written Levenberg-Marquardt optimizer."""
    if init_params is None:
        params = np.zeros(6, dtype=float)
    else:
        params = np.asarray(init_params, dtype=float).reshape(6).copy()

    lambda_lm = float(np.clip(lambda_init, 1e-12, 1e12))
    cost_history: list[float] = []
    lambda_history: list[float] = []
    success = False

    residuals = residual_vector(params, X, uv, K)
    weights = _weights_for_params(params, X, uv, K, robust, huber_delta)
    current_cost = _weighted_cost(residuals, weights)
    cost_history.append(current_cost)
    lambda_history.append(lambda_lm)

    for iteration in range(max_iters):
        residuals = residual_vector(params, X, uv, K)
        weights = _weights_for_params(params, X, uv, K, robust, huber_delta)
        sqrt_weights = np.sqrt(np.maximum(weights, 1e-12))
        rw = sqrt_weights * residuals

        J = compute_numeric_jacobian(params, X, uv, K)
        Jw = sqrt_weights[:, None] * J
        H = Jw.T @ Jw
        g = Jw.T @ rw
        diag_H = np.maximum(np.diag(H), 1e-12)

        accepted = False
        best_delta = np.zeros(6, dtype=float)
        for _ in range(10):
            A = H + lambda_lm * np.diag(diag_H)
            try:
                delta = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(A, g, rcond=None)[0]

            if not np.all(np.isfinite(delta)):
                lambda_lm = float(np.clip(lambda_lm * 10.0, 1e-12, 1e12))
                continue

            candidate_params = params + delta
            candidate_residuals = residual_vector(candidate_params, X, uv, K)
            candidate_weights = _weights_for_params(
                candidate_params, X, uv, K, robust, huber_delta
            )
            candidate_cost = _weighted_cost(candidate_residuals, candidate_weights)

            if candidate_cost < current_cost:
                old_cost = current_cost
                params = candidate_params
                current_cost = candidate_cost
                best_delta = delta
                lambda_lm = float(np.clip(lambda_lm * 0.5, 1e-12, 1e12))
                cost_history.append(current_cost)
                lambda_history.append(lambda_lm)
                accepted = True
                success = True
                if verbose:
                    print(
                        f"iter={iteration:02d}, cost={current_cost:.6e}, "
                        f"lambda={lambda_lm:.3e}, |delta|={np.linalg.norm(delta):.3e}"
                    )
                if np.linalg.norm(delta) < 1e-8 or abs(old_cost - current_cost) < 1e-10:
                    iteration += 1
                    return {
                        "params": params,
                        "rvec": params[:3],
                        "t": params[3:],
                        "R": so3_exp(params[:3]),
                        "cost_history": cost_history,
                        "lambda_history": lambda_history,
                        "num_iters": iteration,
                        "success": success,
                    }
                break

            lambda_lm = float(np.clip(lambda_lm * 2.0, 1e-12, 1e12))

        if not accepted:
            if verbose:
                print(f"iter={iteration:02d}, rejected, lambda={lambda_lm:.3e}")
            lambda_history.append(lambda_lm)
            if lambda_lm >= 1e12:
                break
        elif np.linalg.norm(best_delta) < 1e-8:
            break

    return {
        "params": params,
        "rvec": params[:3],
        "t": params[3:],
        "R": so3_exp(params[:3]),
        "cost_history": cost_history,
        "lambda_history": lambda_history,
        "num_iters": max(0, len(cost_history) - 1),
        "success": success,
    }
