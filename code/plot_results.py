"""Create PDF figures from summary experiment results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep validation possible in restricted environments.
    plt = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
METHODS = ["Ordinary-LM", "Huber-LM"]


def read_summary(path: Path) -> list[dict]:
    """Read summary CSV and convert numeric fields."""
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = dict(row)
            for key, value in row.items():
                if key not in {"experiment_type", "method"}:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def rows_for(rows: list[dict], experiment_type: str, metric_mean: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect x and y arrays for each method."""
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in METHODS:
        filtered = [
            r
            for r in rows
            if r["experiment_type"] == experiment_type and r["method"] == method
        ]
        filtered.sort(key=lambda r: r["setting_value"])
        x = np.array([r["setting_value"] for r in filtered], dtype=float)
        y = np.array([r[metric_mean] for r in filtered], dtype=float)
        output[method] = (x, y)
    return output


def _pdf_escape(text: str) -> str:
    """Escape a text string for a minimal PDF content stream."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_minimal_pdf(path: Path, commands: list[str], width: int = 612, height: int = 396) -> None:
    """Write a simple vector PDF using only the standard library."""
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(pdf))


def _fallback_line_pdf(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    yscale_log: bool = False,
) -> None:
    """Save a small vector line chart without external plotting packages."""
    width, height = 612, 396
    left, right, bottom, top = 72, 36, 70, 52
    plot_w = width - left - right
    plot_h = height - bottom - top
    colors = [(0.12, 0.35, 0.75), (0.80, 0.20, 0.18), (0.12, 0.55, 0.28)]

    clean_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, x_values, y_values in series:
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        if yscale_log:
            y_arr = np.log10(np.maximum(y_arr, 1e-300))
        clean_series.append((label, x_arr, y_arr))

    all_x = np.concatenate([item[1] for item in clean_series if item[1].size > 0])
    all_y = np.concatenate([item[2] for item in clean_series if item[2].size > 0])
    if all_x.size == 0 or all_y.size == 0:
        raise ValueError("No data available for plotting.")

    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if abs(x_max - x_min) < 1e-12:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.5
        y_max += 0.5
    if y_min > 0.0 and not yscale_log:
        y_min = 0.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return bottom + (y - y_min) / (y_max - y_min) * plot_h

    def text(x: float, y: float, value: str, size: int = 10) -> str:
        return f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(value)}) Tj ET"

    commands = [
        "1 1 1 rg 0 0 612 396 re f",
        text(180, 365, title, 14),
        text(260, 20, xlabel, 10),
        text(18, 342, ylabel + (" (log10)" if yscale_log else ""), 10),
    ]

    y_ticks = np.linspace(y_min, y_max, 5)
    commands.append("0.85 0.85 0.85 RG 0.5 w")
    for y_tick in y_ticks:
        y_pos = sy(float(y_tick))
        commands.append(f"{left:.2f} {y_pos:.2f} m {left + plot_w:.2f} {y_pos:.2f} l S")
        commands.append(text(25, y_pos - 3, f"{y_tick:.3g}", 8))

    x_ticks = np.unique(all_x)
    commands.append("0.85 0.85 0.85 RG 0.5 w")
    for x_tick in x_ticks:
        x_pos = sx(float(x_tick))
        commands.append(f"{x_pos:.2f} {bottom:.2f} m {x_pos:.2f} {bottom + plot_h:.2f} l S")
        commands.append(text(x_pos - 8, bottom - 18, f"{x_tick:g}", 8))

    commands.append("0 0 0 RG 1 w")
    commands.append(f"{left:.2f} {bottom:.2f} m {left:.2f} {bottom + plot_h:.2f} l S")
    commands.append(f"{left:.2f} {bottom:.2f} m {left + plot_w:.2f} {bottom:.2f} l S")

    for index, (label, x_arr, y_arr) in enumerate(clean_series):
        color = colors[index % len(colors)]
        r, g, b = color
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG 1.6 w")
        points = [(sx(float(x)), sy(float(y))) for x, y in zip(x_arr, y_arr)]
        if points:
            x0, y0 = points[0]
            path_commands = [f"{x0:.2f} {y0:.2f} m"]
            path_commands.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
            commands.append(" ".join(path_commands) + " S")
            commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
            for x, y in points:
                commands.append(f"{x - 2:.2f} {y - 2:.2f} 4 4 re f")
        legend_x = left + plot_w - 118
        legend_y = bottom + plot_h - 18 - 16 * index
        commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {legend_x:.2f} {legend_y:.2f} 10 4 re f")
        commands.append(text(legend_x + 16, legend_y - 2, label, 9))

    _write_minimal_pdf(out_path, commands, width=width, height=height)


def save_line_plot(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    yscale_log: bool = False,
) -> Path:
    """Save a line plot as PDF, preferring Matplotlib with a standard-library fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        _fallback_line_pdf(series, xlabel, ylabel, title, out_path, yscale_log=yscale_log)
        return out_path

    plt.figure(figsize=(6.2, 4.2))
    for label, x, y in series:
        plt.plot(x, y, marker="o", linewidth=2.0, label=label)
    if yscale_log:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_metric(
    rows: list[dict],
    experiment_type: str,
    metric_mean: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    """Plot one metric and save it as a PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / filename
    series = rows_for(rows, experiment_type, metric_mean)
    plot_series = [(method, series[method][0], series[method][1]) for method in METHODS]
    return save_line_plot(plot_series, xlabel, ylabel, title, out_path)


def main() -> None:
    """Read CSV summaries and generate report-ready PDF figures."""
    summary_path = RESULTS_DIR / "summary_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find {summary_path}. Run `python code/experiment.py --mode quick` first."
        )
    rows = read_summary(summary_path)
    figure_paths = [
        plot_metric(
            rows,
            "noise",
            "clean_reprojection_rmse_mean",
            "noise_sigma",
            "clean reprojection RMSE (pixel)",
            "Noise sensitivity",
            "noise_clean_reprojection_rmse.pdf",
        ),
        plot_metric(
            rows,
            "noise",
            "rotation_error_deg_mean",
            "noise_sigma",
            "rotation error (degree)",
            "Rotation error under noise",
            "noise_rotation_error.pdf",
        ),
        plot_metric(
            rows,
            "outlier",
            "clean_reprojection_rmse_mean",
            "outlier_ratio",
            "clean reprojection RMSE (pixel)",
            "Outlier sensitivity",
            "outlier_clean_reprojection_rmse.pdf",
        ),
        plot_metric(
            rows,
            "outlier",
            "rotation_error_deg_mean",
            "outlier_ratio",
            "rotation error (degree)",
            "Rotation error under outliers",
            "outlier_rotation_error.pdf",
        ),
        plot_metric(
            rows,
            "outlier",
            "translation_error_mean",
            "outlier_ratio",
            "translation error",
            "Translation error under outliers",
            "translation_error.pdf",
        ),
    ]
    print("Generated figures:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
