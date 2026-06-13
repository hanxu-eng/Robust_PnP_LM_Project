"""Create report-ready vector figures from experiment CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep validation possible in restricted environments.
    plt = None
    ScalarMappable = None
    Normalize = None
    Rectangle = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
METHOD_ORDER = ["Ordinary-LM", "Huber-LM"]
METHOD_COLORS = {
    "Ordinary-LM": "#546A7B",
    "Huber-LM": "#D45113",
}
METHOD_MARKERS = {
    "Ordinary-LM": "o",
    "Huber-LM": "s",
}
EXPERIMENT_LABELS = {
    "noise": "Noise sigma",
    "outlier": "Outlier ratio",
    "initialization": "Initial rotation error (deg)",
    "point_count": "Number of 3D-2D matches",
    "huber_delta": "Huber delta",
}


def read_csv_numeric(path: Path) -> list[dict]:
    """Read a CSV file and convert numeric fields where possible."""
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = dict(row)
            for key, value in row.items():
                if key in {"experiment_type", "method"}:
                    continue
                if value == "":
                    converted[key] = float("nan")
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def read_summary(path: Path) -> list[dict]:
    """Read summary CSV results."""
    return read_csv_numeric(path)


def read_detail(path: Path) -> list[dict]:
    """Read per-trial CSV results."""
    return read_csv_numeric(path)


def methods_in(rows: list[dict], experiment_type: str) -> list[str]:
    """Return methods present in a stable order."""
    present = {r["method"] for r in rows if r["experiment_type"] == experiment_type}
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def rows_for(
    rows: list[dict],
    experiment_type: str,
    metric_mean: str,
    metric_std: str | None = None,
    method_list: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collect x, mean and std arrays for each method."""
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    methods = method_list if method_list is not None else methods_in(rows, experiment_type)
    for method in methods:
        filtered = [
            r
            for r in rows
            if r["experiment_type"] == experiment_type and r["method"] == method
        ]
        filtered.sort(key=lambda r: r["setting_value"])
        x = np.array([r["setting_value"] for r in filtered], dtype=float)
        y = np.array([r[metric_mean] for r in filtered], dtype=float)
        if metric_std is None:
            y_std = np.zeros_like(y)
        else:
            y_std = np.array([r[metric_std] for r in filtered], dtype=float)
        output[method] = (x, y, y_std)
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


def _svg_escape(text: str) -> str:
    """Escape a text string for SVG."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_minimal_svg(path: Path, body: list[str], width: int = 612, height: int = 396) -> None:
    """Write a simple SVG using only the standard library."""
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def _fallback_report_figure(path: Path, title: str, lines: list[str]) -> Path:
    """Save a text-only vector figure when Matplotlib is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".svg":
        body = [
            f'<text x="56" y="50" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="700">{_svg_escape(title)}</text>',
            '<text x="56" y="76" font-family="Helvetica, Arial, sans-serif" font-size="10">Matplotlib is not installed; this is a validation fallback.</text>',
        ]
        y = 108
        for line in lines[:14]:
            body.append(
                f'<text x="56" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="10">{_svg_escape(line)}</text>'
            )
            y += 20
        _write_minimal_svg(path, body)
        return path

    def text(x: float, y: float, value: str, size: int = 10) -> str:
        return f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(value)}) Tj ET"

    commands = [
        "1 1 1 rg 0 0 612 396 re f",
        text(56, 350, title, 15),
        text(56, 326, "Matplotlib is not installed; this is a validation fallback.", 9),
    ]
    y = 300
    for line in lines[:14]:
        commands.append(text(56, y, line, 9))
        y -= 18
    _write_minimal_pdf(path, commands)
    return path


def _fallback_line_figure(
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

    if out_path.suffix.lower() == ".svg":
        body = [
            f'<text x="180" y="30" font-family="Helvetica, Arial, sans-serif" font-size="15" font-weight="700">{_svg_escape(title)}</text>',
            f'<text x="260" y="374" font-family="Helvetica, Arial, sans-serif" font-size="10">{_svg_escape(xlabel)}</text>',
            f'<text x="18" y="54" font-family="Helvetica, Arial, sans-serif" font-size="10">{_svg_escape(ylabel + (" (log10)" if yscale_log else ""))}</text>',
        ]
        for y_tick in np.linspace(y_min, y_max, 5):
            y_pos = height - sy(float(y_tick))
            body.append(
                f'<line x1="{left:.2f}" y1="{y_pos:.2f}" x2="{left + plot_w:.2f}" y2="{y_pos:.2f}" stroke="#d9d9d9" stroke-width="0.7"/>'
            )
            body.append(
                f'<text x="25" y="{y_pos + 3:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="8">{y_tick:.3g}</text>'
            )
        for x_tick in np.unique(all_x):
            x_pos = sx(float(x_tick))
            body.append(
                f'<line x1="{x_pos:.2f}" y1="{height - bottom:.2f}" x2="{x_pos:.2f}" y2="{height - bottom - plot_h:.2f}" stroke="#d9d9d9" stroke-width="0.7"/>'
            )
            body.append(
                f'<text x="{x_pos - 8:.2f}" y="{height - bottom + 18:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="8">{x_tick:g}</text>'
            )
        body.append(
            f'<path d="M {left:.2f} {height - bottom:.2f} L {left:.2f} {height - bottom - plot_h:.2f} M {left:.2f} {height - bottom:.2f} L {left + plot_w:.2f} {height - bottom:.2f}" stroke="#000" fill="none" stroke-width="1"/>'
        )
        svg_colors = ["#1f59bf", "#cc332e", "#1f8c47"]
        for index, (label, x_arr, y_arr) in enumerate(clean_series):
            color = svg_colors[index % len(svg_colors)]
            points = [(sx(float(x)), height - sy(float(y))) for x, y in zip(x_arr, y_arr)]
            if points:
                point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                body.append(
                    f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="1.8"/>'
                )
                for x, y in points:
                    body.append(f'<rect x="{x - 2:.2f}" y="{y - 2:.2f}" width="4" height="4" fill="{color}"/>')
            legend_x = left + plot_w - 118
            legend_y = height - (bottom + plot_h - 18 - 16 * index)
            body.append(f'<rect x="{legend_x:.2f}" y="{legend_y - 4:.2f}" width="10" height="4" fill="{color}"/>')
            body.append(
                f'<text x="{legend_x + 16:.2f}" y="{legend_y:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="9">{_svg_escape(label)}</text>'
            )
        _write_minimal_svg(out_path, body, width=width, height=height)
        return

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
        r, g, b = colors[index % len(colors)]
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


def setup_matplotlib_style() -> None:
    """Apply a compact report-friendly Matplotlib style."""
    if plt is None:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


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
        _fallback_line_figure(series, xlabel, ylabel, title, out_path, yscale_log=yscale_log)
        return out_path

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for label, x, y in series:
        color = METHOD_COLORS.get(label, "#2A9D8F")
        marker = METHOD_MARKERS.get(label, "o")
        ax.plot(x, y, marker=marker, linewidth=2.2, markersize=5.5, label=label, color=color)
    if yscale_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metric(
    rows: list[dict],
    experiment_type: str,
    metric_mean: str,
    metric_std: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    fig_format: str = "svg",
    yscale_log: bool = False,
) -> Path:
    """Plot one metric with mean lines and standard-deviation bands."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / f"{Path(filename).stem}.{fig_format}"
    series = rows_for(rows, experiment_type, metric_mean, metric_std)

    if plt is None:
        plot_series = [(method, item[0], item[1]) for method, item in series.items()]
        return save_line_plot(plot_series, xlabel, ylabel, title, out_path, yscale_log=yscale_log)

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for method, (x, y, y_std) in series.items():
        color = METHOD_COLORS.get(method, "#2A9D8F")
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(x, y, marker=marker, linewidth=2.2, markersize=5.5, label=method, color=color)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.14, linewidth=0.0)
    if yscale_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format=fig_format, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_panel(
    ax,
    rows: list[dict],
    experiment_type: str,
    metric_mean: str,
    metric_std: str,
    title: str,
    ylabel: str,
) -> None:
    """Draw one dashboard panel."""
    series = rows_for(rows, experiment_type, metric_mean, metric_std)
    for method, (x, y, y_std) in series.items():
        color = METHOD_COLORS.get(method, "#2A9D8F")
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(x, y, marker=marker, linewidth=2.0, markersize=4.8, label=method, color=color)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.12, linewidth=0.0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(EXPERIMENT_LABELS.get(experiment_type, experiment_type))
    ax.set_ylabel(ylabel)


def plot_performance_dashboard(rows: list[dict], fig_format: str = "svg") -> Path:
    """Save a four-panel dashboard that summarizes the main research questions."""
    out_path = FIGURE_DIR / f"performance_dashboard.{fig_format}"
    if plt is None:
        return _fallback_report_figure(
            out_path,
            "Performance dashboard",
            [
                "Advanced dashboard requires Matplotlib.",
                "Run on the server after `pip install -r requirements.txt`.",
            ],
        )

    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    _plot_panel(
        axes[0, 0],
        rows,
        "noise",
        "clean_reprojection_rmse_mean",
        "clean_reprojection_rmse_std",
        "A. Noise sensitivity",
        "Clean RMSE (px)",
    )
    _plot_panel(
        axes[0, 1],
        rows,
        "outlier",
        "rotation_error_deg_mean",
        "rotation_error_deg_std",
        "B. Outlier sensitivity",
        "Rotation error (deg)",
    )
    _plot_panel(
        axes[1, 0],
        rows,
        "initialization",
        "rotation_error_deg_mean",
        "rotation_error_deg_std",
        "C. Initialization basin",
        "Rotation error (deg)",
    )
    _plot_panel(
        axes[1, 1],
        rows,
        "point_count",
        "clean_reprojection_rmse_mean",
        "clean_reprojection_rmse_std",
        "D. Match count under outliers",
        "Clean RMSE (px)",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Robust PnP-LM stress-test summary", x=0.02, y=1.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, format=fig_format, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_huber_delta_sweep(rows: list[dict], fig_format: str = "svg") -> Path:
    """Plot Huber threshold ablation."""
    out_path = FIGURE_DIR / f"huber_delta_sweep.{fig_format}"
    delta_rows = [
        r for r in rows if r["experiment_type"] == "huber_delta" and r["method"] == "Huber-LM"
    ]
    delta_rows.sort(key=lambda r: r["setting_value"])
    if plt is None:
        series = [
            (
                "Clean RMSE",
                np.array([r["setting_value"] for r in delta_rows], dtype=float),
                np.array([r["clean_reprojection_rmse_mean"] for r in delta_rows], dtype=float),
            )
        ]
        return save_line_plot(
            series,
            "Huber delta",
            "Clean RMSE (px)",
            "Huber threshold ablation",
            out_path,
        )

    setup_matplotlib_style()
    x = np.array([r["setting_value"] for r in delta_rows], dtype=float)
    clean = np.array([r["clean_reprojection_rmse_mean"] for r in delta_rows], dtype=float)
    clean_std = np.array([r["clean_reprojection_rmse_std"] for r in delta_rows], dtype=float)
    rot = np.array([r["rotation_error_deg_mean"] for r in delta_rows], dtype=float)
    rot_std = np.array([r["rotation_error_deg_std"] for r in delta_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].plot(x, clean, marker="o", color="#D45113", linewidth=2.2)
    axes[0].fill_between(x, clean - clean_std, clean + clean_std, color="#D45113", alpha=0.15)
    axes[0].set_title("Clean reprojection error", loc="left", fontweight="bold")
    axes[0].set_xlabel("Huber delta (px)")
    axes[0].set_ylabel("Clean RMSE (px)")

    axes[1].plot(x, rot, marker="s", color="#2A9D8F", linewidth=2.2)
    axes[1].fill_between(x, rot - rot_std, rot + rot_std, color="#2A9D8F", alpha=0.15)
    axes[1].set_title("Pose error", loc="left", fontweight="bold")
    axes[1].set_xlabel("Huber delta (px)")
    axes[1].set_ylabel("Rotation error (deg)")

    fig.suptitle("Huber threshold ablation under 25% outliers", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(out_path, format=fig_format, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_outlier_boxplot(detail_rows: list[dict], fig_format: str = "svg") -> Path:
    """Plot per-trial clean RMSE distribution under outlier stress."""
    out_path = FIGURE_DIR / f"outlier_clean_rmse_boxplot.{fig_format}"
    outlier_rows = [r for r in detail_rows if r["experiment_type"] == "outlier"]
    settings = sorted({float(r["setting_value"]) for r in outlier_rows})
    if plt is None:
        return _fallback_report_figure(
            out_path,
            "Outlier RMSE boxplot",
            ["Advanced boxplot requires Matplotlib.", f"Settings: {settings}"],
        )

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    positions = []
    data = []
    colors = []
    labels = []
    width = 0.28
    for idx, setting in enumerate(settings, start=1):
        for offset, method in [(-width / 1.4, "Ordinary-LM"), (width / 1.4, "Huber-LM")]:
            values = [
                float(r["clean_reprojection_rmse"])
                for r in outlier_rows
                if float(r["setting_value"]) == setting and r["method"] == method
            ]
            if not values:
                continue
            positions.append(idx + offset)
            data.append(values)
            colors.append(METHOD_COLORS.get(method, "#2A9D8F"))
            labels.append(method)

    boxes = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111111", "linewidth": 1.2},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
        patch.set_edgecolor(color)
    for element in ["whiskers", "caps"]:
        for artist in boxes[element]:
            artist.set_color("#4A4A4A")
    ax.set_xticks(range(1, len(settings) + 1))
    ax.set_xticklabels([f"{s:g}" for s in settings])
    ax.set_xlabel("Outlier ratio")
    ax.set_ylabel("Clean RMSE (px)")
    ax.set_title("Per-trial distribution under outliers", loc="left", fontweight="bold")

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="s", linestyle="", markersize=9, label=m)
        for m in METHOD_ORDER
    ]
    ax.legend(handles=handles, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, format=fig_format, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_robustness_gain_heatmap(rows: list[dict], fig_format: str = "svg") -> Path:
    """Plot Ordinary/Huber clean-RMSE ratio across stress tests."""
    out_path = FIGURE_DIR / f"robustness_gain_heatmap.{fig_format}"
    experiments = ["noise", "outlier", "initialization", "point_count"]
    row_cells: list[list[tuple[float, float] | None]] = []
    max_cols = 0
    for experiment_type in experiments:
        settings = sorted({
            float(r["setting_value"])
            for r in rows
            if r["experiment_type"] == experiment_type
        })
        cells: list[tuple[float, float] | None] = []
        for setting in settings:
            ordinary = [
                r
                for r in rows
                if r["experiment_type"] == experiment_type
                and float(r["setting_value"]) == setting
                and r["method"] == "Ordinary-LM"
            ]
            huber = [
                r
                for r in rows
                if r["experiment_type"] == experiment_type
                and float(r["setting_value"]) == setting
                and r["method"] == "Huber-LM"
            ]
            if ordinary and huber:
                ratio = ordinary[0]["clean_reprojection_rmse_mean"] / max(
                    huber[0]["clean_reprojection_rmse_mean"], 1e-12
                )
                cells.append((setting, ratio))
        max_cols = max(max_cols, len(cells))
        row_cells.append(cells)

    if plt is None:
        lines = []
        for experiment_type, cells in zip(experiments, row_cells):
            text = ", ".join(f"{setting:g}: {ratio:.2f}x" for setting, ratio in cells)
            lines.append(f"{experiment_type}: {text}")
        return _fallback_report_figure(out_path, "Robustness gain heatmap", lines)

    data = np.full((len(experiments), max_cols), np.nan, dtype=float)
    labels = [["" for _ in range(max_cols)] for _ in experiments]
    for row_idx, cells in enumerate(row_cells):
        for col_idx, (setting, ratio) in enumerate(cells):
            data[row_idx, col_idx] = ratio
            labels[row_idx][col_idx] = f"{setting:g}\n{ratio:.1f}x"

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    max_value = float(np.nanmax(data)) if np.any(np.isfinite(data)) else 1.0
    norm = Normalize(vmin=1.0, vmax=max(1.0, max_value))
    cmap = plt.get_cmap("YlOrRd")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(data[i, j])),
                    edgecolor="white",
                    linewidth=1.2,
                )
                ax.add_patch(rect)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels([EXPERIMENT_LABELS[e] for e in experiments])
    ax.set_xticks(range(max_cols))
    ax.set_xticklabels([f"level {i + 1}" for i in range(max_cols)])
    ax.set_xlim(-0.5, max_cols - 0.5)
    ax.set_ylim(len(experiments) - 0.5, -0.5)
    ax.set_title("Robustness gain: Ordinary clean RMSE / Huber clean RMSE", loc="left", fontweight="bold")
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color="#1A1A1A")
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Error ratio (higher means Huber improves more)")
    fig.tight_layout()
    fig.savefig(out_path, format=fig_format, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate vector experiment figures.")
    parser.add_argument(
        "--format",
        choices=["svg", "pdf", "both"],
        default="svg",
        help="Output figure format. Default: svg.",
    )
    return parser.parse_args()


def generate_figures(rows: list[dict], detail_rows: list[dict], fig_format: str) -> list[Path]:
    """Generate all figures for one output format."""
    return [
        plot_metric(
            rows,
            "noise",
            "clean_reprojection_rmse_mean",
            "clean_reprojection_rmse_std",
            "noise_sigma",
            "Clean reprojection RMSE (px)",
            "Noise sensitivity",
            "noise_clean_reprojection_rmse.pdf",
            fig_format=fig_format,
        ),
        plot_metric(
            rows,
            "noise",
            "rotation_error_deg_mean",
            "rotation_error_deg_std",
            "noise_sigma",
            "Rotation error (deg)",
            "Rotation error under noise",
            "noise_rotation_error.pdf",
            fig_format=fig_format,
        ),
        plot_metric(
            rows,
            "outlier",
            "clean_reprojection_rmse_mean",
            "clean_reprojection_rmse_std",
            "outlier_ratio",
            "Clean reprojection RMSE (px)",
            "Outlier sensitivity",
            "outlier_clean_reprojection_rmse.pdf",
            fig_format=fig_format,
        ),
        plot_metric(
            rows,
            "outlier",
            "rotation_error_deg_mean",
            "rotation_error_deg_std",
            "outlier_ratio",
            "Rotation error (deg)",
            "Rotation error under outliers",
            "outlier_rotation_error.pdf",
            fig_format=fig_format,
        ),
        plot_metric(
            rows,
            "outlier",
            "translation_error_mean",
            "translation_error_std",
            "outlier_ratio",
            "Translation error",
            "Translation error under outliers",
            "translation_error.pdf",
            fig_format=fig_format,
        ),
        plot_metric(
            rows,
            "initialization",
            "rotation_error_deg_mean",
            "rotation_error_deg_std",
            "initial rotation perturbation (deg)",
            "Rotation error (deg)",
            "Initialization sensitivity",
            "initialization_sensitivity.pdf",
            fig_format=fig_format,
        ),
        plot_metric(
            rows,
            "point_count",
            "clean_reprojection_rmse_mean",
            "clean_reprojection_rmse_std",
            "number of correspondences",
            "Clean reprojection RMSE (px)",
            "Point-count sensitivity under outliers",
            "point_count_sensitivity.pdf",
            fig_format=fig_format,
        ),
        plot_huber_delta_sweep(rows, fig_format=fig_format),
        plot_outlier_boxplot(detail_rows, fig_format=fig_format),
        plot_robustness_gain_heatmap(rows, fig_format=fig_format),
        plot_performance_dashboard(rows, fig_format=fig_format),
    ]


def main() -> None:
    """Read CSV summaries and generate report-ready vector figures."""
    args = parse_args()
    summary_path = RESULTS_DIR / "summary_results.csv"
    detail_path = RESULTS_DIR / "experiment_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find {summary_path}. Run `python code/experiment.py --mode quick` first."
        )
    if not detail_path.exists():
        raise FileNotFoundError(
            f"Cannot find {detail_path}. Run `python code/experiment.py --mode quick` first."
        )

    rows = read_summary(summary_path)
    detail_rows = read_detail(detail_path)
    formats = ["svg", "pdf"] if args.format == "both" else [args.format]
    figure_paths: list[Path] = []
    for fig_format in formats:
        figure_paths.extend(generate_figures(rows, detail_rows, fig_format))
    print("Generated figures:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
