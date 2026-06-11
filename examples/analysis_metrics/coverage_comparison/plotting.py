from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except Exception as exc:  # pragma: no cover - matplotlib optional
    plt = None
    MaxNLocator = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


DensityCurve = tuple[np.ndarray, np.ndarray]


def _ensure_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for coverage comparison plots; "
            f"import failed with: {_MATPLOTLIB_ERROR}"
        )


def density_curve(
    values: np.ndarray,
    bounds: tuple[float, float],
    points: int,
    integer_values: bool = False,
) -> DensityCurve:
    if integer_values:
        min_value = int(np.ceil(bounds[0]))
        max_value = int(np.floor(bounds[1]))
        bins = np.arange(min_value - 0.5, max_value + 1.5, 1.0)
        density, edges = np.histogram(values, bins=bins, density=True)
        xs = np.arange(min_value, max_value + 1, dtype=float)
        return xs, density

    density, edges = np.histogram(
        values,
        bins=max(2, int(points)),
        range=bounds,
        density=True,
    )
    xs = (edges[:-1] + edges[1:]) / 2.0
    return xs, density


def plot_density_curves(
    out_path: Path,
    curve_a: DensityCurve,
    curve_b: DensityCurve,
    dim_label: str,
    label_a: str,
    label_b: str,
    plot_config: Any,
    integer_x: bool = False,
) -> None:
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=plot_config.figsize)
    x_a, y_a = curve_a
    x_b, y_b = curve_b

    ax.plot(
        x_a,
        y_a,
        color=plot_config.color_a,
        linewidth=plot_config.line_width,
        label=label_a,
    )
    ax.fill_between(x_a, y_a, color=plot_config.color_a, alpha=plot_config.alpha)

    ax.plot(
        x_b,
        y_b,
        color=plot_config.color_b,
        linewidth=plot_config.line_width,
        label=label_b,
    )
    ax.fill_between(x_b, y_b, color=plot_config.color_b, alpha=plot_config.alpha)

    ax.set_title(f"{dim_label}")
    ax.set_xlabel(dim_label)
    ax.set_ylabel("density")
    if integer_x and MaxNLocator is not None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        all_x = np.concatenate([x_a, x_b])
        if all_x.size > 0:
            ax.set_xlim(float(np.min(all_x)) - 0.5, float(np.max(all_x)) + 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dimension_pair_scatter(
    out_path: Path,
    x_values_a: np.ndarray,
    y_values_a: np.ndarray,
    x_values_b: np.ndarray,
    y_values_b: np.ndarray,
    x_label: str,
    y_label: str,
    label_a: str,
    label_b: str,
    plot_config: Any,
) -> None:
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=plot_config.figsize)
    ax.scatter(
        x_values_a,
        y_values_a,
        color=plot_config.color_a,
        alpha=plot_config.alpha,
        label=label_a,
        edgecolors="none",
    )
    ax.scatter(
        x_values_b,
        y_values_b,
        color=plot_config.color_b,
        alpha=plot_config.alpha,
        label=label_b,
        edgecolors="none",
    )

    ax.set_title(f"X = {x_label} | Y = {y_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
