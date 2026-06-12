from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def density_curve(
    values,
    bounds,
    points,
    integer_values=False,
):
    if integer_values:
        min_value = int(np.ceil(bounds[0]))
        max_value = int(np.floor(bounds[1]))
        bins = np.arange(min_value - 0.5, max_value + 1.5, 1.0)
        density = np.histogram(values, bins=bins, density=True)[0]
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


def _new_figure(plot_config):
    return plt.subplots(figsize=plot_config.figsize)


def _save_figure(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_density_series(
    ax,
    curve,
    color,
    label,
    plot_config,
):
    xs, ys = curve
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=plot_config.line_width,
        label=label,
    )
    ax.fill_between(xs, ys, color=color, alpha=plot_config.alpha)


def _set_integer_x_axis(ax, curve_a, curve_b):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    min_value = min(float(np.min(curve_a[0])), float(np.min(curve_b[0])))
    max_value = max(float(np.max(curve_a[0])), float(np.max(curve_b[0])))
    ax.set_xlim(min_value - 0.5, max_value + 0.5)


def plot_density_curves(
    out_path,
    curve_a,
    curve_b,
    dim_label,
    label_a,
    label_b,
    plot_config,
    integer_x=False,
):
    fig, ax = _new_figure(plot_config)
    _plot_density_series(ax, curve_a, plot_config.color_a, label_a, plot_config)
    _plot_density_series(ax, curve_b, plot_config.color_b, label_b, plot_config)

    ax.set_title(dim_label)
    ax.set_xlabel(dim_label)
    ax.set_ylabel("density")
    if integer_x:
        _set_integer_x_axis(ax, curve_a, curve_b)
    ax.legend()
    _save_figure(fig, out_path)


def plot_dimension_pair_scatter(
    out_path,
    x_values_a,
    y_values_a,
    x_values_b,
    y_values_b,
    x_label,
    y_label,
    label_a,
    label_b,
    plot_config,
):
    fig, ax = _new_figure(plot_config)
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
    _save_figure(fig, out_path)
