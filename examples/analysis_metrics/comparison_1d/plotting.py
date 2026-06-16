import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def density_curve(values, bounds, points, integer_values=False):
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


def plot_density_curves(
    out_path,
    curve_a,
    curve_b,
    title,
    label_a,
    label_b,
    plot_config,
    integer_x=False,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    xs_a, ys_a = curve_a
    xs_b, ys_b = curve_b

    ax.plot(
        xs_a,
        ys_a,
        color=plot_config.color_a,
        linewidth=plot_config.line_width,
        label=label_a,
    )
    ax.fill_between(xs_a, ys_a, color=plot_config.color_a, alpha=plot_config.alpha)

    ax.plot(
        xs_b,
        ys_b,
        color=plot_config.color_b,
        linewidth=plot_config.line_width,
        label=label_b,
    )
    ax.fill_between(xs_b, ys_b, color=plot_config.color_b, alpha=plot_config.alpha)

    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        min_value = min(float(np.min(xs_a)), float(np.min(xs_b)))
        max_value = max(float(np.max(xs_a)), float(np.max(xs_b)))
        ax.set_xlim(min_value - 0.5, max_value + 0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
