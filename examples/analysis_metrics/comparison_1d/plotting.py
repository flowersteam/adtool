import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from adtool.examples.analysis_metrics.shared import series_colors


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
    curves,
    title,
    labels,
    plot_config,
    integer_x=False,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    colors = series_colors(len(curves), [plot_config.color_a, plot_config.color_b])
    for index, ((xs, ys), label) in enumerate(zip(curves, labels)):
        color = colors[index]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=plot_config.line_width,
            label=label,
        )
        ax.fill_between(xs, ys, color=color, alpha=plot_config.alpha)

    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        min_value = min(float(np.min(xs)) for xs, _ in curves)
        max_value = max(float(np.max(xs)) for xs, _ in curves)
        ax.set_xlim(min_value - 0.5, max_value + 0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
