import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..shared import series_colors


def plot_dimension_pair_scatter(
    out_path,
    series,
    x_label,
    y_label,
    plot_config,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    colors = series_colors(len(series), [plot_config.color_a, plot_config.color_b])
    for index, (x_values, y_values, label) in enumerate(series):
        ax.scatter(
            x_values,
            y_values,
            color=colors[index],
            alpha=plot_config.alpha,
            label=label,
            edgecolors="none",
        )

    ax.set_title(f"X = {x_label} | Y = {y_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
