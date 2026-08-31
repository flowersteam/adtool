import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from ..shared import series_color_map


def aggregate_coordinate_opacities(x_values, y_values, max_opacity):
    """Aggregate exact coordinates and scale opacity within one dataset."""
    if len(x_values) == 0:
        return np.array([]), np.array([]), np.array([])
    coordinates, counts = np.unique(
        np.column_stack((x_values, y_values)),
        axis=0,
        return_counts=True,
    )
    maximum_count = int(np.max(counts))
    opacities = float(max_opacity) * counts.astype(float) / maximum_count
    return coordinates[:, 0], coordinates[:, 1], opacities


def plot_dimension_pair_scatter(
    out_path,
    series,
    x_label,
    y_label,
    plot_config,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    color_keys = [
        ("branch", branch_id) if branch_id is not None else ("series", index)
        for index, (_, _, _, branch_id) in enumerate(series)
    ]
    colors = series_color_map(
        color_keys,
        [plot_config.color_a, plot_config.color_b],
    )
    legend_handles = []
    for index, (x_values, y_values, label, branch_id) in enumerate(series):
        x_coordinates, y_coordinates, opacities = aggregate_coordinate_opacities(
            x_values,
            y_values,
            plot_config.max_opacity,
        )
        color = colors[color_keys[index]]
        rgba = np.tile(to_rgba(color), (len(opacities), 1))
        rgba[:, 3] = opacities
        ax.scatter(
            x_coordinates,
            y_coordinates,
            color=rgba,
            edgecolors="none",
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=color,
                alpha=plot_config.max_opacity,
                label=label,
            )
        )

    ax.set_title(f"X = {x_label} | Y = {y_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
