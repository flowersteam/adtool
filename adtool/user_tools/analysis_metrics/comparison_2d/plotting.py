import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..shared import series_color_map


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
    for index, (x_values, y_values, label, branch_id) in enumerate(series):
        ax.scatter(
            x_values,
            y_values,
            color=colors[color_keys[index]],
            alpha=plot_config.alpha,
            label=label,
            edgecolors="none",
        )

    ax.set_title(f"X = {x_label} | Y = {y_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
