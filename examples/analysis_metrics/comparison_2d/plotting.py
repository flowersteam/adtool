import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
