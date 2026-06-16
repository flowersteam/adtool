import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_progression_curves(
    out_path,
    steps_a,
    counts_a,
    steps_b,
    counts_b,
    title,
    label_a,
    label_b,
    y_label,
    plot_config,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    ax.plot(
        steps_a,
        counts_a,
        color=plot_config.color_a,
        linewidth=plot_config.line_width,
        label=label_a,
    )
    ax.plot(
        steps_b,
        counts_b,
        color=plot_config.color_b,
        linewidth=plot_config.line_width,
        label=label_b,
    )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
