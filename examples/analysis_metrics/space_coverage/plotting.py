import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adtool.examples.analysis_metrics.shared import series_colors


def plot_progression_curves(
    out_path,
    series,
    title,
    y_label,
    plot_config,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    colors = series_colors(len(series), [plot_config.color_a, plot_config.color_b])
    for index, (steps, counts, label) in enumerate(series):
        ax.plot(
            steps,
            counts,
            color=colors[index],
            linewidth=plot_config.line_width,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
