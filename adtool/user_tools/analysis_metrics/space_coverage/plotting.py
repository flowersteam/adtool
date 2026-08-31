import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..shared import series_color_map


def plot_progression_curves(
    out_path,
    series,
    title,
    y_label,
    plot_config,
):
    fig, ax = plt.subplots(figsize=plot_config.figsize)
    color_keys = []
    for index, item in enumerate(series):
        color_keys.extend(
            ("branch", branch_id) if branch_id is not None else ("series", index)
            for _, _, _, branch_id in item["segments"]
        )
        color_keys.extend(
            ("branch", branch_id)
            for _, _, _, branch_id in item["checkpoints"]
        )
    colors = series_color_map(
        color_keys,
        [plot_config.color_a, plot_config.color_b],
    )
    used_labels = set()
    for index, item in enumerate(series):
        for steps, counts, label, branch_id in item["segments"]:
            color_key = (
                ("branch", branch_id)
                if branch_id is not None
                else ("series", index)
            )
            color = colors[color_key]
            legend_label = label if label not in used_labels else "_nolegend_"
            used_labels.add(label)
            ax.plot(
                steps,
                counts,
                color=color,
                linewidth=plot_config.line_width,
                label=legend_label,
            )
        for step, count, _, branch_id in item["checkpoints"]:
            color = colors[("branch", branch_id)]
            ax.scatter(
                [step],
                [count],
                color=color,
                edgecolors="white",
                linewidths=0.7,
                s=34,
                zorder=3,
            )
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
