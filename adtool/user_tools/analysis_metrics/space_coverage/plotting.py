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
    ordered_segments = []
    series_index = 0
    for item in series:
        for segment in reversed(item["segments"]):
            branch_id = segment[3]
            color_key = (
                ("branch", branch_id)
                if branch_id is not None
                else ("series", series_index)
            )
            ordered_segments.append((segment, color_key))
            series_index += 1

    color_keys = []
    color_keys.extend(color_key for _, color_key in ordered_segments)
    for item in series:
        color_keys.extend(
            ("branch", branch_id)
            for _, _, _, branch_id, _ in item["checkpoints"]
        )
    colors = series_color_map(color_keys)
    used_labels = set()
    for (steps, counts, label, _, selected_color), color_key in ordered_segments:
        color = selected_color or colors[color_key]
        legend_label = label if label not in used_labels else "_nolegend_"
        used_labels.add(label)
        ax.plot(
            steps,
            counts,
            color=color,
            linewidth=plot_config.line_width,
            label=legend_label,
        )

    for item in series:
        for step, count, _, branch_id, selected_color in item["checkpoints"]:
            color = selected_color or colors[("branch", branch_id)]
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
