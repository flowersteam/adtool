import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adtool.user_tools.analysis_metrics.shared import AnalysisImage, AnalysisModule


class MutualMissHeatmapModule(AnalysisModule):
    module_id = "mutual_miss_heatmap"

    def run(self, datasets, labels, run_dir) -> dict:
        matrices = [
            np.mean(
                [
                    np.asarray(payload["raw_output"]["mutual"]["miss"], dtype=float)
                    for payload in dataset.payloads
                ],
                axis=0,
            )
            for dataset in datasets
        ]

        vmin = min(float(matrix.min()) for matrix in matrices)
        vmax = max(float(matrix.max()) for matrix in matrices)

        images = []
        for dataset_index, (label, matrix) in enumerate(zip(labels, matrices)):
            title = f"{label} average DDR row-miss count per discovery"
            image_name = f"mutual_miss_heatmap_{dataset_index}.png"
            self._plot_heatmap(run_dir / image_name, matrix, title, vmin, vmax)
            images.append(
                AnalysisImage(
                    file=image_name,
                    title=title,
                    plot_type="matrix heatmap",
                    dimensions=[0, 1],
                    bounds=[
                        [0, int(matrix.shape[1] - 1)],
                        [0, int(matrix.shape[0] - 1)],
                    ],
                ).to_payload()
            )

        return {
            "title": "Average DDR row-miss count",
            "images": images,
            "series": list(labels),
            "summary": [
                f"{len(labels)} datasets",
                f"{len(images)} graphs",
            ],
        }

    def _plot_heatmap(self, out_path, matrix, title, vmin, vmax):
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        image = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="upper",
        )
        ax.set_title(title)
        ax.set_xlabel("DDR bank index")
        ax.set_ylabel("DDR row index")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = float(matrix[row_index, col_index])
                text_color = "black" if value > (vmin + vmax) / 2.0 else "white"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, label="Mean DDR row misses per discovery")
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
