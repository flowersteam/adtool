from dataclasses import dataclass, field

from adtool.examples.analysis_metrics.shared import ProjectionConfig, load_projection_config


DEFAULT_POINTS = 256
DEFAULT_COLOR_A = "#4c78a8"
DEFAULT_COLOR_B = "#f58518"
DEFAULT_ALPHA = 0.35
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_FIGSIZE = (7.0, 4.0)
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_DIMENSIONS = "all"


@dataclass(frozen=True)
class Comparison1DPlotConfig:
    points: int = DEFAULT_POINTS
    color_a: str = DEFAULT_COLOR_A
    color_b: str = DEFAULT_COLOR_B
    alpha: float = DEFAULT_ALPHA
    line_width: float = DEFAULT_LINE_WIDTH
    figsize: tuple = DEFAULT_FIGSIZE
    output_format: str = DEFAULT_OUTPUT_FORMAT


@dataclass(frozen=True)
class Comparison1DConfig:
    projection: ProjectionConfig
    dimensions: object = DEFAULT_DIMENSIONS
    plot: object = field(default_factory=Comparison1DPlotConfig)


def load_comparison_1d_config(section):
    plot = section.get("plot") or {}
    return Comparison1DConfig(
        projection=load_projection_config(section),
        dimensions=section.get("dimensions", DEFAULT_DIMENSIONS),
        plot=Comparison1DPlotConfig(
            points=int(plot.get("points", DEFAULT_POINTS)),
            color_a=str(plot.get("color_a", DEFAULT_COLOR_A)),
            color_b=str(plot.get("color_b", DEFAULT_COLOR_B)),
            alpha=float(plot.get("alpha", DEFAULT_ALPHA)),
            line_width=float(plot.get("line_width", DEFAULT_LINE_WIDTH)),
            figsize=tuple(map(float, plot.get("figsize", DEFAULT_FIGSIZE))),
            output_format=str(plot.get("format", DEFAULT_OUTPUT_FORMAT)).lstrip("."),
        ),
    )
