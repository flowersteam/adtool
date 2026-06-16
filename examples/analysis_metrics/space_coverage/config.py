from dataclasses import dataclass, field

from adtool.examples.analysis_metrics.shared import ProjectionConfig, load_projection_config


DEFAULT_COLOR_A = "#4c78a8"
DEFAULT_COLOR_B = "#f58518"
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_FIGSIZE = (7.0, 4.0)
DEFAULT_OUTPUT_FORMAT = "png"


@dataclass(frozen=True)
class SpaceCoverageMetricConfig:
    path: str
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SpaceCoveragePlotConfig:
    color_a: str = DEFAULT_COLOR_A
    color_b: str = DEFAULT_COLOR_B
    line_width: float = DEFAULT_LINE_WIDTH
    figsize: tuple = DEFAULT_FIGSIZE
    output_format: str = DEFAULT_OUTPUT_FORMAT


@dataclass(frozen=True)
class SpaceCoverageConfig:
    projection: ProjectionConfig
    metric: SpaceCoverageMetricConfig
    plot: object = field(default_factory=SpaceCoveragePlotConfig)


def load_space_coverage_config(section):
    metric = section["metric"]
    plot = section.get("plot") or {}
    return SpaceCoverageConfig(
        projection=load_projection_config(section),
        metric=SpaceCoverageMetricConfig(
            path=metric["path"],
            config=dict(metric.get("config") or {}),
        ),
        plot=SpaceCoveragePlotConfig(
            color_a=str(plot.get("color_a", DEFAULT_COLOR_A)),
            color_b=str(plot.get("color_b", DEFAULT_COLOR_B)),
            line_width=float(plot.get("line_width", DEFAULT_LINE_WIDTH)),
            figsize=tuple(map(float, plot.get("figsize", DEFAULT_FIGSIZE))),
            output_format=str(plot.get("format", DEFAULT_OUTPUT_FORMAT)).lstrip("."),
        ),
    )
