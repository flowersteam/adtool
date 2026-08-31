from dataclasses import dataclass, field

from ..shared import ProjectionConfig, load_projection_config


DEFAULT_COLOR_A = "#4c78a8"
DEFAULT_COLOR_B = "#f58518"
DEFAULT_MAX_OPACITY = 0.8
DEFAULT_FIGSIZE = (7.0, 4.0)
DEFAULT_OUTPUT_FORMAT = "png"


@dataclass(frozen=True)
class Comparison2DPlotConfig:
    color_a: str = DEFAULT_COLOR_A
    color_b: str = DEFAULT_COLOR_B
    max_opacity: float = DEFAULT_MAX_OPACITY
    figsize: tuple = DEFAULT_FIGSIZE
    output_format: str = DEFAULT_OUTPUT_FORMAT


@dataclass(frozen=True)
class Comparison2DConfig:
    projection: ProjectionConfig
    pairs: list = field(default_factory=list)
    plot: object = field(default_factory=Comparison2DPlotConfig)


def load_comparison_2d_config(section):
    plot = section.get("plot") or {}
    max_opacity = float(plot.get("max_opacity", DEFAULT_MAX_OPACITY))
    if not 0.0 < max_opacity <= DEFAULT_MAX_OPACITY:
        raise ValueError(
            f"comparison_2d plot.max_opacity must be in (0, {DEFAULT_MAX_OPACITY}]"
        )
    return Comparison2DConfig(
        projection=load_projection_config(section),
        pairs=[(int(x_dim), int(y_dim)) for x_dim, y_dim in section.get("pairs", [])],
        plot=Comparison2DPlotConfig(
            color_a=str(plot.get("color_a", DEFAULT_COLOR_A)),
            color_b=str(plot.get("color_b", DEFAULT_COLOR_B)),
            max_opacity=max_opacity,
            figsize=tuple(map(float, plot.get("figsize", DEFAULT_FIGSIZE))),
            output_format=str(plot.get("format", DEFAULT_OUTPUT_FORMAT)).lstrip("."),
        ),
    )
