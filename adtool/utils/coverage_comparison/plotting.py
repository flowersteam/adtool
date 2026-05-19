from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - matplotlib optional
    plt = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


def _ensure_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for coverage comparison plots; "
            f"import failed with: {_MATPLOTLIB_ERROR}"
        )


def _scott_bandwidth(values: np.ndarray) -> float:
    if values.size <= 1:
        return 1.0
    std = float(np.std(values))
    if std == 0.0:
        return 1.0
    return std * (values.size ** (-1.0 / 5.0))


def _kde_density(
    values: np.ndarray,
    xs: np.ndarray,
    bandwidth: Optional[float],
) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(xs)

    bw = float(bandwidth) if bandwidth is not None else _scott_bandwidth(values)
    bw = max(bw, 1e-9)
    diffs = xs[:, None] - values[None, :]
    kernel = np.exp(-0.5 * (diffs / bw) ** 2) / (bw * np.sqrt(2.0 * np.pi))
    return np.mean(kernel, axis=1)


def compute_density_curve(
    values: np.ndarray,
    bounds: Tuple[float, float],
    points: int,
    bandwidth: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(bounds[0], bounds[1], points)
    density = _kde_density(values, xs, bandwidth)
    return xs, density


def plot_density_curves(
    out_path,
    random_curve: Tuple[np.ndarray, np.ndarray],
    tool_curve: Tuple[np.ndarray, np.ndarray],
    label: str,
    random_color: str,
    tool_color: str,
    alpha: float,
    line_width: float,
    figsize: Tuple[float, float],
) -> None:
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    random_x, random_y = random_curve
    tool_x, tool_y = tool_curve

    ax.plot(random_x, random_y, color=random_color, linewidth=line_width, label="random")
    ax.fill_between(random_x, random_y, color=random_color, alpha=alpha)

    ax.plot(tool_x, tool_y, color=tool_color, linewidth=line_width, label="tool")
    ax.fill_between(tool_x, tool_y, color=tool_color, alpha=alpha)

    ax.set_title(f"Coverage: {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
