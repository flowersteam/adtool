import colorsys


DIVERSE_COLOR_PALETTE = (
    "#e13b3f",  # red
    "#3488c5",  # blue
    "#3aae3f",  # green
    "#ff7f0e",  # orange
    "#a06dcc",  # purple
    "#a6695d",  # brown
    "#e77ac6",  # pink
    "#e5c100",  # yellow
    "#28bfd0",  # cyan
    "#929292",  # gray
    "#159a9c",  # teal
    "#c1c329",  # olive
    "#b84d58",  # burgundy
    "#16b58a",  # emerald
    "#7d5ba6",  # violet
)


def _append_unique(colors, candidates, count):
    if len(colors) >= count:
        return
    used = {color.lower() for color in colors}
    for candidate in candidates:
        normalized = str(candidate).lower()
        if normalized in used:
            continue
        colors.append(str(candidate))
        used.add(normalized)
        if len(colors) == count:
            break


def _rgb(color):
    value = str(color).lstrip("#")
    if len(value) == 3:
        value = "".join(component * 2 for component in value)
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return None


def _color_distance(left, right):
    return sum(
        (left_value - right_value) ** 2
        for left_value, right_value in zip(left, right)
    )


def _append_most_distinct(colors, candidates, count):
    remaining = []
    seen = {color.lower() for color in colors}
    for candidate in candidates:
        normalized = str(candidate).lower()
        if normalized not in seen:
            remaining.append(str(candidate))
            seen.add(normalized)

    while remaining and len(colors) < count:
        existing_rgb = [rgb for rgb in map(_rgb, colors) if rgb is not None]

        def minimum_distance(candidate):
            candidate_rgb = _rgb(candidate)
            if candidate_rgb is None or not existing_rgb:
                return float("inf")
            return min(
                _color_distance(candidate_rgb, color_rgb)
                for color_rgb in existing_rgb
            )

        best_index = max(
            range(len(remaining)),
            key=lambda index: minimum_distance(remaining[index]),
        )
        colors.append(remaining.pop(best_index))


def series_colors(count, first_colors=()):
    """Return distinct, deliberately varied categorical colors."""
    count = max(0, int(count))
    if count == 0:
        return []
    colors = []
    _append_unique(colors, first_colors, count)
    _append_most_distinct(colors, DIVERSE_COLOR_PALETTE, count)

    styles = (
        (0.78, 0.90),
        (0.56, 0.96),
        (0.86, 0.78),
        (0.64, 0.84),
    )
    generated = []
    for generated_index in range(max(64, count * 8)):
        hue = (0.11 + generated_index * 0.61803398875) % 1.0
        saturation, value = styles[generated_index % len(styles)]
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        generated.append(
            f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
        )
    _append_most_distinct(colors, generated, count)
    return colors


def series_color_map(keys, first_colors=()):
    """Assign one distinct color to each unique key, preserving key order."""
    unique_keys = list(dict.fromkeys(keys))
    return dict(zip(unique_keys, series_colors(len(unique_keys), first_colors)))
