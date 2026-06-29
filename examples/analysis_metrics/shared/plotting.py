import colorsys


def series_colors(count, first_colors):
    colors = list(first_colors[:count])
    while len(colors) < count:
        hue = (0.08 + (len(colors) - len(first_colors)) * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.95)
        colors.append(
            f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
        )
    return colors
