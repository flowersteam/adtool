import io
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _sum_non_negative(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.where(arr < 0, 0.0, arr)
    return float(np.sum(arr))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_layer_counts(core_output: Dict[str, Any], layer: str) -> Dict[str, Dict[str, float]]:
    if layer == "ddr":
        miss_combined = _sum_non_negative(core_output.get("miss", 0.0))
        hit_combined = _sum_non_negative(core_output.get("hits", 0.0))
        miss_read = _sum_non_negative(core_output.get("miss_read", 0.0))
        hit_read = _sum_non_negative(core_output.get("hits_read", 0.0))
        miss_write = _sum_non_negative(core_output.get("miss_write", 0.0))
        hit_write = _sum_non_negative(core_output.get("hits_write", 0.0))
    else:
        miss_combined = _safe_float(core_output.get("L2_miss", 0.0))
        hit_combined = _safe_float(core_output.get("L2_hit", 0.0))
        miss_read = _safe_float(core_output.get("L2_miss_read", 0.0))
        hit_read = _safe_float(core_output.get("L2_hit_read", 0.0))
        miss_write = _safe_float(core_output.get("L2_miss_write", 0.0))
        hit_write = _safe_float(core_output.get("L2_hit_write", 0.0))

    return {
        "combined": {"miss": miss_combined, "hit": hit_combined},
        "read": {"miss": miss_read, "hit": hit_read},
        "write": {"miss": miss_write, "hit": hit_write},
    }


def _normalize_program(program: Any) -> Dict[int, Tuple[str, int]]:
    if not isinstance(program, dict):
        return {}
    normalized = {}
    for cycle_raw, op_raw in program.items():
        try:
            cycle = int(cycle_raw)
        except (TypeError, ValueError):
            continue
        if isinstance(op_raw, (list, tuple)) and len(op_raw) == 2:
            op_type = str(op_raw[0])
            try:
                address = int(op_raw[1])
            except (TypeError, ValueError):
                continue
            normalized[cycle] = (op_type, address)
    return dict(sorted(normalized.items()))


def _draw_core_count_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    title: str,
    origin: Tuple[int, int],
    panel_size: Tuple[int, int],
    ddr_counts: Dict[str, Dict[str, float]],
    l2_counts: Dict[str, Dict[str, float]],
) -> None:
    ox, oy = origin
    width, height = panel_size
    draw.rectangle([ox, oy, ox + width, oy + height], outline=(180, 180, 180), width=2)
    draw.text((ox + 8, oy + 6), title, fill=(20, 20, 20), font=font)

    miss_color = (209, 82, 82)
    hit_color = (72, 168, 72)
    rows = [
        ("DDR Combined Hit", ddr_counts["combined"]["hit"], hit_color),
        ("DDR Read Miss", ddr_counts["read"]["miss"], miss_color),
        ("DDR Read Hit", ddr_counts["read"]["hit"], hit_color),
        ("DDR Write Miss", ddr_counts["write"]["miss"], miss_color),
        ("DDR Write Hit", ddr_counts["write"]["hit"], hit_color),
        ("L2 Combined Hit", l2_counts["combined"]["hit"], hit_color),
        ("L2 Read Miss", l2_counts["read"]["miss"], miss_color),
        ("L2 Read Hit", l2_counts["read"]["hit"], hit_color),
        ("L2 Write Miss", l2_counts["write"]["miss"], miss_color),
        ("L2 Write Hit", l2_counts["write"]["hit"], hit_color),
    ]

    max_count = max([v for _, v, _ in rows] + [1.0])
    x0 = ox + 170
    x1 = ox + width - 14
    bar_w = max(10, x1 - x0)
    y = oy + 30
    row_h = max(16, int((height - 40) / max(len(rows), 1)))

    for label, value, color in rows:
        draw.text((ox + 10, y), label, fill=(45, 45, 45), font=font)
        fill_w = int(bar_w * (value / max_count))
        draw.rectangle([x0, y + 2, x0 + bar_w, y + row_h - 4], outline=(210, 210, 210), width=1)
        if fill_w > 0:
            draw.rectangle([x0, y + 2, x0 + fill_w, y + row_h - 4], fill=color)
        draw.text((x0 + bar_w + 6, y), f"{value:.0f}", fill=(40, 40, 40), font=font)
        y += row_h


def _draw_summary_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    origin: Tuple[int, int],
    panel_size: Tuple[int, int],
    core0: Dict[str, Any],
    core1: Dict[str, Any],
    mutual: Dict[str, Any],
) -> None:
    ox, oy = origin
    width, height = panel_size
    draw.rectangle([ox, oy, ox + width, oy + height], outline=(180, 180, 180), width=2)
    draw.text((ox + 8, oy + 6), "Timing Figure", fill=(20, 20, 20), font=font)

    solo0 = _safe_float(core0.get("time_core0", 0))
    mutual0 = _safe_float(mutual.get("time_core0", 0))
    solo1 = _safe_float(core1.get("time_core1", 0))
    mutual1 = _safe_float(mutual.get("time_core1", 0))

    max_t = max([solo0, mutual0, solo1, mutual1, 1.0])
    x_label = ox + 10
    x_bar = ox + 130
    bar_w = width - 210
    y = oy + 34
    h = 14

    # Coherent timing palette: same hue per core, darker for mutual runtime.
    rows = [
        ("Core0 solo", solo0, (248, 193, 122)),
        ("Core0 mutual", mutual0, (137, 180, 245)),
        ("Core1 solo", solo1, (248, 193, 122)),
        ("Core1 mutual", mutual1, (137, 180, 245)),
    ]
    for label, val, color in rows:
        draw.text((x_label, y), label, fill=(45, 45, 45), font=font)
        draw.rectangle([x_bar, y, x_bar + bar_w, y + h], outline=(210, 210, 210), width=1)
        fill_w = int(bar_w * (val / max_t)) if max_t > 0 else 0
        if fill_w > 0:
            draw.rectangle([x_bar, y, x_bar + fill_w, y + h], fill=color)
        draw.text((x_bar + bar_w + 6, y), f"{val:.0f}", fill=(40, 40, 40), font=font)
        y += 22

    draw.text((x_label, oy + height - 24), "Solo vs mutual runtime per core", fill=(55, 55, 55), font=font)


def _draw_instruction_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    origin: Tuple[int, int],
    panel_size: Tuple[int, int],
    params: Dict[str, Any],
) -> None:
    ox, oy = origin
    width, height = panel_size
    draw.rectangle([ox, oy, ox + width, oy + height], outline=(180, 180, 180), width=2)
    draw.text((ox + 8, oy + 6), "Instruction Programs", fill=(20, 20, 20), font=font)

    dyn = params.get("dynamic_params", {}) if isinstance(params, dict) else {}
    prog0 = _normalize_program(dyn.get("core0", {}))
    prog1 = _normalize_program(dyn.get("core1", {}))

    col_w = (width - 30) // 2
    left_x = ox + 10
    right_x = left_x + col_w + 10
    y0 = oy + 26

    draw.text((left_x, y0), f"Core0 instructions ({len(prog0)})", fill=(45, 45, 45), font=font)
    draw.text((right_x, y0), f"Core1 instructions ({len(prog1)})", fill=(45, 45, 45), font=font)

    max_lines = 10

    def _render_program_lines(program: Dict[int, Tuple[str, int]], x: int, y: int):
        items = list(program.items())
        if len(items) <= max_lines:
            visible = items
            overflow = 0
        else:
            visible = items[: max_lines - 1]
            overflow = len(items) - len(visible)

        for cycle, (op_type, addr) in visible:
            draw.text((x, y), f"{cycle:>4}: {op_type:<5} @{addr}", fill=(60, 60, 60), font=font)
            y += 16
        if overflow > 0:
            draw.text((x, y), f"... +{overflow} more", fill=(90, 90, 90), font=font)

    _render_program_lines(prog0, left_x, y0 + 16)
    _render_program_lines(prog1, right_x, y0 + 16)


def render_interference_dashboard(payload: Dict[str, Any]) -> bytes:
    if isinstance(payload, dict) and "output" in payload:
        output = payload.get("output", {})
        params = payload.get("params", {})
    else:
        output = payload if isinstance(payload, dict) else {}
        params = {}

    core0 = output.get("core0", {}) if isinstance(output, dict) else {}
    core1 = output.get("core1", {}) if isinstance(output, dict) else {}
    mutual = output.get("mutual", {}) if isinstance(output, dict) else {}

    image = Image.new("RGB", (1200, 980), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((14, 10), "Interference Summary", fill=(10, 10, 10), font=font)

    ddr0 = _extract_layer_counts(core0, "ddr")
    ddr1 = _extract_layer_counts(core1, "ddr")
    l20 = _extract_layer_counts(core0, "l2")
    l21 = _extract_layer_counts(core1, "l2")

    _draw_core_count_panel(draw, font, "Core0 Hit/Miss Counts", (14, 36), (575, 500), ddr0, l20)
    _draw_core_count_panel(draw, font, "Core1 Hit/Miss Counts", (610, 36), (575, 500), ddr1, l21)
    _draw_summary_panel(draw, font, (14, 552), (1171, 190), core0, core1, mutual)
    _draw_instruction_panel(draw, font, (14, 758), (1171, 206), params)

    byte_img = io.BytesIO()
    image.save(byte_img, format="PNG")
    byte_img.seek(0)
    return byte_img.getvalue()
