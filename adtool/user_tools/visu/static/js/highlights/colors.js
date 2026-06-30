const GOLDEN_ANGLE_DEGREES = 137.508;

export function highlightColor(index) {
    const hue = (index * GOLDEN_ANGLE_DEGREES) % 360;
    return `hsl(${hue.toFixed(1)} 84% 56%)`;
}
