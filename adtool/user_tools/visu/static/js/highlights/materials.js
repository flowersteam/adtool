const TEXTURE_SIZE = 128;

export function createHighlightMaterialCache({ THREE, opacity }) {
    const cache = new Map();

    function drawTexture(colors) {
        const canvas = document.createElement("canvas");
        canvas.width = TEXTURE_SIZE;
        canvas.height = TEXTURE_SIZE;

        const context = canvas.getContext("2d");
        const center = TEXTURE_SIZE / 2;
        const radius = center;
        const step = (Math.PI * 2) / colors.length;

        for (let index = 0; index < colors.length; index += 1) {
            const start = -Math.PI / 2 + index * step;
            const end = start + step;

            context.beginPath();
            context.moveTo(center, center);
            context.arc(center, center, radius, start, end);
            context.closePath();
            context.fillStyle = colors[index];
            context.fill();
        }

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    function materialForColors(colors) {
        const key = colors.join("|");
        if (cache.has(key)) {
            return cache.get(key);
        }

        const texture = drawTexture(colors);
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            opacity,
            depthWrite: false,
        });
        cache.set(key, material);
        return material;
    }

    return {
        materialForColors,
    };
}
