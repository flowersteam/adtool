import { clamp, isVideoPath, prettifyEntryLabel } from "./utils.js";

export function createPreviewController(elements) {
    let currentPreviewSource = null;

    function applyScale(value) {
        const scalePercent = clamp(Number(value) || 100, 70, 250);
        const width = Math.round(430 * (scalePercent / 100));
        const height = Math.round(270 * (scalePercent / 100));
        document.documentElement.style.setProperty("--preview-width", `${width}px`);
        document.documentElement.style.setProperty("--preview-height", `${height}px`);
        elements.previewSizeSlider.value = `${scalePercent}`;
        elements.previewSizeValue.textContent = `${scalePercent}%`;
    }

    function hide() {
        elements.previewCard.style.display = "none";
        currentPreviewSource = null;
        elements.previewVideo.pause();
        elements.previewVideo.removeAttribute("src");
        elements.previewVideo.style.display = "none";
        elements.previewImage.removeAttribute("src");
        elements.previewImage.style.display = "none";
    }

    function placeAt(x, y) {
        const margin = 14;
        const rect = elements.previewCard.getBoundingClientRect();
        const width = Math.min(rect.width || 430, window.innerWidth - margin * 2);
        const height = Math.min(rect.height || 310, window.innerHeight - margin * 2);
        const maxLeft = Math.max(margin, window.innerWidth - width - margin);
        const maxTop = Math.max(margin, window.innerHeight - height - margin);
        const left = clamp(x + 18, margin, maxLeft);
        const top = clamp(y + 14, margin, maxTop);
        elements.previewCard.style.left = `${left}px`;
        elements.previewCard.style.top = `${top}px`;
    }

    function placeNearElement(element) {
        const rect = element.getBoundingClientRect();
        placeAt(rect.right, rect.top);
    }

    function schedulePlacement(pointerEvent = null, anchorElement = null) {
        requestAnimationFrame(() => {
            if (elements.previewCard.style.display === "none") {
                return;
            }

            if (pointerEvent) {
                placeAt(pointerEvent.clientX, pointerEvent.clientY);
            } else if (anchorElement) {
                placeNearElement(anchorElement);
            }
        });
    }

    function showForSource(src, pointerEvent = null, anchorElement = null) {
        elements.previewMeta.textContent = prettifyEntryLabel(src);
        elements.previewCard.style.display = "block";

        if (currentPreviewSource !== src) {
            currentPreviewSource = src;
            elements.previewVideo.pause();
            elements.previewVideo.removeAttribute("src");
            elements.previewVideo.style.display = "none";
            elements.previewImage.removeAttribute("src");
            elements.previewImage.style.display = "none";

            if (isVideoPath(src)) {
                elements.previewVideo.src = src;
                elements.previewVideo.style.display = "block";
                elements.previewVideo.currentTime = 0;
                elements.previewVideo.play().catch(() => {});
            } else {
                elements.previewImage.src = src;
                elements.previewImage.style.display = "block";
            }
        }

        schedulePlacement(pointerEvent, anchorElement);
    }

    function showForPlane(plane, event) {
        showForSource(plane.userData.sourcePath, event, null);
    }

    return {
        applyScale,
        hide,
        showForSource,
        showForPlane,
    };
}
