import {
    prettifyEntryLabel,
    selectedEntryPreviewFallback,
    selectedEntryPreviewPath,
} from "./utils.js";

export function createSelectionController({
    entriesList,
    getPlanes,
    setFocusedSource = () => {},
    preview,
    updatePlaneStyle,
    updateTotals,
}) {
    const entries = new Set();
    const nodes = new Map();

    function addEntryToList(src) {
        if (nodes.has(src)) {
            return;
        }

        const li = document.createElement("li");
        li.dataset.src = src;
        li.addEventListener("mouseenter", () => setFocusedSource(src));
        li.addEventListener("mouseleave", () => setFocusedSource(null));
        li.addEventListener("focusin", () => setFocusedSource(src));
        li.addEventListener("focusout", (event) => {
            if (!li.contains(event.relatedTarget)) {
                setFocusedSource(null);
            }
        });

        const entryMain = document.createElement("div");
        entryMain.className = "entryMain";

        const previewImage = document.createElement("img");
        previewImage.className = "entryPreview";
        previewImage.alt = "Selected discovery preview";
        previewImage.loading = "lazy";
        previewImage.decoding = "async";

        const previewSrc = selectedEntryPreviewPath(src);
        const fallbackSrc = selectedEntryPreviewFallback(src);
        previewImage.src = previewSrc;
        if (fallbackSrc !== previewSrc) {
            previewImage.addEventListener("error", () => {
                previewImage.src = fallbackSrc;
            }, { once: true });
        }

        previewImage.addEventListener("mouseenter", () => preview.showForSource(src, null, previewImage));
        previewImage.addEventListener("focus", () => preview.showForSource(src, null, previewImage));
        previewImage.addEventListener("mouseleave", preview.hide);
        previewImage.addEventListener("blur", preview.hide);

        const span = document.createElement("span");
        span.className = "entryLabel";
        span.textContent = prettifyEntryLabel(src);
        span.title = span.textContent;

        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.textContent = "Remove";
        removeButton.addEventListener("click", () => unselect(src));

        entryMain.appendChild(previewImage);
        entryMain.appendChild(span);
        li.appendChild(entryMain);
        li.appendChild(removeButton);
        entriesList.appendChild(li);
        nodes.set(src, li);
    }

    function select(src) {
        if (entries.has(src)) {
            return;
        }
        entries.add(src);
        addEntryToList(src);
        updateTotals();
    }

    function unselect(src) {
        if (!entries.has(src)) {
            return;
        }

        entries.delete(src);
        const node = nodes.get(src);
        if (node) {
            if (node.dataset.src === src) {
                setFocusedSource(null);
            }
            node.remove();
            nodes.delete(src);
        }

        for (const plane of getPlanes()) {
            if (plane.userData.sourcePath === src) {
                plane.userData.selected = false;
                updatePlaneStyle(plane);
            }
        }
        updateTotals();
    }

    function togglePlane(plane) {
        const src = plane.userData.sourcePath;
        plane.userData.selected = !plane.userData.selected;
        if (plane.userData.selected) {
            select(src);
        } else {
            unselect(src);
        }
        updatePlaneStyle(plane);
    }

    function clear() {
        entries.clear();
        nodes.clear();
        setFocusedSource(null);
        entriesList.innerHTML = "";
        for (const plane of getPlanes()) {
            plane.userData.selected = false;
            updatePlaneStyle(plane);
        }
        updateTotals();
    }

    return {
        clear,
        has: (src) => entries.has(src),
        size: () => entries.size,
        togglePlane,
        unselect,
        values: () => Array.from(entries),
    };
}
