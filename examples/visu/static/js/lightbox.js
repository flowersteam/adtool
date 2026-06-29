export function createGraphLightbox(elements) {
    function open(src, title) {
        elements.graphLightboxTitle.textContent = title;
        elements.graphLightboxImage.src = src;
        elements.graphLightboxImage.alt = `Expanded analysis graph for ${title}`;
        elements.graphLightbox.hidden = false;
    }

    function close() {
        elements.graphLightbox.hidden = true;
        elements.graphLightboxImage.removeAttribute("src");
    }

    return { open, close };
}
