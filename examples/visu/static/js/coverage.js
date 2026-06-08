import { COVERAGE_LOAD_GRACE_MS, LOAD_RETRY_MS } from "./config.js";
import { getCoverageStatus, getCoverageSummary, responseErrorMessage } from "./api.js";
import { formatNumber, formatRange, sleep } from "./utils.js";

export function createCoverageController({ elements, lightbox }) {
    let coverageRequestId = 0;
    let coverageEnabled = false;

    function setEmpty(visible, title = "No coverage run found", message = "") {
        elements.coverageEmpty.hidden = !visible;
        const heading = elements.coverageEmpty.querySelector("h3");
        const text = elements.coverageEmpty.querySelector("p");
        if (heading) {
            heading.textContent = title;
        }
        if (text) {
            text.textContent = message;
        }
    }

    async function initializeNavigation(showDiscoveriesPage) {
        try {
            const status = await getCoverageStatus();
            coverageEnabled = Boolean(status.enabled);
        } catch {
            coverageEnabled = false;
        }

        elements.coverageTab.hidden = !coverageEnabled;
        if (!coverageEnabled && elements.coveragePage.classList.contains("active")) {
            showDiscoveriesPage();
        }
        return coverageEnabled;
    }

    function statBox(value, label) {
        const node = document.createElement("div");
        node.className = "statBox";

        const valueNode = document.createElement("span");
        valueNode.className = "statValue";
        valueNode.textContent = value;

        const labelNode = document.createElement("span");
        labelNode.className = "statLabel";
        labelNode.textContent = label;

        node.appendChild(valueNode);
        node.appendChild(labelNode);
        return node;
    }

    function render(summary) {
        elements.coverageGrid.innerHTML = "";
        elements.coverageStats.innerHTML = "";
        setEmpty(false);

        elements.coverageSubtitle.textContent = summary.run_name
            ? `Showing ${summary.run_name}`
            : "Showing latest coverage run";

        const datasetALabel = summary.dataset_a_label || "first set";
        const datasetBLabel = summary.dataset_b_label || "second set";
        elements.coverageStats.appendChild(statBox(formatNumber(summary.dataset_a_count), datasetALabel));
        elements.coverageStats.appendChild(statBox(formatNumber(summary.dataset_b_count), datasetBLabel));
        elements.coverageStats.appendChild(statBox(formatNumber(summary.dim_count), "embedding dimensions"));
        elements.coverageStats.appendChild(statBox(formatNumber((summary.images || []).length), "graphs"));

        const images = Array.isArray(summary.images) ? summary.images : [];
        const labels = Array.isArray(summary.labels) ? summary.labels : [];
        const bounds = Array.isArray(summary.bounds) ? summary.bounds : [];

        for (const [index, image] of images.entries()) {
            const imageInfo = typeof image === "string"
                ? { file: image, url: `/coverage/${image}` }
                : image;
            const title = labels[index] || imageInfo.title || imageInfo.file || `Graph ${index + 1}`;
            const card = document.createElement("article");
            card.className = "coverageCard";

            const header = document.createElement("div");
            header.className = "coverageCardHeader";

            const titleNode = document.createElement("div");
            titleNode.className = "coverageCardTitle";
            titleNode.textContent = title;
            titleNode.title = title;

            const metaNode = document.createElement("div");
            metaNode.className = "coverageCardMeta";
            metaNode.textContent = formatRange(bounds[index]);

            const img = document.createElement("img");
            img.loading = "lazy";
            img.decoding = "async";
            img.alt = `Coverage graph for ${title}`;
            img.src = imageInfo.url || `/coverage/${imageInfo.file}`;

            const imageButton = document.createElement("button");
            imageButton.className = "coverageImageButton";
            imageButton.type = "button";
            imageButton.setAttribute("aria-label", `Open ${title} larger`);
            imageButton.addEventListener("click", () => {
                lightbox.open(img.src, title);
            });

            header.appendChild(titleNode);
            header.appendChild(metaNode);
            card.appendChild(header);
            imageButton.appendChild(img);
            card.appendChild(imageButton);
            elements.coverageGrid.appendChild(card);
        }

        if (images.length === 0) {
            setEmpty(
                true,
                "No coverage graphs found",
                "The coverage summary exists, but it does not list generated images.",
            );
        }
    }

    async function load() {
        const requestId = ++coverageRequestId;
        const deadline = performance.now() + COVERAGE_LOAD_GRACE_MS;
        let lastErrorMessage = "Coverage summary could not be loaded.";
        let fatalErrorMessage = "";

        elements.reloadCoverageButton.disabled = true;
        elements.coverageSubtitle.textContent = "Loading coverage summary...";
        elements.coverageGrid.innerHTML = "";
        elements.coverageStats.innerHTML = "";
        setEmpty(false);

        try {
            let summary = null;
            while (summary === null) {
                try {
                    const response = await getCoverageSummary();
                    if (response.ok) {
                        summary = await response.json();
                        break;
                    }

                    lastErrorMessage = await responseErrorMessage(response);
                    if (response.status === 422 || response.status === 400) {
                        fatalErrorMessage = lastErrorMessage;
                        throw new Error(fatalErrorMessage);
                    }
                } catch (error) {
                    if (fatalErrorMessage) {
                        throw error;
                    }
                    // Retry transient startup/load failures until the grace window expires.
                }

                if (performance.now() >= deadline) {
                    throw new Error("coverage summary unavailable");
                }
                if (requestId !== coverageRequestId) {
                    return;
                }
                await sleep(LOAD_RETRY_MS);
            }

            if (requestId !== coverageRequestId) {
                return;
            }
            render(summary);
        } catch {
            if (requestId !== coverageRequestId) {
                return;
            }
            elements.coverageGrid.innerHTML = "";
            elements.coverageStats.innerHTML = "";
            elements.coverageSubtitle.textContent = "Coverage could not be loaded.";
            setEmpty(true, "Coverage format issue", lastErrorMessage);
        } finally {
            if (requestId === coverageRequestId) {
                elements.reloadCoverageButton.disabled = false;
            }
        }
    }

    return {
        initializeNavigation,
        isEnabled: () => coverageEnabled,
        load,
    };
}
