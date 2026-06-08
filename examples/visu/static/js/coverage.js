import { getCoverageRuns, getCoverageStatus, responseErrorMessage } from "./api.js";
import { formatNumber, formatRange } from "./utils.js";

export function createCoverageController({ elements, lightbox }) {
    let coverageRequestId = 0;
    let coverageEnabled = true;

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

    async function initializeNavigation() {
        try {
            await getCoverageStatus();
        } catch {
            // Navigation remains available; load() will show the actionable empty state.
        }

        setEnabled(true);
        return coverageEnabled;
    }

    function setEnabled(enabled) {
        coverageEnabled = Boolean(enabled);
        elements.coverageTab.hidden = !coverageEnabled;
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

    function runTitle(summary) {
        return summary.run_name || "coverage run";
    }

    function renderImageCard(summary, image, index) {
        const imageInfo = typeof image === "string"
            ? { file: image, url: `/coverage/${image}` }
            : image;
        const labels = Array.isArray(summary.labels) ? summary.labels : [];
        const bounds = Array.isArray(summary.bounds) ? summary.bounds : [];
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
            lightbox.open(img.src, `${runTitle(summary)} / ${title}`);
        });

        header.appendChild(titleNode);
        header.appendChild(metaNode);
        card.appendChild(header);
        imageButton.appendChild(img);
        card.appendChild(imageButton);
        return card;
    }

    function renderRun(summary) {
        const section = document.createElement("section");
        section.className = "coverageRunSection";

        const header = document.createElement("div");
        header.className = "coverageRunHeader";

        const titleNode = document.createElement("h3");
        titleNode.textContent = runTitle(summary);
        titleNode.title = titleNode.textContent;

        const metaNode = document.createElement("div");
        metaNode.className = "coverageRunMeta";
        const datasetALabel = summary.dataset_a_label || "first set";
        const datasetBLabel = summary.dataset_b_label || "second set";
        metaNode.textContent = [
            `${datasetALabel}: ${formatNumber(summary.dataset_a_count)}`,
            `${datasetBLabel}: ${formatNumber(summary.dataset_b_count)}`,
            `${formatNumber(summary.dim_count)} dims`,
        ].join(" | ");

        const grid = document.createElement("div");
        grid.className = "coverageRunGrid";

        const images = Array.isArray(summary.images) ? summary.images : [];
        for (const [index, image] of images.entries()) {
            grid.appendChild(renderImageCard(summary, image, index));
        }

        if (images.length === 0) {
            const empty = document.createElement("div");
            empty.className = "emptyPanel compactEmptyPanel";
            empty.textContent = "This run has no generated graphs.";
            grid.appendChild(empty);
        }

        header.appendChild(titleNode);
        header.appendChild(metaNode);
        section.appendChild(header);
        section.appendChild(grid);
        return section;
    }

    function render(payload) {
        const runs = Array.isArray(payload.runs) ? payload.runs : [];
        elements.coverageGrid.innerHTML = "";
        elements.coverageStats.innerHTML = "";
        setEmpty(false);

        const graphCount = runs.reduce(
            (total, run) => total + (Array.isArray(run.images) ? run.images.length : 0),
            0,
        );
        elements.coverageSubtitle.textContent = payload.coverage_runs_dir
            ? `Showing ${runs.length} coverage runs from ${payload.coverage_runs_dir}`
            : `Showing ${runs.length} coverage runs`;

        elements.coverageStats.appendChild(statBox(formatNumber(runs.length), "coverage runs"));
        elements.coverageStats.appendChild(statBox(formatNumber(graphCount), "graphs"));
        if (runs.length > 0) {
            const latest = runs[0];
            elements.coverageStats.appendChild(statBox(formatNumber(latest.dataset_a_count), latest.dataset_a_label || "first set"));
            elements.coverageStats.appendChild(statBox(formatNumber(latest.dataset_b_count), latest.dataset_b_label || "second set"));
        }

        for (const summary of runs) {
            elements.coverageGrid.appendChild(renderRun(summary));
        }

        if (runs.length === 0) {
            setEmpty(
                true,
                "No coverage run found",
                "Run a coverage comparison to create coverage_runs/coverage_run_* outputs.",
            );
        }
    }

    async function load() {
        const requestId = ++coverageRequestId;
        let lastErrorMessage = "Coverage runs could not be loaded.";
        let emptyTitle = "Coverage format issue";

        elements.reloadCoverageButton.disabled = true;
        elements.coverageSubtitle.textContent = "Loading coverage runs...";
        elements.coverageGrid.innerHTML = "";
        elements.coverageStats.innerHTML = "";
        setEmpty(false);

        try {
            const response = await getCoverageRuns();
            if (!response.ok) {
                lastErrorMessage = await responseErrorMessage(response, lastErrorMessage);
                if (response.status === 404) {
                    emptyTitle = "No coverage run found";
                }
                throw new Error(lastErrorMessage);
            }

            if (requestId !== coverageRequestId) {
                return;
            }
            render(await response.json());
        } catch {
            if (requestId !== coverageRequestId) {
                return;
            }
            elements.coverageGrid.innerHTML = "";
            elements.coverageStats.innerHTML = "";
            elements.coverageSubtitle.textContent = "Coverage could not be loaded.";
            setEmpty(true, emptyTitle, lastErrorMessage);
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
        setEnabled,
    };
}
