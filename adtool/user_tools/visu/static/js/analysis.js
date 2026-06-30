import { getAnalysisRuns, getAnalysisStatus, responseErrorMessage } from "./api.js";
import { formatNumber, formatRange } from "./utils.js";

export function createAnalysisController({ elements, lightbox }) {
    let analysisRequestId = 0;
    let analysisEnabled = true;

    function summaryDatasets(summary) {
        return Array.isArray(summary.datasets) ? summary.datasets : [];
    }

    function setEmpty(visible, title = "No analysis run found", message = "") {
        elements.analysisEmpty.hidden = !visible;
        const heading = elements.analysisEmpty.querySelector("h3");
        const text = elements.analysisEmpty.querySelector("p");
        if (heading) {
            heading.textContent = title;
        }
        if (text) {
            text.textContent = message;
        }
    }

    async function initializeNavigation() {
        try {
            await getAnalysisStatus();
        } catch {
            // Navigation remains available; load() will show the actionable empty state.
        }

        setEnabled(true);
        return analysisEnabled;
    }

    function setEnabled(enabled) {
        analysisEnabled = Boolean(enabled);
        elements.analysisTab.hidden = !analysisEnabled;
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
        return summary.run_name || "analysis run";
    }

    function moduleEntries(summary) {
        const order = Array.isArray(summary.module_order) ? summary.module_order : [];
        const modules = summary.modules || {};
        return order
            .filter((moduleName) => modules[moduleName])
            .map((moduleName) => [moduleName, modules[moduleName]]);
    }

    function formatAnalysisBounds(bounds) {
        if (Array.isArray(bounds) && Array.isArray(bounds[0])) {
            const parts = [];
            const xBounds = formatRange(bounds[0]);
            const yBounds = formatRange(bounds[1]);
            if (xBounds) {
                parts.push(`X ${xBounds}`);
            }
            if (yBounds) {
                parts.push(`Y ${yBounds}`);
            }
            return parts.join(" | ");
        }
        return formatRange(bounds);
    }

    function renderImageCard(runName, imageInfo, fallbackTitle, fallbackBounds) {
        const title = imageInfo.title || fallbackTitle || imageInfo.file || "Graph";
        const bounds = imageInfo.bounds ?? fallbackBounds;
        const card = document.createElement("article");
        card.className = "analysisCard";

        const header = document.createElement("div");
        header.className = "analysisCardHeader";

        const titleNode = document.createElement("div");
        titleNode.className = "analysisCardTitle";
        titleNode.textContent = title;
        titleNode.title = title;

        const metaNode = document.createElement("div");
        metaNode.className = "analysisCardMeta";
        metaNode.textContent = formatAnalysisBounds(bounds);

        const img = document.createElement("img");
        img.loading = "lazy";
        img.decoding = "async";
        img.alt = `Analysis graph for ${title}`;
        img.src = imageInfo.url || `/analysis_files/${imageInfo.file}`;

        const imageButton = document.createElement("button");
        imageButton.className = "analysisImageButton";
        imageButton.type = "button";
        imageButton.setAttribute("aria-label", `Open ${title} larger`);
        imageButton.addEventListener("click", () => {
            lightbox.open(img.src, `${runName} / ${title}`);
        });

        header.appendChild(titleNode);
        header.appendChild(metaNode);
        card.appendChild(header);
        imageButton.appendChild(img);
        card.appendChild(imageButton);
        return card;
    }

    function moduleMeta(moduleSummary) {
        const summary = Array.isArray(moduleSummary.summary)
            ? moduleSummary.summary.filter((item) => typeof item === "string" && item)
            : [];
        if (summary.length > 0) {
            return summary.join(" | ");
        }
        return `${formatNumber((moduleSummary.images || []).length)} graphs`;
    }

    function renderModule(summary, moduleName, moduleSummary) {
        const section = document.createElement("section");
        section.className = "analysisRunSection";

        const header = document.createElement("div");
        header.className = "analysisRunHeader";

        const titleNode = document.createElement("h3");
        titleNode.textContent = moduleSummary.title || moduleName;
        titleNode.title = titleNode.textContent;

        const metaNode = document.createElement("div");
        metaNode.className = "analysisRunMeta";
        metaNode.textContent = moduleMeta(moduleSummary);

        const grid = document.createElement("div");
        grid.className = "analysisRunGrid";

        const images = Array.isArray(moduleSummary.images) ? moduleSummary.images : [];
        const labels = Array.isArray(moduleSummary.labels) ? moduleSummary.labels : [];
        const bounds = Array.isArray(moduleSummary.bounds) ? moduleSummary.bounds : [];
        for (const [index, image] of images.entries()) {
            grid.appendChild(
                renderImageCard(
                    runTitle(summary),
                    image,
                    labels[index] || `${moduleSummary.title || moduleName} ${index + 1}`,
                    bounds[index],
                ),
            );
        }

        if (images.length === 0) {
            const empty = document.createElement("div");
            empty.className = "emptyPanel compactEmptyPanel";
            empty.textContent = "This module has no generated graphs.";
            grid.appendChild(empty);
        }

        header.appendChild(titleNode);
        header.appendChild(metaNode);
        section.appendChild(header);
        section.appendChild(grid);
        return section;
    }

    function renderRun(summary) {
        const section = document.createElement("section");
        section.className = "analysisRunSection";

        const header = document.createElement("div");
        header.className = "analysisRunHeader";

        const titleNode = document.createElement("h3");
        titleNode.textContent = runTitle(summary);
        titleNode.title = titleNode.textContent;

        const metaNode = document.createElement("div");
        metaNode.className = "analysisRunMeta";
        const datasets = summaryDatasets(summary);
        const moduleCount = moduleEntries(summary).length;
        metaNode.textContent = [
            ...datasets.map((dataset) => `${dataset.label}: ${formatNumber(dataset.count || 0)}`),
            `${formatNumber(moduleCount)} modules`,
        ].join(" | ");

        const body = document.createElement("div");
        const modules = moduleEntries(summary);
        for (const [moduleName, moduleSummary] of modules) {
            body.appendChild(renderModule(summary, moduleName, moduleSummary));
        }

        if (modules.length === 0) {
            const empty = document.createElement("div");
            empty.className = "emptyPanel compactEmptyPanel";
            empty.textContent = "This run has no generated modules.";
            body.appendChild(empty);
        }

        header.appendChild(titleNode);
        header.appendChild(metaNode);
        section.appendChild(header);
        section.appendChild(body);
        return section;
    }

    function render(payload) {
        const runs = Array.isArray(payload.runs) ? payload.runs : [];
        elements.analysisGrid.innerHTML = "";
        elements.analysisStats.innerHTML = "";
        setEmpty(false);

        const graphCount = runs.reduce(
            (total, run) => total + moduleEntries(run).reduce(
                (moduleTotal, [, moduleSummary]) => moduleTotal + ((moduleSummary.images || []).length),
                0,
            ),
            0,
        );
        elements.analysisSubtitle.textContent = payload.analysis_runs_dir
            ? `Showing ${runs.length} analysis runs from ${payload.analysis_runs_dir}`
            : `Showing ${runs.length} analysis runs`;

        elements.analysisStats.appendChild(statBox(formatNumber(runs.length), "analysis runs"));
        elements.analysisStats.appendChild(statBox(formatNumber(graphCount), "graphs"));
        if (runs.length > 0) {
            const latest = runs[0];
            const datasets = summaryDatasets(latest);
            const totalDiscoveries = datasets.reduce(
                (count, dataset) => count + Number(dataset.count || 0),
                0,
            );
            elements.analysisStats.appendChild(statBox(formatNumber(datasets.length), "datasets"));
            elements.analysisStats.appendChild(statBox(formatNumber(totalDiscoveries), "discoveries"));
        }

        for (const summary of runs) {
            elements.analysisGrid.appendChild(renderRun(summary));
        }

        if (runs.length === 0) {
            setEmpty(
                true,
                "No analysis run found",
                "Run an analysis to create analysis_runs/analysis_run_* outputs.",
            );
        }
    }

    async function load() {
        const requestId = ++analysisRequestId;
        let lastErrorMessage = "Analysis runs could not be loaded.";
        let emptyTitle = "Analysis format issue";

        elements.reloadAnalysisButton.disabled = true;
        elements.analysisSubtitle.textContent = "Loading analysis runs...";
        elements.analysisGrid.innerHTML = "";
        elements.analysisStats.innerHTML = "";
        setEmpty(false);

        try {
            const response = await getAnalysisRuns();
            if (!response.ok) {
                lastErrorMessage = await responseErrorMessage(response, lastErrorMessage);
                if (response.status === 404) {
                    emptyTitle = "No analysis run found";
                }
                throw new Error(lastErrorMessage);
            }

            if (requestId !== analysisRequestId) {
                return;
            }
            render(await response.json());
        } catch {
            if (requestId !== analysisRequestId) {
                return;
            }
            elements.analysisGrid.innerHTML = "";
            elements.analysisStats.innerHTML = "";
            elements.analysisSubtitle.textContent = "Analysis could not be loaded.";
            setEmpty(true, emptyTitle, lastErrorMessage);
        } finally {
            if (requestId === analysisRequestId) {
                elements.reloadAnalysisButton.disabled = false;
            }
        }
    }

    return {
        initializeNavigation,
        isEnabled: () => analysisEnabled,
        load,
        setEnabled,
    };
}
