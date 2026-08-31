import { runAnalysis, runRandomRun } from "./api.js";

const ANALYSIS_INPUTS_STORAGE_KEY = "adtool.analysis.inputs.v2";
const LEGACY_ANALYSIS_INPUTS_STORAGE_KEY = "adtool.analysis.inputs.v1";

function compactFailureMessage(error, fallback) {
    const detail = error?.message || fallback;
    const summary = detail.split("\n", 1)[0];
    return `${summary} Full details were logged to the browser and server consoles.`;
}

function trimmedValue(element) {
    return element.value.trim();
}

export function createAnalysisActions({
    analysis,
    elements,
    showPage,
    updateStatus,
}) {
    function datasetRows() {
        return Array.from(elements.analysisDatasetList.querySelectorAll(".analysisComparisonRow"));
    }

    function datasetValues() {
        return datasetRows().map((row) => ({
            path: trimmedValue(row.querySelector(".analysisComparisonPathInput")),
            label: trimmedValue(row.querySelector(".analysisComparisonLabelInput")),
        }));
    }

    function datasetInputValues() {
        return datasetRows().map((row) => ({
            path: row.querySelector(".analysisComparisonPathInput").value,
            label: row.querySelector(".analysisComparisonLabelInput").value,
        }));
    }

    function saveInputs() {
        const state = {
            randomConfigPath: elements.randomConfigPath.value,
            randomIterations: elements.randomIterationsInput.value,
            randomSeed: elements.randomSeedInput.value,
            analysisConfigPath: elements.analysisConfigPath.value,
            datasets: datasetInputValues(),
        };
        try {
            window.localStorage.setItem(
                ANALYSIS_INPUTS_STORAGE_KEY,
                JSON.stringify(state),
            );
        } catch (error) {
            console.warn("Could not save analysis inputs in browser storage.", error);
        }
    }

    function storedInputs() {
        try {
            const serialized = window.localStorage.getItem(ANALYSIS_INPUTS_STORAGE_KEY)
                || window.localStorage.getItem(LEGACY_ANALYSIS_INPUTS_STORAGE_KEY);
            if (!serialized) {
                return null;
            }
            const state = JSON.parse(serialized);
            return state && typeof state === "object" ? state : null;
        } catch (error) {
            console.warn("Could not restore analysis inputs from browser storage.", error);
            return null;
        }
    }

    function createDatasetRow(path = "", label = "") {
        const row = document.createElement("div");
        row.className = "analysisComparisonRow";

        const pathField = document.createElement("label");
        pathField.className = "compactField wideField";
        const pathText = document.createElement("span");
        pathText.textContent = "Discoveries or checkpoint path";
        const pathInput = document.createElement("input");
        pathInput.type = "text";
        pathInput.autocomplete = "off";
        pathInput.placeholder = "/path/to/discoveries/or/checkpoint";
        pathInput.className = "pathInput analysisComparisonPathInput";
        pathInput.value = path;
        pathField.appendChild(pathText);
        pathField.appendChild(pathInput);

        const labelField = document.createElement("label");
        labelField.className = "compactField";
        const labelText = document.createElement("span");
        labelText.textContent = "Dataset label";
        const labelInput = document.createElement("input");
        labelInput.type = "text";
        labelInput.autocomplete = "off";
        labelInput.className = "analysisComparisonLabelInput";
        labelInput.value = label;
        labelField.appendChild(labelText);
        labelField.appendChild(labelInput);

        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.className = "ghostButton analysisComparisonRemoveButton";
        removeButton.textContent = "Remove";
        removeButton.addEventListener("click", () => {
            row.remove();
            saveInputs();
        });

        row.appendChild(pathField);
        row.appendChild(labelField);
        row.appendChild(removeButton);
        elements.analysisDatasetList.appendChild(row);
        return row;
    }

    function restoreInputs() {
        const state = storedInputs();
        if (!state) {
            return;
        }

        const scalarInputs = [
            [elements.randomConfigPath, state.randomConfigPath],
            [elements.randomIterationsInput, state.randomIterations],
            [elements.randomSeedInput, state.randomSeed],
            [elements.analysisConfigPath, state.analysisConfigPath],
        ];
        for (const [input, value] of scalarInputs) {
            if (typeof value === "string") {
                input.value = value;
            }
        }

        const datasets = Array.isArray(state.datasets)
            ? state.datasets
            : state.comparisons;
        if (!Array.isArray(datasets)) {
            return;
        }

        datasetRows().forEach((row) => row.remove());
        for (const dataset of datasets) {
            if (!dataset || typeof dataset !== "object") {
                continue;
            }
            createDatasetRow(
                typeof dataset.path === "string" ? dataset.path : "",
                typeof dataset.label === "string" ? dataset.label : "",
            );
        }
    }

    function initialize() {
        restoreInputs();
        const scalarInputs = [
            elements.randomConfigPath,
            elements.randomIterationsInput,
            elements.randomSeedInput,
            elements.analysisConfigPath,
        ];
        scalarInputs.forEach((input) => input.addEventListener("input", saveInputs));
        elements.analysisDatasetList.addEventListener("input", saveInputs);
        if (datasetRows().length === 0) {
            createDatasetRow();
        }
        saveInputs();
    }

    async function launchRandomRun() {
        const configPath = trimmedValue(elements.randomConfigPath);
        if (!configPath) {
            updateStatus("Random config path is required.");
            elements.randomConfigPath.focus();
            return;
        }

        elements.randomRunButton.disabled = true;
        updateStatus("Running random baseline...");
        try {
            const payload = await runRandomRun({
                config_file: configPath,
                nb_iterations: elements.randomIterationsInput.value,
                seed: elements.randomSeedInput.value,
            });
            const row = datasetRows()[0] || createDatasetRow();
            row.querySelector(".analysisComparisonPathInput").value = payload.discoveries_dir;
            saveInputs();
            updateStatus(`Random run complete: ${payload.discoveries_dir}`);
        } catch (error) {
            updateStatus(compactFailureMessage(error, "Random run failed."));
        } finally {
            elements.randomRunButton.disabled = false;
        }
    }

    async function launchAnalysis() {
        const datasets = datasetValues().filter((entry) => entry.path);
        if (datasets.length === 0) {
            updateStatus("At least one discoveries or checkpoint path is required.");
            datasetRows()[0]?.querySelector(".analysisComparisonPathInput")?.focus();
            return;
        }

        const configFile = trimmedValue(elements.analysisConfigPath);
        const resolvedConfigFile = configFile.toLowerCase() === "none" ? "" : configFile;

        elements.runAnalysisButton.disabled = true;
        elements.addAnalysisComparisonButton.disabled = true;
        elements.reloadAnalysisButton.disabled = true;
        updateStatus("Running analysis...");
        try {
            const payload = await runAnalysis({
                discovery_paths: datasets.map((entry) => entry.path),
                labels: datasets.map((entry) => entry.label),
                config_file: resolvedConfigFile || null,
            });
            analysis.setEnabled(true);
            updateStatus(`Analysis complete: ${payload.run_dir}`);
            showPage("analysis");
        } catch (error) {
            updateStatus(compactFailureMessage(error, "Analysis run failed."));
        } finally {
            elements.runAnalysisButton.disabled = false;
            elements.addAnalysisComparisonButton.disabled = false;
            elements.reloadAnalysisButton.disabled = false;
        }
    }

    function addDatasetRow() {
        createDatasetRow("", "");
        saveInputs();
    }

    function toggleAnalysisPanel() {
        const collapsed = elements.analysisPanelBody.hidden;
        elements.analysisPanelBody.hidden = !collapsed;
        elements.analysisPanelToggle.setAttribute("aria-expanded", String(collapsed));
        elements.analysisPanelToggle.textContent = collapsed ? "Hide" : "Show";
    }

    return {
        addDatasetRow,
        initialize,
        launchAnalysis,
        launchRandomRun,
        toggleAnalysisPanel,
    };
}
