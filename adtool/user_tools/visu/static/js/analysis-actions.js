import { runAnalysis, runRandomRun } from "./api.js";

function trimmedValue(element) {
    return element.value.trim();
}

export function createAnalysisActions({
    analysis,
    elements,
    showPage,
    updateStatus,
}) {
    function comparisonRows() {
        return Array.from(elements.analysisComparisonList.querySelectorAll(".analysisComparisonRow"));
    }

    function comparisonValues() {
        return comparisonRows().map((row) => ({
            path: trimmedValue(row.querySelector(".analysisComparisonPathInput")),
            label: trimmedValue(row.querySelector(".analysisComparisonLabelInput")),
        }));
    }

    function createComparisonRow(path = "", label = "") {
        const row = document.createElement("div");
        row.className = "analysisComparisonRow";

        const pathField = document.createElement("label");
        pathField.className = "compactField wideField";
        const pathText = document.createElement("span");
        pathText.textContent = "Comparison discoveries path";
        const pathInput = document.createElement("input");
        pathInput.type = "text";
        pathInput.autocomplete = "off";
        pathInput.placeholder = "/path/to/discoveries";
        pathInput.className = "pathInput analysisComparisonPathInput";
        pathInput.value = path;
        pathField.appendChild(pathText);
        pathField.appendChild(pathInput);

        const labelField = document.createElement("label");
        labelField.className = "compactField";
        const labelText = document.createElement("span");
        labelText.textContent = "Comparison label";
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
        });

        row.appendChild(pathField);
        row.appendChild(labelField);
        row.appendChild(removeButton);
        elements.analysisComparisonList.appendChild(row);
        return row;
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
            elements.analysisTargetPath.value = payload.discoveries_dir;
            updateStatus(`Random run complete: ${payload.discoveries_dir}`);
        } catch (error) {
            updateStatus(error.message || "Random run failed. Check server logs.");
        } finally {
            elements.randomRunButton.disabled = false;
        }
    }

    async function launchAnalysis() {
        const comparisons = comparisonValues().filter((entry) => entry.path);
        if (comparisons.length === 0) {
            updateStatus("At least one comparison discoveries path is required.");
            elements.analysisTargetPath.focus();
            return;
        }

        const configFile = trimmedValue(elements.analysisConfigPath);
        const resolvedConfigFile = configFile.toLowerCase() === "none" ? "" : configFile;
        const primaryLabel = trimmedValue(elements.analysisLabelA) || "IMGEP";

        elements.runAnalysisButton.disabled = true;
        elements.addAnalysisComparisonButton.disabled = true;
        elements.reloadAnalysisButton.disabled = true;
        updateStatus("Running analysis...");
        try {
            const payload = await runAnalysis({
                comparison_paths: comparisons.map((entry) => entry.path),
                comparison_labels: comparisons.map((entry) => entry.label),
                config_file: resolvedConfigFile || null,
                primary_label: primaryLabel,
            });
            analysis.setEnabled(true);
            updateStatus(`Analysis complete: ${payload.run_dir}`);
            showPage("analysis");
        } catch (error) {
            updateStatus(error.message || "Analysis run failed. Check server logs.");
        } finally {
            elements.runAnalysisButton.disabled = false;
            elements.addAnalysisComparisonButton.disabled = false;
            elements.reloadAnalysisButton.disabled = false;
        }
    }

    function addComparisonRow() {
        createComparisonRow("", "");
    }

    function toggleAnalysisPanel() {
        const collapsed = elements.analysisPanelBody.hidden;
        elements.analysisPanelBody.hidden = !collapsed;
        elements.analysisPanelToggle.setAttribute("aria-expanded", String(collapsed));
        elements.analysisPanelToggle.textContent = collapsed ? "Hide" : "Show";
    }

    return {
        addComparisonRow,
        launchAnalysis,
        launchRandomRun,
        toggleAnalysisPanel,
    };
}
