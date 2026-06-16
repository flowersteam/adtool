import { runCoverageAnalysis, runRandomRun } from "./api.js";

function trimmedValue(element) {
    return element.value.trim();
}

export function createAnalysisActions({
    coverage,
    elements,
    showPage,
    updateStatus,
}) {
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
            elements.coverageComparePath.value = payload.discoveries_dir;
            updateStatus(`Random run complete: ${payload.discoveries_dir}`);
        } catch (error) {
            updateStatus(error.message || "Random run failed. Check server logs.");
        } finally {
            elements.randomRunButton.disabled = false;
        }
    }

    async function launchCoverageAnalysis() {
        const discoveryPath = trimmedValue(elements.coverageComparePath);
        if (!discoveryPath) {
            updateStatus("Discoveries path is required.");
            elements.coverageComparePath.focus();
            return;
        }

        const configFile = trimmedValue(elements.coverageConfigPath);
        const resolvedConfigFile = configFile.toLowerCase() === "none" ? "" : configFile;
        const labelA = trimmedValue(elements.coverageLabelA) || "IMGEP";
        const labelB = trimmedValue(elements.coverageLabelB) || "baseline";

        elements.coverageCompareButton.disabled = true;
        elements.reloadCoverageButton.disabled = true;
        updateStatus("Running coverage analysis...");
        try {
            const payload = await runCoverageAnalysis({
                path: discoveryPath,
                config_file: resolvedConfigFile || null,
                label_a: labelA,
                label_b: labelB,
            });
            coverage.setEnabled(true);
            updateStatus(`Coverage analysis complete: ${payload.run_dir}`);
            showPage("coverage");
        } catch (error) {
            updateStatus(error.message || "Coverage analysis failed. Check server logs.");
        } finally {
            elements.coverageCompareButton.disabled = false;
            elements.reloadCoverageButton.disabled = false;
        }
    }

    function toggleCoverageActions() {
        const collapsed = elements.coverageActionsBody.hidden;
        elements.coverageActionsBody.hidden = !collapsed;
        elements.coverageActionsToggle.setAttribute("aria-expanded", String(collapsed));
        elements.coverageActionsToggle.textContent = collapsed ? "Hide" : "Show";
    }

    return {
        launchCoverageAnalysis,
        launchRandomRun,
        toggleCoverageActions,
    };
}
