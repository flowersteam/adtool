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
        const discoveryPath = trimmedValue(elements.analysisTargetPath);
        if (!discoveryPath) {
            updateStatus("Discoveries path is required.");
            elements.analysisTargetPath.focus();
            return;
        }

        const configFile = trimmedValue(elements.analysisConfigPath);
        const resolvedConfigFile = configFile.toLowerCase() === "none" ? "" : configFile;
        const labelA = trimmedValue(elements.analysisLabelA) || "IMGEP";
        const labelB = trimmedValue(elements.analysisLabelB) || "baseline";

        elements.runAnalysisButton.disabled = true;
        elements.reloadAnalysisButton.disabled = true;
        updateStatus("Running analysis...");
        try {
            const payload = await runAnalysis({
                path: discoveryPath,
                config_file: resolvedConfigFile || null,
                label_a: labelA,
                label_b: labelB,
            });
            analysis.setEnabled(true);
            updateStatus(`Analysis complete: ${payload.run_dir}`);
            showPage("analysis");
        } catch (error) {
            updateStatus(error.message || "Analysis run failed. Check server logs.");
        } finally {
            elements.runAnalysisButton.disabled = false;
            elements.reloadAnalysisButton.disabled = false;
        }
    }

    function toggleAnalysisPanel() {
        const collapsed = elements.analysisPanelBody.hidden;
        elements.analysisPanelBody.hidden = !collapsed;
        elements.analysisPanelToggle.setAttribute("aria-expanded", String(collapsed));
        elements.analysisPanelToggle.textContent = collapsed ? "Hide" : "Show";
    }

    return {
        launchAnalysis,
        launchRandomRun,
        toggleAnalysisPanel,
    };
}
