function clampNumberToField(value, field) {
    let nextValue = Number(value);
    if (!Number.isFinite(nextValue)) {
        return null;
    }
    if (field.min !== undefined) {
        nextValue = Math.max(Number(field.min), nextValue);
    }
    if (field.max !== undefined) {
        nextValue = Math.min(Number(field.max), nextValue);
    }
    return nextValue;
}

function boundsLabel(field) {
    if (field.min === undefined || field.max === undefined) {
        return "";
    }
    return `${field.min} to ${field.max}`;
}

function numberInput(value, field, placeholder, onChange) {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.placeholder = placeholder;
    input.value = value ?? "";
    let lastCommittedValue = value ?? "";
    if (field.min !== undefined) {
        input.min = `${field.min}`;
    }
    if (field.max !== undefined) {
        input.max = `${field.max}`;
    }
    input.addEventListener("input", () => {
        if (input.value === "") {
            return;
        }
        const nextValue = clampNumberToField(input.value, field);
        if (nextValue === null) {
            return;
        }
        input.value = `${nextValue}`;
        lastCommittedValue = nextValue;
        onChange(nextValue);
    });
    input.addEventListener("blur", () => {
        if (input.value === "") {
            input.value = `${lastCommittedValue}`;
        }
    });
    return input;
}

function createRuleModeToggle(rule, onRuleChange) {
    const wrapper = document.createElement("div");
    wrapper.className = "highlightRuleModeToggle";

    for (const mode of ["off", "highlight", "hide"]) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "highlightDockButton";
        button.textContent = mode === "off"
            ? "Off"
            : mode === "highlight"
                ? "Highlight"
                : "Hide";
        button.classList.toggle("active", (rule.mode || "off") === mode);
        button.addEventListener("click", () => {
            onRuleChange(rule.rule_id, { mode });
            for (const sibling of wrapper.querySelectorAll("button")) {
                sibling.classList.toggle("active", sibling === button);
            }
        });
        wrapper.appendChild(button);
    }

    return wrapper;
}

function createClauseEditor(rule, field, clause, onRuleClauseChange) {
    const row = document.createElement("div");
    row.className = "highlightClauseRow";

    const lowerWrap = document.createElement("label");
    lowerWrap.className = "highlightBoundInput";
    const lowerSign = document.createElement("span");
    lowerSign.className = "highlightBoundSign";
    lowerSign.textContent = "≥";
    const lowerInput = numberInput(
        clause.lower,
        field,
        "Higher",
        (value) => onRuleClauseChange(rule.rule_id, clause.clause_id, { lower: value }),
    );
    lowerWrap.appendChild(lowerSign);
    lowerWrap.appendChild(lowerInput);

    const upperWrap = document.createElement("label");
    upperWrap.className = "highlightBoundInput";
    const upperSign = document.createElement("span");
    upperSign.className = "highlightBoundSign";
    upperSign.textContent = "≤";
    const upperInput = numberInput(
        clause.upper,
        field,
        "Lower",
        (value) => onRuleClauseChange(rule.rule_id, clause.clause_id, { upper: value }),
    );
    upperWrap.appendChild(upperSign);
    upperWrap.appendChild(upperInput);

    row.appendChild(lowerWrap);
    row.appendChild(upperWrap);

    return row;
}

function createAddRuleControls({
    addRuleFieldId,
    addableFields,
    onAddRule,
    onAddRuleFieldChange,
    onRestoreDefaultRules,
    showRules,
}) {
    const wrapper = document.createElement("div");
    wrapper.className = "highlightRuleBuilder";
    wrapper.hidden = !showRules || addableFields.length === 0;

    const select = document.createElement("select");
    for (const field of addableFields) {
        const option = document.createElement("option");
        option.value = field.field_id;
        option.textContent = field.label;
        option.selected = field.field_id === addRuleFieldId;
        select.appendChild(option);
    }
    select.addEventListener("change", () => {
        onAddRuleFieldChange(select.value);
    });

    const button = document.createElement("button");
    button.type = "button";
    button.className = "highlightDockButton";
    button.textContent = "Add Rule";
    button.addEventListener("click", onAddRule);

    wrapper.appendChild(select);
    wrapper.appendChild(button);

    const restoreButton = document.createElement("button");
    restoreButton.type = "button";
    restoreButton.className = "highlightDockButton highlightRuleBuilderRestore";
    restoreButton.textContent = "Restore Defaults";
    restoreButton.addEventListener("click", onRestoreDefaultRules);
    wrapper.appendChild(restoreButton);

    return wrapper;
}

function updateDockState(section, toggleButton, showShell, showRules, dockCollapsed) {
    section.hidden = !showShell;
    section.classList.toggle("collapsed", dockCollapsed);
    if (toggleButton) {
        toggleButton.textContent = dockCollapsed ? "Show" : "Hide";
        toggleButton.hidden = !showRules;
    }
}

export function renderHighlightRules({
    addRuleFieldId,
    addableFields,
    dockCollapsed,
    fieldsById,
    list,
    onAddRule,
    onAddRuleFieldChange,
    onDockToggle,
    onRemoveRule,
    onRestoreDefaultRules,
    onRuleChange,
    onRuleClauseChange,
    rules,
    section,
    showRules,
}) {
    const toggleButton = section.querySelector("#highlightRulesToggleButton");
    if (toggleButton) {
        toggleButton.onclick = onDockToggle;
    }

    updateDockState(
        section,
        toggleButton,
        rules.length > 0 || addableFields.length > 0,
        showRules,
        dockCollapsed,
    );
    list.hidden = dockCollapsed || !showRules;
    list.innerHTML = "";

    const builder = createAddRuleControls({
        addRuleFieldId,
        addableFields,
        onAddRule,
        onAddRuleFieldChange,
        onRestoreDefaultRules,
        showRules: showRules && !dockCollapsed,
    });
    list.appendChild(builder);

    for (const rule of rules) {
        const field = fieldsById.get(rule.field_id);
        if (!field) {
            continue;
        }

        const card = document.createElement("div");
        card.className = "highlightRuleCard";

        const header = document.createElement("div");
        header.className = "highlightRuleHeader";

        const title = document.createElement("span");
        title.className = "highlightRuleTitle";
        title.textContent = rule.label;

        const headerActions = document.createElement("div");
        headerActions.className = "highlightRuleHeaderActions";

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "highlightDockButton highlightRuleDeleteButton";
        deleteButton.textContent = "Delete";
        deleteButton.addEventListener("click", () => {
            onRemoveRule(rule.rule_id);
        });

        const swatch = document.createElement("button");
        swatch.type = "button";
        swatch.className = "highlightColorSwatch";
        swatch.style.background = rule.color;

        const colorInput = document.createElement("input");
        colorInput.type = "color";
        colorInput.className = "highlightColorInput";
        colorInput.value = rule.color;
        colorInput.addEventListener("input", () => {
            swatch.style.background = colorInput.value;
            onRuleChange(rule.rule_id, { color: colorInput.value });
        });
        colorInput.addEventListener("change", () => {
            onRuleChange(rule.rule_id, { color: colorInput.value });
        });
        swatch.addEventListener("click", () => {
            const rect = swatch.getBoundingClientRect();
            const pickerWidth = 240;
            const left = Math.max(
                12,
                Math.min(window.innerWidth - pickerWidth - 12, rect.left + (rect.width / 2) - (pickerWidth / 2)),
            );
            const top = Math.max(
                12,
                Math.min(window.innerHeight - 40, rect.bottom + 8),
            );
            colorInput.style.left = `${left}px`;
            colorInput.style.top = `${top}px`;
            colorInput.click();
        });

        headerActions.appendChild(deleteButton);
        headerActions.appendChild(swatch);
        headerActions.appendChild(colorInput);

        header.appendChild(title);
        header.appendChild(headerActions);

        const controls = document.createElement("div");
        controls.className = "highlightRuleControls";
        controls.appendChild(createRuleModeToggle(rule, onRuleChange));
        controls.appendChild(createClauseEditor(rule, field, rule.clauses[0], onRuleClauseChange));

        if (field.value_type === "number") {
            const bounds = document.createElement("span");
            bounds.className = "highlightRuleBounds";
            bounds.textContent = boundsLabel(field);
            controls.appendChild(bounds);
        }

        card.appendChild(header);
        card.appendChild(controls);
        list.appendChild(card);
    }
}
