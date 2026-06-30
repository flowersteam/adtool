function nextId(prefix) {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export function nextRuleId() {
    return nextId("custom_rule");
}

function clampBound(value, field) {
    if (value === null || value === undefined || value === "") {
        return null;
    }

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

function normalizedClause(clause, field) {
    const lower = clampBound(clause.lower, field);
    const upper = clampBound(clause.upper, field);
    return {
        clause_id: clause.clause_id || nextId("clause"),
        lower: lower ?? field.min ?? upper ?? 0,
        upper: upper ?? field.max ?? lower ?? 0,
    };
}

function ruleClauses(rule, field) {
    return [normalizedClause(rule.clauses[0], field)];
}

function normalizedLabel(rule, field) {
    return rule.rule_kind === "custom" ? field.label : (rule.label || field.label);
}

export function normalizeHighlightRule(rule, previousRule, field, color) {
    return {
        ...rule,
        color: previousRule?.color || rule.color || color,
        deleted: Boolean(previousRule?.deleted || rule.deleted),
        field_id: field.field_id,
        label: normalizedLabel(rule, field),
        mode: previousRule?.mode || rule.mode || (rule.enabled_by_default ? "highlight" : "off"),
        rule_kind: rule.rule_kind || "default",
        clauses: ruleClauses(previousRule?.clauses ? previousRule : rule, field),
    };
}

export function mergeHighlightRules(schemaRules, storedRules, fieldsById, colorForIndex) {
    const storedDefaultsById = new Map(
        storedRules
            .filter((rule) => rule.rule_kind !== "custom")
            .map((rule) => [rule.rule_id, rule]),
    );
    const customRules = storedRules.filter((rule) => rule.rule_kind === "custom");

    const mergedDefaults = schemaRules
        .filter((rule) => fieldsById.has(rule.field_id))
        .map((rule, index) => (
            normalizeHighlightRule(
                { ...rule, rule_kind: "default" },
                storedDefaultsById.get(rule.rule_id),
                fieldsById.get(rule.field_id),
                colorForIndex(index),
            )
        ));

    const mergedCustom = customRules
        .filter((rule) => fieldsById.has(rule.field_id))
        .map((rule, index) => (
            normalizeHighlightRule(
                rule,
                rule,
                fieldsById.get(rule.field_id),
                rule.color || colorForIndex(mergedDefaults.length + index),
            )
        ));

    return [...mergedDefaults, ...mergedCustom];
}

export function fieldSupportsCustomHighlightRule(field) {
    return field.value_type === "number";
}

export function createCustomHighlightRule(field, color) {
    return {
        clauses: [{
            clause_id: nextId("clause"),
            lower: field.min ?? 0,
            upper: field.max ?? 0,
        }],
        color,
        field_id: field.field_id,
        label: field.label,
        mode: "off",
        rule_id: nextRuleId(),
        rule_kind: "custom",
    };
}
