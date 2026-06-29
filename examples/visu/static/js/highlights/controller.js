import { readDiscoveryHighlights } from "./api.js";
import { highlightColor } from "./colors.js";
import { hasHiddenRuleMatch, matchedRuleColors } from "./matching.js";
import { renderHighlightRules } from "./panel.js";
import {
    createCustomHighlightRule,
    fieldSupportsCustomHighlightRule,
    mergeHighlightRules,
} from "./rules.js";
import { loadStoredHighlightRules, saveStoredHighlightRules } from "./storage.js";

export function createHighlightController({ elements, onRulesChange }) {
    let addRuleFieldId = "";
    let dockCollapsed = false;
    let fieldsById = new Map();
    let filtersDetected = false;
    let rules = [];
    let storageKey = "";

    function colorForIndex(index) {
        return highlightColor(index);
    }

    function visibleRules() {
        return rules.filter((rule) => !rule.deleted);
    }

    function persistRules() {
        saveStoredHighlightRules(storageKey, rules);
    }

    function render() {
        renderHighlightRules({
            addRuleFieldId,
            addableFields: [...fieldsById.values()].filter(fieldSupportsCustomHighlightRule),
            dockCollapsed,
            onDockToggle: toggleDock,
            onAddRule: addRule,
            onAddRuleFieldChange: setAddRuleFieldId,
            onRemoveRule: removeRule,
            onRestoreDefaultRules: restoreDefaultRules,
            onRuleClauseChange: updateRuleClause,
            section: elements.highlightRulesSection,
            list: elements.highlightRulesList,
            rules: visibleRules(),
            fieldsById,
            onRuleChange: updateRule,
            showRules: filtersDetected,
        });
    }

    function updateRule(ruleId, patch) {
        rules = rules.map((rule) => (
            rule.rule_id === ruleId
                ? { ...rule, ...patch }
                : rule
        ));
        persistRules();
        onRulesChange(patch);
    }

    function updateRuleClause(ruleId, clauseId, patch) {
        rules = rules.map((rule) => {
            if (rule.rule_id !== ruleId || !Array.isArray(rule.clauses)) {
                return rule;
            }
            return {
                ...rule,
                clauses: rule.clauses.map((clause) => (
                    clause.clause_id === clauseId
                        ? { ...clause, ...patch }
                        : clause
                )),
            };
        });
        persistRules();
        onRulesChange({ clauses: true });
    }

    function setAddRuleFieldId(fieldId) {
        addRuleFieldId = fieldId;
    }

    function toggleDock() {
        dockCollapsed = !dockCollapsed;
        render();
    }

    function addRule() {
        const field = fieldsById.get(addRuleFieldId);
        if (!field) {
            return;
        }
        rules = [
            ...rules,
            createCustomHighlightRule(field, colorForIndex(rules.length)),
        ];
        persistRules();
        render();
        onRulesChange({ rules: true });
    }

    function removeRule(ruleId) {
        rules = rules
            .map((rule) => (
                rule.rule_id === ruleId
                    ? { ...rule, deleted: true }
                    : rule
            ))
            .filter((rule) => !(rule.rule_kind === "custom" && rule.deleted));
        persistRules();
        render();
        onRulesChange({ rules: true });
    }

    function restoreDefaultRules() {
        rules = rules.map((rule) => (
            rule.rule_kind === "default"
                ? { ...rule, deleted: false }
                : rule
        ));
        persistRules();
        render();
        onRulesChange({ rules: true });
    }

    async function refreshSchema() {
        const schema = await readDiscoveryHighlights();
        const addableFields = schema.fields.filter(fieldSupportsCustomHighlightRule);
        const storedRules = loadStoredHighlightRules(schema.storage_key);

        storageKey = schema.storage_key;
        filtersDetected = Boolean(schema.filters_detected);
        fieldsById = new Map(schema.fields.map((field) => [field.field_id, field]));
        addRuleFieldId = fieldsById.has(addRuleFieldId)
            ? addRuleFieldId
            : (addableFields[0]?.field_id || "");
        rules = mergeHighlightRules(
            schema.rules,
            storedRules,
            fieldsById,
            colorForIndex,
        );

        persistRules();
        render();
        onRulesChange({});
    }

    function matchedColors(actualValues) {
        return matchedRuleColors(visibleRules(), actualValues);
    }

    function isVisible(actualValues) {
        return !hasHiddenRuleMatch(visibleRules(), actualValues);
    }

    return {
        isVisible,
        matchedColors,
        refreshSchema,
    };
}
