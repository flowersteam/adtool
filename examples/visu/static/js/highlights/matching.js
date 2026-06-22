function toNumber(value) {
    return Number(value);
}

function ruleMode(rule) {
    if (rule.mode) {
        return rule.mode;
    }
    return rule.enabled ? "highlight" : "off";
}

function compareLower(actualValue, lower) {
    if (lower === null || lower === undefined) {
        return true;
    }
    return toNumber(actualValue) >= toNumber(lower);
}

function compareUpper(actualValue, upper) {
    if (upper === null || upper === undefined) {
        return true;
    }
    return toNumber(actualValue) <= toNumber(upper);
}

function clauseMatches(clause, actualValue) {
    if (clause.lower === null && clause.upper === null) {
        return false;
    }
    if (
        clause.lower !== null
        && clause.upper !== null
        && toNumber(clause.lower) > toNumber(clause.upper)
    ) {
        return false;
    }
    return (
        compareLower(actualValue, clause.lower)
        && compareUpper(actualValue, clause.upper)
    );
}

export function ruleMatches(rule, actualValues) {
    const actualValue = actualValues?.[rule.field_id];
    return Array.isArray(rule.clauses)
        && rule.clauses.some((clause) => clauseMatches(clause, actualValue));
}

export function matchedRuleColors(rules, actualValues) {
    return rules
        .filter((rule) => (
            ruleMode(rule) === "highlight"
            && ruleMatches(rule, actualValues)
        ))
        .map((rule) => rule.color);
}

export function hasHiddenRuleMatch(rules, actualValues) {
    return rules.some((rule) => (
        ruleMode(rule) === "hide"
        && ruleMatches(rule, actualValues)
    ));
}
