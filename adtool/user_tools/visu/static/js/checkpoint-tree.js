function formatNode(node) {
    const branch = node.name.match(/-branch-([^-]+)$/)?.[1] || node.name;
    const step = Number(node.step);
    return Number.isFinite(step) ? `${branch} - ${step} steps` : branch;
}

function svgElement(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
}

export function createCheckpointTree({ elements, onVisibilityChange, onHoverChange }) {
    let nodes = new Map();
    let children = new Map();
    let hidden = new Set();

    function descendants(name) {
        const result = [name];
        for (const child of children.get(name) || []) {
            result.push(...descendants(child));
        }
        return result;
    }

    function ancestors(name) {
        const result = [];
        let current = nodes.get(name);
        while (current?.parent) {
            result.push(current.parent);
            current = nodes.get(current.parent);
        }
        return result;
    }

    function layoutTree(roots) {
        let leafIndex = 0;
        let maxDepth = 0;
        const layout = new Map();
        const place = (node, depth) => {
            maxDepth = Math.max(maxDepth, depth);
            const childPositions = (children.get(node.name) || []).map(
                (name) => place(nodes.get(name), depth + 1),
            );
            const x = childPositions.length
                ? childPositions.reduce((sum, child) => sum + child.x, 0) / childPositions.length
                : leafIndex++;
            const position = { x, depth };
            layout.set(node.name, position);
            return position;
        };
        roots.forEach((root) => place(root, 0));
        const leaves = Math.max(leafIndex, 1);
        // The SVG viewBox preserves branch geometry while the canvas itself
        // always fits the dock width.
        const width = Math.max(360, leaves * 84 + 72);
        for (const position of layout.values()) {
            position.x = leaves === 1 ? width / 2 : 36 + position.x * (width - 72) / (leaves - 1);
        }
        return { layout, width, height: Math.max(140, (maxDepth + 1) * 76 + 28) };
    }

    function render() {
        elements.checkpointTree.replaceChildren();
        if (nodes.size === 0) {
            elements.checkpointTreeSection.hidden = true;
            return;
        }
        elements.checkpointTreeSection.hidden = false;
        const roots = [...nodes.values()].filter((node) => !node.parent || !nodes.has(node.parent));
        const { layout, width, height } = layoutTree(roots);
        const canvas = document.createElement("div");
        canvas.className = "checkpointTreeCanvas";
        canvas.style.width = "100%";
        canvas.style.height = `${height}px`;
        const lines = svgElement("svg", {
            viewBox: `0 0 ${width} ${height}`,
            preserveAspectRatio: "none",
        });
        for (const node of nodes.values()) {
            if (!node.parent || !layout.has(node.parent)) continue;
            const parent = layout.get(node.parent);
            const child = layout.get(node.name);
            lines.append(svgElement("path", {
                d: `M ${parent.x} ${parent.depth * 76 + 30} C ${parent.x} ${parent.depth * 76 + 62}, ${child.x} ${child.depth * 76 - 2}, ${child.x} ${child.depth * 76 + 30}`,
                class: "checkpointBranch",
            }));
        }
        canvas.append(lines);
        for (const node of nodes.values()) {
            const position = layout.get(node.name);
            const button = document.createElement("button");
            button.type = "button";
            button.className = "checkpointNode";
            button.setAttribute("aria-label", formatNode(node));
            button.style.left = `${position.x / width * 100}%`;
            button.style.top = `${position.depth * 76 + 30}px`;
            button.classList.toggle("hiddenCheckpoint", hidden.has(node.name));
            button.addEventListener("click", () => {
                const affected = descendants(node.name);
                if (hidden.has(node.name)) {
                    hidden.delete(node.name);
                    ancestors(node.name).forEach((name) => hidden.delete(name));
                } else {
                    affected.forEach((name) => hidden.add(name));
                }
                render();
                onVisibilityChange(new Set(hidden));
            });
            button.addEventListener("mouseenter", () => {
                elements.checkpointTreeHoverLabel.textContent = formatNode(node);
                button.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
                onHoverChange(node.name, new Set(ancestors(node.name)));
            });
            button.addEventListener("mouseleave", () => {
                elements.checkpointTreeHoverLabel.textContent = "";
                onHoverChange(null, new Set());
            });
            canvas.append(button);
        }
        elements.checkpointTree.append(canvas);
    }

    async function refresh() {
        try {
            const response = await fetch("/static/checkpoints.json", { cache: "no-store" });
            const payload = response.ok ? await response.json() : { nodes: [] };
            nodes = new Map((payload.nodes || []).map((node) => [node.name, node]));
            children = new Map();
            for (const node of nodes.values()) {
                if (node.parent && nodes.has(node.parent)) {
                    const siblings = children.get(node.parent) || [];
                    siblings.push(node.name);
                    children.set(node.parent, siblings);
                }
            }
            hidden = new Set([...hidden].filter((name) => nodes.has(name)));
            render();
        } catch {
            nodes = new Map();
            children = new Map();
            render();
        }
    }

    elements.checkpointTreeToggleButton.addEventListener("click", () => {
        const collapsed = !elements.checkpointTreeSection.classList.contains("collapsed");
        elements.checkpointTreeSection.classList.toggle("collapsed", collapsed);
        elements.checkpointTreeToggleButton.textContent = collapsed ? "Show" : "Hide";
    });
    return { refresh };
}
