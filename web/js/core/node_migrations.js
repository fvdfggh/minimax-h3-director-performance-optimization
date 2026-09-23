/** Keeping older Director node definitions working.
 *
 * A saved workflow pins the node definition it was created with, so this plugin
 * will always meet nodes whose outputs, widgets or names have changed since. These
 * helpers normalise a node's widget values and strip / migrate outputs that no
 * longer exist — a workflow older than the current schema must load with its links
 * intact rather than silently losing an edge.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { app } from "../../../scripts/app.js";

export function clearAllDirectorRunStatus() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        node._minimaxEditor?.clearRunProgress?.();
    }
}

/** Options a combo widget offers — ComfyUI keeps them on ``widget.options``. */
export function comboWidgetOptions(w) {
    const o = w?.options;
    if (!o) return null;
    if (Array.isArray(o)) return o;
    if (Array.isArray(o.values)) return o.values;
    if (Array.isArray(o.options)) return o.options;
    return null;
}

/**
 * Repair widget values that a stale workflow can no longer validate.
 *
 * * A widget added *after* the workflow was saved has no ``widgets_values``
 *   entry and lands as an empty string; the backend converts INT inputs with
 *   ``int(value)``, so an empty ``second_seed`` aborts the whole prompt before
 *   ``execute()`` is ever reached.
 * * A combo saved before its model existed keeps the frontend's
 *   ``(place models in: …)`` placeholder, which stops matching the option list
 *   as soon as the model is downloaded.
 *
 * Both are "this value would fail validation anyway" cases, so replacing it can
 * only turn a hard failure into a run.
 */
export function sanitizeWidgetValues(node) {
    for (const w of node.widgets || []) {
        if (!w || w.name == null) continue;
        const options = comboWidgetOptions(w);
        if (options && options.length) {
            if (w.value != null && !options.includes(w.value)) {
                // The frontend's empty-folder placeholder — never a real choice.
                const real = options.find(
                    (v) => typeof v !== "string" || !v.includes("place models in"),
                );
                if (real != null) {
                    console.warn(
                        `[MiniMax] ${node.type}.${w.name}: "${w.value}" 已不在选项列表中，改用 "${real}"。`,
                    );
                    w.value = real;
                }
            }
            continue;
        }
        if (w.value === "" && (w.type === "number" || w.type === "int" || w.type === "float")) {
            // 回退到控件自身的 INPUT_TYPES 默认值（如 second_denoise 的 1.0），
            // 而不是一律写 0 —— 否则新增的浮点控件在旧工作流里会被强制成 0.0。
            // 没有有效数字默认时才回退到 0（保持 second_seed 等整数控件的原行为）。
            const def = w.options?.default;
            const fallback =
                typeof def === "number" && Number.isFinite(def)
                    ? def
                    : (typeof w.defaultValue === "number" && Number.isFinite(w.defaultValue)
                        ? w.defaultValue
                        : 0);
            console.warn(`[MiniMax] ${node.type}.${w.name}: 数值为空，回退为 ${fallback}。`);
            w.value = fallback;
        }
    }
}

export function sanitizeAllWidgetValues() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        try {
            sanitizeWidgetValues(node);
        } catch (e) {
            /* best-effort */
        }
    }
}

/** Old workflows may still list removed output slots (e.g. segment_images). */
export function isMiniMaxH3DirectorOptNode(node) {
    const cls = node?.comfyClass || node?.type || "";
    return cls === "MiniMaxH3DirectorOpt" || cls === "ComfyMiniMaxH3DirectorOpt";
}

export function isDirectorNodeDef(nodeType, nodeData) {
    const cls = nodeType?.comfyClass || nodeData?.name || "";
    return cls === "MiniMaxH3DirectorOpt" || cls === "ComfyMiniMaxH3DirectorOpt";
}

export function stripDeprecatedDirectorOutputs(node) {
    if (!isMiniMaxH3DirectorOptNode(node) || !node.outputs?.length) return;
    const stale = new Set(["segment_images"]);
    for (let i = node.outputs.length - 1; i >= 0; i--) {
        if (stale.has(node.outputs[i]?.name)) {
            node.removeOutput(i);
        }
    }
}

/** Reorder legacy output links after slot layout changes. */
export function migrateDirectorOutputLinks(node) {
    if (!isMiniMaxH3DirectorOptNode(node)) return;
    const graph = app.graph ?? app.canvas?.graph;
    const links = graph?.links;
    if (!links?.length) return;
    const outputs = node.outputs || [];
    const byName = Object.fromEntries(
        outputs.map((o, i) => [o?.name, i]).filter(([n]) => !!n)
    );

    for (const link of links) {
        if (!link || String(link.origin_id) !== String(node.id)) continue;
        const target = graph.getNodeById?.(link.target_id);
        const input = target?.inputs?.[link.target_slot];
        const inputType = (input?.type || "").toUpperCase();

        // Old layouts had report at slot 1 or 3 as STRING.
        if (inputType === "STRING" && byName.report != null && link.origin_slot !== byName.report) {
            link.origin_slot = byName.report;
            continue;
        }
        // Old layouts had fps last (slot 5) as FLOAT.
        if (inputType === "FLOAT" && byName.fps != null && link.origin_slot !== byName.fps) {
            link.origin_slot = byName.fps;
            continue;
        }
        // Old layouts had frame_count at slot 2 as INT.
        if (inputType === "INT" && byName.frame_count != null && link.origin_slot !== byName.frame_count) {
            link.origin_slot = byName.frame_count;
        }
    }
}

export function normalizeDirectorOutputs(node) {
    stripDeprecatedDirectorOutputs(node);
    migrateDirectorOutputLinks(node);
}
