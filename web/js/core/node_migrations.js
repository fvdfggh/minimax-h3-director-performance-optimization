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
import { tAnyLocale } from "../minimax_i18n.js";
import { DIRECTOR_GROUP_LABEL_KEYS } from "./widget_labels.js";

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
        const numeric = w.type === "number" || w.type === "int" || w.type === "float";
        const boolish = w.type === "toggle" || w.type === "boolean" || w.type === "bool";
        if (!numeric && !boolish) continue;
        // 空串 / undefined 才会走到这里；字符串控件里空串是合法值，不动。
        if (w.value !== "" && w.value != null) continue;
        // 回退顺序：**工作流里的值** → 节点创建时抓的 INPUT_TYPES 默认值 → 控件自带 default。
        // 以前只有后两者，都拿不到时写 0 —— 二采固定种子 second_seed 的「莫名变成 0」
        // 就是这么来的：文件里明明是 20240/1，控件却是空的，于是被兜底成 0。
        const loaded = node?.__mmxLoadedValues?.[w.name];
        const captured = node?.__mmxWidgetDefaults?.[w.name];
        const asNumber = (v) => (typeof v === "string" && v.trim() !== "" ? Number(v) : v);
        if (numeric) {
            const fallback =
                [loaded, captured, w.options?.default, w.defaultValue]
                    .map(asNumber)
                    .find((v) => typeof v === "number" && Number.isFinite(v))
                ?? 0;   // 实在没有来源才写 0：总比让后端 int("") 报错好
            const source = fallback === asNumber(loaded)
                ? "（工作流里的值）"
                : fallback === asNumber(captured) ? "（INPUT_TYPES 默认值）" : "（无来源，兜底 0）";
            console.warn(
                `[MiniMax] ${node.type}.${w.name}: 数值为空，回退为 ${fallback}${source}。`,
            );
            w.value = fallback;
        } else {
            const boolFallback = typeof loaded === "boolean" ? loaded
                : (typeof captured === "boolean" ? captured : null);
            if (boolFallback != null) {
                console.warn(
                    `[MiniMax] ${node.type}.${w.name}: 布尔为空，回退为 ${boolFallback}`
                    + `${boolFallback === loaded ? "（工作流里的值）" : "（INPUT_TYPES 默认值）"}。`,
                );
                w.value = boolFallback;
            }
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

/**
 * Widgets deleted from the Director node, keyed by the widget that *followed*
 * them in the old INPUT_TYPES order.
 *
 * A saved workflow pins ``widgets_values`` positionally, and ComfyUI hands that
 * array to the widgets in order when the node is configured (skipping
 * ``serialize === false``). Deleting a widget therefore shifts every later value
 * by one — an old workflow would put ``second_denoise``'s 1.0 into
 * ``second_seed`` and 20240 into ``asr_check`` (turning the ASR check on). Each
 * entry names the successor, whose *new* index is exactly the slot the removed
 * widget used to occupy.
 */
const REMOVED_DIRECTOR_WIDGETS = [
    { name: "second_denoise", followedBy: "second_seed" },
    // 「段间锥形重绘」开关（conn_noise）也已移除，行为恒开；它是「二级采样」组
    // 标题的前一个控件，所以旧值占的是 bd_grp_second 现在的下标。
    { name: "conn_noise", followedBy: "bd_grp_second" },
];

/** Widgets that take part in the positional ``widgets_values`` array. */
function serializableWidgets(node) {
    return (node?.widgets || []).filter((w) => w && w.serialize !== false);
}

/**
 * Snapshot each widget's INPUT_TYPES default.
 *
 * ``configure`` is the only moment where the node still holds its schema defaults:
 * ComfyUI overwrites ``widgets[i].value`` from the saved array right after this
 * wrapper returns. Capturing them is what lets :func:`sanitizeWidgetValues` put a
 * *real* default back — a plain INT/BOOLEAN widget carries no ``options.default``,
 * so the old code fell back to 0 and wrote it into ``second_seed`` on every load.
 */
function captureWidgetDefaults(node) {
    if (!node || node.__mmxWidgetDefaults) return;
    const out = {};
    for (const w of serializableWidgets(node)) {
        if (w?.name && (typeof w.value === "number" || typeof w.value === "boolean")) {
            out[w.name] = w.value;
        }
    }
    node.__mmxWidgetDefaults = out;
}

/**
 * Can ``value`` legally live in widget ``w``?
 *
 * A stale positional array leaves values on the wrong widgets — a BOOLEAN ends up
 * holding「主模型 (model)」, an INT holding a group label — and once such a file is
 * saved the mismatch is preserved *by name*, so it survives every reload. Only
 * types whose value space is unambiguous are checked: combo values are localized
 * labels that change with the UI language, so validating them would reset a
 * perfectly good value after a locale switch.
 */
function acceptsWidgetValue(w, value) {
    const type = String(w?.type || "").toLowerCase();
    if (type === "toggle" || type === "boolean" || type === "bool") {
        return typeof value === "boolean" || value === 0 || value === 1;
    }
    if (type === "number" || type === "slider") {
        if (typeof value === "number") return Number.isFinite(value);
        return typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value));
    }
    return true;
}

/** ``value`` if the widget can hold it, otherwise the widget's own default. */
function sanitizeWidgetValue(w, value) {
    return acceptsWidgetValue(w, value) ? value : w?.value;
}

/** A widget's schema default (``configure`` captured them before values applied). */
function widgetDefault(node, w) {
    const captured = node?.__mmxWidgetDefaults?.[w?.name];
    if (typeof captured === "number" || typeof captured === "boolean") return captured;
    return w?.value;
}

/**
 * Remember what the workflow said for each widget (``name → value``).
 *
 * :func:`sanitizeWidgetValues` has to repair widgets that end up *empty* — one added
 * after the file was saved has no stored entry. It used to fall back to 0, which is
 * exactly where users' ``second_seed = 0`` came from (the file said 20240 or 1, the
 * widget was empty, and the repair invented a number). With the loaded values kept
 * here it can put back what the workflow held instead of guessing.
 */
function rememberLoadedValues(node, widgets, values) {
    const out = {};
    widgets.forEach((w, i) => {
        if (w?.name && i < values.length) out[w.name] = values[i];
    });
    node.__mmxLoadedValues = out;
}

/**
 * Opt-in diagnostic: log every widget value this load restored.
 *
 * 「这个参数怎么每次刷新都变回去」needs one fact to be answerable — is that value what
 * the saved file holds, or did something rewrite it while loading? Enable with
 * ``localStorage.mmx_director_debug_widgets = "1"`` (then F5) and the console prints
 * the loaded map once per node.
 */
function logRestoredValues(node, widgets, values) {
    try {
        if (localStorage.getItem("mmx_director_debug_widgets") !== "1") return;
    } catch (e) {
        return;   // storage disabled (private mode): logging is not worth a throw
    }
    const pairs = {};
    widgets.forEach((w, i) => {
        if (w?.name) pairs[w.name] = values[i];
    });
    console.info(`[MiniMax] ${node?.type} 载入控件值（name → 值）:`, pairs);
}

/** 每次载入都打印的几个「用户最常问」的控件，一行、可搜、无需开开关。 */
const LOAD_LOG_WIDGETS = ["use_sigmas", "seed", "second_seed", "frame_rate", "steps"];

/**
 * One always-on line naming what this load actually produced.
 *
 * The value on screen and the value in the file disagreeing has exactly two causes —
 * the loader was handed something else, or something rewrote the widget afterwards —
 * and this line splits them: paste it next to「界面上显示的是多少」and the difference is
 * immediately visible. Field debugging beats a hundred questions.
 */
function logLoadedSummary(node, widgets, values) {
    const byName = {};
    widgets.forEach((w, i) => {
        if (w?.name) byName[w.name] = values[i];
    });
    const parts = LOAD_LOG_WIDGETS
        .filter((name) => name in byName)
        .map((name) => `${name}=${JSON.stringify(byName[name])}`);
    console.info(
        `[MiniMax] ${node?.type} 载入控件值: ${parts.join(" ")} (共 ${widgets.length} 项)`,
    );
}

/**
 * Repair widget values that a workflow saved **while they were shifted**.
 *
 * The shift itself is already handled (REMOVED_DIRECTOR_WIDGETS, and the named map
 * makes loads order-independent), but nothing repaired the *aftermath*: a file saved
 * with shifted values freezes the wrong values **by name**, so every later load hands
 * them back — and the user sees「我明明改过，刷新又变回默认」. Only values whose type
 * was impossible were being replaced, which is why a group header could keep showing a
 * model name forever.
 *
 * A group header is the proof: its value *is* its own label, so anything that is not
 * one of that key's translations (in any locale) can only be a leftover. Once proven,
 * that widget and everything after it go back to their defaults — a shift walks the
 * whole tail, so nothing after the marker can be trusted, and the fields before it
 * (steps / sampler / shift…) are left exactly as saved.
 *
 * ``read(w, i)`` / ``write(w, i, value)`` abstract over the two stores a load can
 * carry: the named map or the positional array.
 */
function healDriftedWidgetValues(node, widgets, read, write) {
    const suspect = [];
    let firstBad = -1;
    for (let i = 0; i < widgets.length; i++) {
        const w = widgets[i];
        if (!w?.name) continue;
        const got = read(w, i);
        if (got === undefined) continue;
        const labelKey = DIRECTOR_GROUP_LABEL_KEYS[w.name];
        const badLabel = !!labelKey && typeof got === "string"
            && !tAnyLocale(labelKey).includes(got);
        const badType = !acceptsWidgetValue(w, got);
        if (badLabel || badType) {
            firstBad = firstBad < 0 ? i : firstBad;
            suspect.push(`${w.name}=${JSON.stringify(got)}`);
        }
    }
    if (firstBad < 0) return 0;
    let fixed = 0;
    for (let i = firstBad; i < widgets.length; i++) {
        const w = widgets[i];
        if (!w?.name) continue;
        const labelKey = DIRECTOR_GROUP_LABEL_KEYS[w.name];
        const want = labelKey ? tAnyLocale(labelKey)[0] : widgetDefault(node, w);
        if (want === undefined || read(w, i) === want) continue;
        write(w, i, want);
        fixed += 1;
    }
    console.warn(
        `[MiniMax] ${node?.type}: **本次载入的数据**里控件值处于错位状态`
        + `（${suspect.join("、")}）——旧版控件删改后保存的残留，会被按名字固化，`
        + `所以刷新后看起来像「被重置成默认值」。已把从 ${widgets[firstBad]?.name} 起的 `
        + `${fixed} 个控件复位为默认值。`
        + "⚠ 这可能是**浏览器里的旧会话副本**而不是磁盘上的工作流文件："
        + "若磁盘文件里的值是好的，请用「Workflow → Open」重新打开该工作流（覆盖会话副本），"
        + "核对「二级采样」组后 **Ctrl+S** 保存一次 —— 之后就不会再错位。",
    );
    return fixed;
}

/**
 * Replace type-impossible entries of a positional array with each widget's default.
 *
 * Only indexes that a widget actually owns are checked; the tail beyond the widget
 * list is a removed widget's leftovers and is ignored here (it is dropped by
 * :func:`stripRemovedDirectorWidgetValues`).
 */
function sanitizePositionalValues(node, values) {
    const widgets = serializableWidgets(node);
    for (let i = 0; i < widgets.length && i < values.length; i++) {
        const fixed = sanitizeWidgetValue(widgets[i], values[i]);
        if (fixed !== values[i]) {
            console.warn(
                `[MiniMax] ${node?.type}: 控件 ${widgets[i]?.name} 的值 ${JSON.stringify(values[i])}`
                + " 类型不合法（旧工作流错位的残留），已回退为默认值。",
            );
            values[i] = fixed;
        }
    }
}

/**
 * Drop positional values that no longer have a widget behind them.
 *
 * Only fires when the array is still longer than the widget list — i.e. the
 * workflow was saved while the removed widgets existed. Anything saved after the
 * removal is left untouched.
 */
export function stripRemovedDirectorWidgetValues(node, info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values)) return;
    const widgets = serializableWidgets(node);
    // 被删控件在旧数组里的位置，就等于它的 successor 在新列表里的下标（旧：removed,
    // successor；新：successor 顶上 removed 的位置 ⇒ 下标相同）。**必须按这个下标升序
    // 处理**：先删小下标、再删大下标时，前面删掉的那一格已经把后面的值整体左移了一位，
    // 而 successor 的新下标里恰好也少算了这一位，两者正好抵消 —— 所以直接 splice(newIdx)
    // 就是旧数组里 removed 那一格。（若按 REMOVED 列表原顺序处理，第 2 个条目的下标会
    // 多算一位，删掉的是 successor 自己的值，整个后半段继续错位。）
    const jobs = [];
    for (const removed of REMOVED_DIRECTOR_WIDGETS) {
        const idx = widgets.findIndex((w) => w?.name === removed.followedBy);
        if (idx >= 0) jobs.push({ name: removed.name, idx });
    }
    jobs.sort((a, b) => a.idx - b.idx);
    for (const job of jobs) {
        // 没有多余值就说明这份存档晚于该次删除：剩下的删除条目对它不适用。
        if (values.length <= widgets.length) break;
        values.splice(job.idx, 1);
        console.warn(
            `[MiniMax] ${node?.type}: 已删除控件 ${job.name}，`
            + `丢弃其在 widgets_values 中下标 ${job.idx} 的旧值。`,
        );
    }
    // Types are checked *after* the splice: whatever still lands on a widget that
    // cannot hold it (a BOOLEAN with「主模型 (model)」) is a leftover of the drift.
    sanitizePositionalValues(node, values);
    // Positions only mean something once the array and the widgets are the same
    // length; with leftovers still in it the alignment is unknown, so healing waits
    // for the next save (which writes the named map).
    if (values.length <= widgets.length) {
        healDriftedWidgetValues(
            node,
            widgets,
            (w, i) => (i < values.length ? values[i] : undefined),
            (w, i, value) => { if (i < values.length) values[i] = value; },
        );
    }
    rememberLoadedValues(node, widgets, values);
    logLoadedSummary(node, widgets, values);
    if (values.length > widgets.length) {
        // Still longer: the workflow predates other removals too, and where those
        // widgets sat cannot be reconstructed — every value after the gap lands on
        // the wrong widget (asr_check / use_sigmas end up holding strings). A
        // re-save writes widgets_values_named alongside the array, which
        // restoreDirectorWidgetValues then reads, realigning it for good.
        console.warn(
            `[MiniMax] ${node?.type}: widgets_values 仍有 ${values.length - widgets.length} 个多余值，`
            + "控件会被错位赋值（如 asr_check / use_sigmas 被塞进字符串）。"
            + "请在 ComfyUI 中核对控件后重新保存该工作流，之后刷新不再错位。",
        );
    }
}

/**
 * Restore widget values *by name* when the workflow carries a named map.
 *
 * ComfyUI only ever reads the positional ``widgets_values`` array, so any widget
 * deleted since the workflow was saved shifts every later value. Saving the map
 * alongside it makes the next load independent of widget order: names present in
 * the map win, anything added later keeps its own default.
 */
export function restoreDirectorWidgetValues(node, info) {
    const named = info?.widgets_values_named;
    if (!named || typeof named !== "object") {
        stripRemovedDirectorWidgetValues(node, info);
        return;
    }
    const widgets = serializableWidgets(node);
    // Repair a file whose values were frozen in a shifted state *before* reading it —
    // otherwise the shift is restored faithfully on every single load.
    healDriftedWidgetValues(
        node,
        widgets,
        (w) => (Object.prototype.hasOwnProperty.call(named, w.name) ? named[w.name] : undefined),
        (w, _i, value) => { named[w.name] = value; },
    );
    const values = [];
    for (const w of widgets) {
        const hit = w?.name && Object.prototype.hasOwnProperty.call(named, w.name);
        values.push(hit ? sanitizeWidgetValue(w, named[w.name]) : w.value);
    }
    // Rewrite the array ComfyUI is about to apply positionally so both paths agree.
    info.widgets_values = values;
    rememberLoadedValues(node, widgets, values);
    logLoadedSummary(node, widgets, values);
    logRestoredValues(node, widgets, values);
}

/**
 * Make Director widget values survive widget deletions.
 *
 * Load: normalise the saved values *before* ComfyUI applies them — patching
 * ``onConfigure`` would be too late, the shifted values are already on the
 * widgets by then. Delegates to whatever ``configure`` was in place (inherited or
 * previously patched) so this stays pure pre-processing.
 *
 * ComfyUI's own ``LGraphNode.serialize()`` already writes a ``widgets_values_named``
 * map next to the positional array, but it only *restores* from it when the
 * ``namedValuesRestore`` setting is on — otherwise the stale positional array wins
 * and every refresh re-shifts the values. Reading the map here makes the Director
 * independent of that setting (and of widget order).
 */
/**
 * Put the loaded values onto the widgets **by name**, after ComfyUI applied its array.
 *
 * ComfyUI only ever applies ``widgets_values`` *positionally*, and it can miss the
 * target (a widget it skips while assigning, an order that drifted, a value that
 * lands on the wrong slot because the file predates a widget). The result is exactly
 * the complaint「刷新后参数被重置」: our loaded values are right, the widgets are not.
 * Writing them by name afterwards is order-independent and makes the last word ours.
 */
function applyLoadedValuesByName(node) {
    const loaded = node?.__mmxLoadedValues;
    if (!loaded) return 0;
    let applied = 0;
    for (const w of node.widgets || []) {
        const name = w?.name;
        if (!name || !Object.prototype.hasOwnProperty.call(loaded, name)) continue;
        const want = loaded[name];
        if (want === undefined) continue;
        const labelKey = DIRECTOR_GROUP_LABEL_KEYS[name];
        const value = labelKey ? (tAnyLocale(labelKey)[0] ?? want) : sanitizeWidgetValue(w, want);
        if (value !== undefined && w.value !== value) {
            w.value = value;
            applied += 1;
        }
    }
    return applied;
}

/**
 * Make Director widget values survive widget deletions.
 *
 * Load: normalise the saved values *before* ComfyUI applies them — patching
 * ``onConfigure`` would be too late, the shifted values are already on the
 * widgets by then. Delegates to whatever ``configure`` was in place (inherited or
 * previously patched) so this stays pure pre-processing.
 *
 * ComfyUI's own ``LGraphNode.serialize()`` already writes a ``widgets_values_named``
 * map next to the positional array, but it only *restores* from it when the
 * ``namedValuesRestore`` setting is on — otherwise the stale positional array wins
 * and every refresh re-shifts the values. Reading the map here makes the Director
 * independent of that setting (and of widget order).
 */
export function patchDirectorWidgetValueMigration(nodeType) {
    const proto = nodeType?.prototype;
    if (!proto || proto.__mmxWidgetValuesMigrated) return;
    const prevConfigure = proto.configure;
    if (typeof prevConfigure !== "function") return;
    proto.__mmxWidgetValuesMigrated = true;
    proto.configure = function (info) {
        try {
            // 先抓默认值，再恢复存档值 —— 顺序不能反。
            captureWidgetDefaults(this);
            restoreDirectorWidgetValues(this, info);
        } catch (e) {
            /* best-effort: a stale array must never block loading */
        }
        const out = prevConfigure.apply(this, arguments);
        // 再按名字落一次：ComfyUI 那一步是**按位置**应用，漏掉/错位就会表现为
        // 「刷新后参数被重置」——值我们已经读对了，只是没落到控件上。
        try {
            const n = applyLoadedValuesByName(this);
            if (n) {
                console.info(`[MiniMax] ${this?.type} 按名字复写 ${n} 个控件值（按位置应用没落实）`);
            }
        } catch (e) {
            /* best-effort */
        }
        return out;
    };
}
