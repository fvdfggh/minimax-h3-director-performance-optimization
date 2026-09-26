/** Applying i18n labels to a Director node's widgets.
 *
 * ComfyUI renders a widget's raw ``name`` unless a ``label`` is set, and the
 * ``control_after_generate`` combo is created by ComfyUI core, so it has no name of
 * ours to map. This module owns both the name → i18n-key tables and the single pass
 * that applies them, including the regex fallback for that linked control and for
 * the canvas-drawn group headers (``_mmxGroupI18nKey``).
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { t } from "../minimax_i18n.js";

const DIRECTOR_WIDGET_LABEL_KEYS = {
    seed: "widget.seed",
    control_after_generate: "widget.controlAfterGenerate",
    "control after generate": "widget.controlAfterGenerate",
};

const DIRECTOR_WIDGET_TOOLTIP_KEYS = {};

export const DIRECTOR_GROUP_LABEL_KEYS = {
    bd_grp_sample: "widget.grpSample",
    bd_grp_advanced: "widget.grpAdvanced",
    // 二级采样组：以前漏在这张表外，标题永远不被规范化（英文界面也一直是中文），
    // 存档里一旦被写坏就再也纠正不回来。
    bd_grp_second: "widget.grpSecond",
};

export function applyDirectorWidgetLabels(node) {
    for (const w of node.widgets || []) {
        const name = String(w.name || "");
        const key = DIRECTOR_WIDGET_LABEL_KEYS[name]
            || (/(control[_\s]?after[_\s]?generate|生成前后)/i.test(name) || /生成前后/.test(String(w.label || ""))
                ? "widget.controlAfterGenerate"
                : null);
        if (key) {
            w.label = t(key);
            if (w.options) w.options.label = w.label;
        }
        const tipKey = DIRECTOR_WIDGET_TOOLTIP_KEYS[name];
        if (tipKey && w.options) w.options.tooltip = t(tipKey);
        const gKey = DIRECTOR_GROUP_LABEL_KEYS[name] || w._mmxGroupI18nKey;
        if (gKey) {
            const label = t(gKey);
            w._mmxGroupI18nKey = gKey;
            w._bdGroupLabel = label;
            w.value = label;
            if (w.element) w.element.textContent = label;
        }
        // Linked seed → control_after_generate combo (ComfyUI core).
        for (const linked of w.linkedWidgets || []) {
            const ln = String(linked?.name || linked?.label || "");
            if (/(control[_\s]?after[_\s]?generate|生成前后)/i.test(ln)) {
                linked.label = t("widget.controlAfterGenerate");
                if (linked.options) linked.options.label = linked.label;
            }
        }
    }
}
