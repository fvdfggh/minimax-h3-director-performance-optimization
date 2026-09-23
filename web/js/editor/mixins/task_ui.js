/** task_ui mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { getStableWorkflowId } from "../../core/graph_refs.js";
import { applyDirectorWidgetLabels } from "../../core/widget_labels.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { updateFl2vDetailUI, updateFl2vToolbarBtns } from "../../minimax_fl2v.js";
import { resolveTaskKey, taskUsesReferenceAudios, taskUsesReferenceImages, taskUsesReferenceVideo } from "../../minimax_gen_timeline.js";
import { applyI18nDom, aspectDisplayLabel, getLocale, t, taskDisplayLabel } from "../../minimax_i18n.js";
import { updateR2vToolbarBtns } from "../../minimax_image_batch.js";
export const task_uiMixin = {
    isGlobalMode() { return (this.timeline.editMode || "global") === "global"; },
    /** r2v batch: show timeline.global as shared params for all asset groups. */
    usesR2vCommonPanel() {
        return !!this.isR2vBatch?.();
    },
    isR2vCommonEnabled() {
        if (!this.usesR2vCommonPanel()) return false;
        return true;
    },
    /** UI-only fold; shared page is always expanded (no fold state any more). */
    isR2vCommonCollapsed() {
        return false;
    },
    getWorkflowId() {
        return getStableWorkflowId();
    },
    /** Global / shared-ref panel owns timeline.global refs + prompt when enabled. */
    usesGlobalRefPanel() {
        return this.isGlobalMode() || this.isR2vCommonEnabled();
    },
    syncR2vCommonCollapse() {
        const r2v = this.usesR2vCommonPanel();
        this.globalPanel?.classList.toggle("bd-r2v-common-panel", r2v);
        this.globalPanel?.classList.toggle("bd-r2v-common-collapsed", false);
        // r2v: the whole 公共参数 block is removed — shared assets/prompt live on
        // their own paginator page inside the batch card instead.
        this.splitEl?.classList.toggle("hidden", !!r2v);
        this.r2vCommonHint?.classList.toggle("hidden", true);
        this.r2vCommonFold?.classList.toggle("hidden", true);
        this.r2vCommonToggle?.classList.toggle("hidden", true);
        this.r2vCommonStatus?.classList.toggle("hidden", true);
        if (r2v && this.timeline?.global) {
            // Keep the persisted flag in sync so the backend merge always runs.
            this.timeline.global.commonEnabled = true;
            this.timeline.global.commonCollapsed = false;
        }
        if (r2v && this.globalPrompt) {
            this.globalPrompt.placeholder = t("placeholder.r2vCommonPrompt");
            this.globalPrompt.setAttribute("data-i18n-placeholder", "placeholder.r2vCommonPrompt");
        }
        // Keep shared layout class in sync so image/audio slot chrome paints correctly.
        // Shared assets are always on in r2v, so the layout class is always applied.
        if (r2v) {
            this.globalPromptLayout?.classList.toggle("bd-rv2v-layout", true);
            this.globalPanel?.classList.toggle("bd-rv2v-panel", true);
        }
    },
    setEditMode(mode) {
        this.timeline.editMode = mode;
        this.root.querySelector('[data-a="mode-global"]').classList.toggle("active", mode === "global");
        this.root.querySelector('[data-a="mode-segment"]').classList.toggle("active", mode === "segment");
        this.updateModeUI();
        this.commit();
    },
    updateModeUI() {
        const global = this.isGlobalMode();
        const r2vCommon = this.usesR2vCommonPanel();
        const r2vOn = this.isR2vCommonEnabled();
        this.globalPanel.style.display = (global || r2vCommon) ? "flex" : "none";
        this.segmentPanel.style.display = (global || r2vCommon) ? "none" : "flex";
        this.syncR2vCommonCollapse();
        this.updateReferenceImageVisibility({
            // Show shared ref chrome only when r2v common is enabled (expanded).
            hideTimeline: (this.isImageBatch() && !r2vOn) || this.isGenMode(),
            seg: (global || r2vOn) ? null : this.timeline.segments[this.selectedIndex],
        });
        if (!global && !r2vCommon) this.updateSelectionUI();
        else {
            this.updateSelectionUI();
            if (taskUsesReferenceVideo(this.getTaskKey())) this.renderRefVideoSlot();
        }
        this.updateLiveSamplePanel();
    },
    getRefTarget() {
        if (this.usesGlobalRefPanel()) return this.timeline.global;
        const seg = this.timeline.segments[this.selectedIndex];
        return seg || this.timeline.global;
    },
    getDisplayPrompt(seg) {
        if (this.isGlobalMode()) return this.timeline.global?.prompt || "";
        return seg?.prompt || "";
    },
    populateTaskSelect(el, selected) {
        if (!el) return;
        const opts = this.taskTypeWidget?.options?.values || [];
        const prev = selected || el.value;
        el.innerHTML = "";
        for (const v of opts) {
            const o = document.createElement("option");
            o.value = v;
            const key = resolveTaskKey(v);
            o.textContent = taskDisplayLabel(key) || v;
            el.appendChild(o);
        }
        if (prev) el.value = prev;
    },
    refreshAspectSelectLabels() {
        if (!this.outAspect) return;
        const cur = this.outAspect.value;
        for (const opt of this.outAspect.options || []) {
            opt.textContent = aspectDisplayLabel(opt.value);
        }
        if (cur) this.outAspect.value = cur;
    },
    applyLocale() {
        this.root?.classList.toggle("locale-en", getLocale() === "en");
        this.root?.classList.toggle("locale-zh", getLocale() !== "en");
        applyI18nDom(this.root);
        applyDirectorWidgetLabels(this.node);
        this.populateTaskSelect(this.globalTask, this.taskTypeWidget?.value || this.globalTask?.value);
        this.refreshAspectSelectLabels();
        // Re-apply dynamic UI strings that overwrite data-i18n nodes.
        this.updateVideoNameLabel?.();
        this.updateRunSelectUI?.();
        this.updateOutputPreview?.();
        this.updateSelectionUI?.();
        this.refreshLoopButtonTitle?.();
        this.refreshLiveTaePreviewButton?.();
        this.updateLiveSamplePanel?.();
        this.syncTimelineZoomUI?.();
        this.syncExternalGroupsTimeline?.();
        updateFl2vDetailUI?.(this);
        updateFl2vToolbarBtns?.(this);
        updateR2vToolbarBtns?.(this);
        this.renderImageBatchGroups?.();
        const r2vOn = this.isR2vCommonEnabled?.();
        this.syncR2vCommonCollapse?.();
        this.syncRv2vRefLayoutClasses?.({
            hideTimeline: (this.isImageBatch?.() && !r2vOn) || this.isGenMode?.(),
            seg: this.usesGlobalRefPanel?.() ? null : this.timeline?.segments?.[this.selectedIndex],
        });
        if (this.usesGlobalRefPanel?.() && taskUsesReferenceImages(this.getTaskKey())) {
            if (this.timeline?.global) this.timeline.global.refs = this.timeline.global.refs || [];
            this.renderRefSlots?.(this.timeline.global?.refs, this.globalRefsBox, true);
        } else if (!this.usesGlobalRefPanel?.()) {
            const seg = this.timeline?.segments?.[this.selectedIndex];
            if (seg && taskUsesReferenceImages(resolveTaskKey(seg.taskType || this.getTaskKey()))) {
                this.renderRefSlots?.(seg.refs, this.segRefsBox, false);
            }
        }
        if (taskUsesReferenceAudios(this.getTaskKey())) this.renderRefAudioSlots?.();
        if (this.usesR2vCommonPanel?.()) this.renderR2vCommonVideoSlots?.();
        this.scheduleRender?.();
        this.node?.setDirtyCanvas?.(true, true);
    }
};
