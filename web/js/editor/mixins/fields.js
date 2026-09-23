/** fields mixin for the Director editor (fields).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { resolveTaskKey } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const fieldsMixin = {
    onGlobalField(field, value) {
        this.timeline.global = this.timeline.global || { refs: [] };
        if (field === "taskType") {
            const prevTaskKey = this._taskKey || resolveTaskKey(this.timeline.global?.taskType || "");
            this.timeline.global[field] = value;
            const prevMode = this._directorMode || "video";
            if (this.globalTask && this.globalTask.value !== value) this.globalTask.value = value;
            if (this.taskTypeWidget) this.taskTypeWidget.value = value;
            if (prevTaskKey === "ads2v" && resolveTaskKey(value) !== "ads2v") {
                this._stopRefVideoPreviews();
            }
            this.applyTaskLayout(prevMode, prevTaskKey);
            this.updateSegmentContinuityUI();
        } else {
            this.timeline.global[field] = value;
        }
        if (field === "prompt" && this.globalPromptWidget) this.globalPromptWidget.value = value;
        this.scheduleTimelineSync();
        if (field === "prompt") this._schedulePromptRender();
        else this.scheduleRender();
    },
    /** Debounced render for prompt typing — avoids full canvas redraw on every keystroke. */
    _schedulePromptRender() {
        if (this._promptRenderTimer != null) return;
        this._promptRenderTimer = setTimeout(() => {
            this._promptRenderTimer = null;
            this.scheduleRender();
        }, 160);
    },
    onSegField(field, value) {
        const seg = this.timeline.segments[this.selectedIndex];
        if (!seg) return;
        seg[field] = value;
        this.scheduleTimelineSync();
        this._schedulePromptRender();
    },
    onNegativePrompt(value) {
        if (this.negativePromptWidget) this.negativePromptWidget.value = value;
        if (this.globalNegative && this.globalNegative.value !== value) this.globalNegative.value = value;
        if (this.segNegative && this.segNegative.value !== value) this.segNegative.value = value;
        this._markNodeDirtyLight();
    },
    toggleLoop() {
        this.isLooping = !this.isLooping;
        const btn = this.root.querySelector('[data-a="loop"]');
        btn?.classList.toggle("active", this.isLooping);
        this.refreshLoopButtonTitle();
    },
    refreshLoopButtonTitle() {
        const btn = this.root?.querySelector('[data-a="loop"]');
        if (!btn) return;
        btn.title = this.isLooping ? t("player.loopEnabled") : t("player.loopOff");
        btn.removeAttribute("data-i18n-title");
    }
};
