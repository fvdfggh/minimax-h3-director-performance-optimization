/** modes mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { getDirectorMode, isVideoBatchTask } from "../../minimax_gen_timeline.js";
export const modesMixin = {
    getDirectorMode() {
        return getDirectorMode(this.globalTask?.value || this.taskTypeWidget?.value);
    },
    isGenMode() {
        const mode = this.getDirectorMode();
        return mode !== "video" && mode !== "prompt_batch" && mode !== "fl2v";
    },
    isImageBatch() {
        const mode = this.getDirectorMode();
        return mode === "prompt_batch" || mode === "image_batch";
    },
    isGenBlank() {
        return this.getDirectorMode() === "gen_blank";
    },
    isGenImage() {
        return this.getDirectorMode() === "gen_image";
    },
    isFl2vMode() {
        return this.getDirectorMode() === "fl2v";
    },
    isR2vBatch() {
        return this.isImageBatch() && this.getTaskKey() === "r2v";
    },
    /** t2v / i2v / r2v: duration groups on the main timeline track. */
    usesBatchTimeline() {
        return this.isImageBatch() && isVideoBatchTask(this.getTaskKey());
    },
    _syncR2vCardSelection() {
        if (!this.isImageBatch() || !this.batchList) return;
        const runSelectOn = this.isRunSelectEnabled() && this.supportsRunSelect();
        const focusSel = this.isR2vBatch();
        const cards = this.batchList.querySelectorAll(".bd-batch-card");
        cards.forEach((el) => {
            const i = parseInt(el.dataset.batchIndex, 10);
            if (!Number.isFinite(i)) return;
            const runOn = !runSelectOn || this.isSegmentRunEnabled(i);
            el.classList.toggle("selected", focusSel && i === this.selectedIndex);
            el.classList.toggle("run-on", runSelectOn && runOn);
            el.classList.toggle("run-skipped", runSelectOn && !runOn);
            const cb = el.querySelector(".bd-batch-run-check");
            if (cb) cb.checked = runOn;
            // 方案B: 分段导出 / 二次采样按钮与「选择运行」状态解耦,
            // 不被 run-skipped 整卡片灰化波及, 其可用与否仅由各自功能决定。
            this._decoupleRunSelectFromExportUI(el);
        });
        this.batchPicker?.querySelectorAll?.(".bd-batch-pick").forEach((el) => {
            const i = parseInt(el.dataset.batchIndex, 10);
            if (!Number.isFinite(i)) return;
            const runOn = !runSelectOn || this.isSegmentRunEnabled(i);
            el.classList.toggle("selected", i === this.selectedIndex);
            el.classList.toggle("run-on", runSelectOn && runOn);
            el.classList.toggle("run-skipped", runSelectOn && !runOn);
            const cb = el.querySelector(".bd-batch-run-check");
            if (cb) cb.checked = runOn;
            this._decoupleRunSelectFromExportUI(el);
        });
    },
    _decoupleRunSelectFromExportUI(scope) {
        if (!scope || !scope.querySelectorAll) return;
        scope.querySelectorAll(
            '[data-a="seg-export"],[data-a="second-sample"],' +
            '.bd-seg-export-badge,.bd-second-sample-badge,.bd-batch-preview,' +
            '.bd-r2v-thumb,.bd-batch-video'
        ).forEach((node) => {
            node.style.opacity = "1";
            if ("disabled" in node) node.disabled = false;
        });
    },
    onTaskTypeChanged(value) {
        this.onGlobalField("taskType", value);
    }
};
