/** run_selection mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { getStableWorkflowId } from "../../core/graph_refs.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { fl2vStartIndices } from "../../minimax_fl2v.js";
import { isPromptBatchTask, resolveTaskKey } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const run_selectionMixin = {
    getTaskKey() {
        return resolveTaskKey(
            this.globalTask?.value
            || this.timeline.global?.taskType
            || this.taskTypeWidget?.value,
        );
    },
    getRunnableSegmentCount() {
        if (this.isFl2vMode()) return fl2vStartIndices(this).length;
        return this.timeline.segments?.length || 0;
    },
    supportsRunSelect() {
        const n = this.getRunnableSegmentCount();
        if (n < 2) return false;
        const mode = this.getDirectorMode();
        if (mode === "video") return true;
        if (mode === "fl2v") return true;
        if (this.isImageBatch()) return isPromptBatchTask(this.getTaskKey());
        return false;
    },
    getRunProgressSegmentTotal() {
        const n = this.getRunnableSegmentCount();
        if (!this.isRunSelectEnabled() || n < 2) return Math.max(n, 1);
        const count = (this.timeline.runSelection || []).length;
        return count > 0 ? count : Math.max(n, 1);
    },
    isRunSelectEnabled() {
        return !!this.timeline.runSelectEnabled;
    },
    normalizeRunSelection() {
        if (!this.isRunSelectEnabled()) return;
        if (this.isFl2vMode()) {
            const valid = new Set(fl2vStartIndices(this));
            this.timeline.runSelection = [...new Set(
                (this.timeline.runSelection || []).filter((i) => valid.has(i)),
            )].sort((a, b) => a - b);
            return;
        }
        const n = this.getRunnableSegmentCount();
        if (n < 1) return;
        this.timeline.runSelection = [...new Set(
            (this.timeline.runSelection || []).filter((i) => i >= 0 && i < n),
        )].sort((a, b) => a - b);
    },
    dropRunSelectionIndex(removedIndex) {
        if (!this.isRunSelectEnabled()) return;
        const removed = parseInt(removedIndex, 10);
        if (!Number.isFinite(removed)) return;
        this.timeline.runSelection = [...new Set(
            (this.timeline.runSelection || [])
                .map((i) => (i > removed ? i - 1 : i))
                .filter((i) => i !== removed && i >= 0),
        )].sort((a, b) => a - b);
    },
    moveRunSelectionIndex(fromRank, toRank) {
        if (!this.isRunSelectEnabled()) return;
        const from = parseInt(fromRank, 10);
        const to = parseInt(toRank, 10);
        if (!Number.isFinite(from) || !Number.isFinite(to) || from === to) return;
        this.timeline.runSelection = [...new Set(
            (this.timeline.runSelection || []).map((i) => {
                if (i === from) return to;
                if (from < to) return (i > from && i <= to) ? i - 1 : i;
                return (i >= to && i < from) ? i + 1 : i;
            }),
        )].sort((a, b) => a - b);
    },
    dropSegmentSlotCache(index) {
        const nodeId = String(this.node?.id ?? "");
        if (!nodeId) return;
        api.fetchApi("/minimax/director_opt/remove_segment_slot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_id: nodeId,
                workflow_name: getStableWorkflowId(),
                index: parseInt(index, 10) || 0,
            }),
        }).catch((err) => {
            console.warn("[MiniMax H3 Director Opt] segment cache drop failed:", err);
        });
    },
    /** Bookkeeping every group / segment deletion must do. */
    onSegmentRemoved(index) {
        this.dropSegmentSlotCache(index);
        this.dropRunSelectionIndex(index);
    },
    isSegmentRunEnabled(index) {
        if (!this.isRunSelectEnabled()) return true;
        return (this.timeline.runSelection || []).includes(index);
    },
    /** Map of plan index -> canAlignToNext, refreshed from the backend. */
    _alignToNextMap() {
        return this._alignToNextCache || {};
    },
    async refreshAlignToNextStatus() {
        if (!this.isRunSelectEnabled() || !this.supportsRunSelect()) {
            this._alignToNextCache = {};
            return;
        }
        const payload = {
            node_id: String(this.node?.id ?? ""),
            timeline_data: this.buildTimelinePayload(),
            task_type: this.globalTask?.value || this.taskTypeWidget?.value || "",
            global_prompt: this.timeline.global?.prompt || "",
            total_frames: this.getTotalFrames(),
            frame_rate: this.timeline.output?.frameRate || 24,
            width: this.timeline.output?.width || 864,
            height: this.timeline.output?.height || 480,
            ref_max_size: this.refMaxWidget?.value || this.timeline.output?.longEdge || 864,
            workflow_name: getStableWorkflowId(),
        };
        try {
            const resp = await api.fetchApi("/minimax/director_opt/align_to_next_status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) throw new Error(String(resp.status));
            const data = await resp.json();
            const map = {};
            (data?.segments || []).forEach((row) => {
                // Align-to-next availability depends only on the next segment's
                // cached AV latent; it is independent of this segment's continuity
                // flag. Do NOT gate it on `row.continuity` — the first segment has
                // continuity=false (no predecessor to pin) and must stay selectable.
                map[row.index] = !!row.canAlignToNext;
            });
            this._alignToNextCache = map;
        } catch (e) {
            // Keep the last-known align-to-next state instead of wiping it; a
            // transient failure shouldn't grey every「对齐下段」control.
            console.warn("[MiniMax H3 Director Opt] align-to-next status refresh failed:", e);
        }
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    },
    canAlignToNext(index) {
        if (!this.isRunSelectEnabled()) return false;
        const seg = this.timeline.batch?.segments?.[index] ?? this.timeline.segments?.[index];
        if (!seg) return false;
        const n = this.getRunnableSegmentCount();
        if (index >= n - 1) return false; // last segment has no next segment
        const map = this._alignToNextMap();
        if (index in map) return map[index];
        return false; // unknown until the status call lands — stay disabled
    },
    toggleSegmentRun(index) {
        if (!this.isRunSelectEnabled()) return;
        if (this.isFl2vMode()) {
            if (!this.timeline.segments?.[index]?.isStartFrame) return;
        } else {
            const n = this.getRunnableSegmentCount();
            if (index < 0 || index >= n) return;
        }
        const sel = new Set(this.timeline.runSelection || []);
        if (sel.has(index)) sel.delete(index);
        else sel.add(index);
        this.timeline.runSelection = [...sel].sort((a, b) => a - b);
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    },
    toggleRunSelectMode() {
        if (!this.supportsRunSelect()) return;
        this.timeline.runSelectEnabled = !this.timeline.runSelectEnabled;
        //「对齐下段」only exists in this mode, so its availability is fetched
        // here — that is also when the user asked "what can I tick?".
        void this.refreshAlignToNextStatus();
        if (this.timeline.runSelectEnabled) {
            if (!(this.timeline.runSelection || []).length) {
                if (this.isFl2vMode()) {
                    this.timeline.runSelection = fl2vStartIndices(this);
                } else {
                    const n = this.getRunnableSegmentCount();
                    this.timeline.runSelection = Array.from({ length: n }, (_, i) => i);
                }
            } else {
                this.normalizeRunSelection();
            }
        }
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    },
    setRunSelectionAll(on) {
        if (!this.isRunSelectEnabled()) return;
        if (this.isFl2vMode()) {
            this.timeline.runSelection = on ? fl2vStartIndices(this) : [];
            this.updateRunSelectUI();
            this.commit(false, { syncTimeline: true });
            this.scheduleRender();
            return;
        }
        const n = this.getRunnableSegmentCount();
        this.timeline.runSelection = on ? Array.from({ length: n }, (_, i) => i) : [];
        this.updateRunSelectUI();
        this.commit(false, { syncTimeline: true });
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    },
    updateRunSelectUI() {
        const n = this.getRunnableSegmentCount();
        const canRunSelect = this.supportsRunSelect();
        const enabled = this.isRunSelectEnabled() && canRunSelect;
        // r2v uses timeline checkboxes (fl2v-style); other batch tasks use the card bar.
        const useBatchBar = this.isImageBatch() && canRunSelect && !this.isR2vBatch();
        this.btnRunSelectToggle?.classList.toggle("active", enabled);
        this.btnRunSelectToggle?.classList.toggle("bd-btn-run-select", true);
        this.btnRunSelectToggle?.classList.toggle("hidden", !canRunSelect || useBatchBar);
        this.batchRunSelectBtn?.classList.toggle("active", enabled);
        this.batchRunSelectBtn?.classList.toggle("hidden", !useBatchBar);
        this.runSelectAllWrap?.classList.toggle("hidden", !enabled || useBatchBar);
        this.batchRunSelectAllWrap?.classList.toggle("hidden", !enabled || !useBatchBar);
        // Keep the chip hidden while a run is active — otherwise commit/sync
        // re-shows it on top of the green progress title.
        const running = !!this.runStatusEl?.classList.contains("active");
        this.runSelectBar?.classList.toggle("hidden", !enabled || running);
        if (!canRunSelect) return;
        this.normalizeRunSelection();
        const count = (this.timeline.runSelection || []).length;
        const syncAllCb = (cb) => {
            if (!cb) return;
            cb.checked = count >= n && n > 0;
            cb.indeterminate = count > 0 && count < n;
        };
        syncAllCb(this.runSelectAllCb);
        syncAllCb(this.batchRunSelectAllCb);
        const label = t(this.isImageBatch() ? "unit.group" : "unit.segment");
        if (!this.runSelectSummary) return;
        if (!count) {
            this.runSelectSummary.textContent = t("runSelect.noneChecked", { unit: label });
            this.runSelectSummary.style.color = "#f88";
        } else if (count >= n) {
            this.runSelectSummary.textContent = t("runSelect.all", { n, unit: label });
            this.runSelectSummary.style.color = "#aaa";
        } else {
            const nums = (this.timeline.runSelection || []).map((i) => i + 1).join(", ");
            const exportHint = this.timeline.output?.exportMode === "segments"
                ? t("runSelect.exportOnlyChecked")
                : t("runSelect.fillUnchecked");
            this.runSelectSummary.textContent = count === 1
                ? t("runSelect.sampleOne", { unit: label, nums, hint: exportHint })
                : t("runSelect.sampleMany", { count, unit: label, nums, hint: exportHint });
            this.runSelectSummary.style.color = "#4fff8f";
        }
    },
    /** Drop live run-select flags (mode switch). Stashed workspaces keep their own copy. */
    _clearLiveRunSelection() {
        this.timeline.runSelectEnabled = false;
        this.timeline.runSelection = [];
    }
};
