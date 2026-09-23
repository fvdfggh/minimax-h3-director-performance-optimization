/** export_pickers mixin for the Director editor (export_pickers).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { getStableWorkflowId } from "../../core/graph_refs.js";
import { t } from "../../minimax_i18n.js";
import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
export const export_pickersMixin = {
    /** Stored under ``timeline.output.segmentExport``; written by the picker. */
    _segmentExportConfig() {
        const cfg = this.timeline.output?.segmentExport || {};
        return {
            enabled: !!cfg.enabled,
            mode: cfg.mode === "continuous" ? "continuous" : "piecewise",
            // Which pass's cache the picker reads: "1st" (seg_*) or "2nd" (seg2_*).
            source: cfg.source === "2nd" ? "2nd" : "1st",
            indices: Array.isArray(cfg.indices)
                ? cfg.indices.map((i) => parseInt(i, 10)).filter((i) => i >= 0)
                : [],
        };
    },
    _segmentExportPayload() {
        return { segmentExport: this._segmentExportConfig() };
    },
    getSegmentExportNodeId() {
        return String(this.node?.id ?? "");
    },
    async _fetchSegmentExportStatus(source) {
        const payload = {
            node_id: this.getSegmentExportNodeId(),
            timeline_data: this.buildTimelinePayload(),
            task_type: this.globalTask?.value || this.taskTypeWidget?.value || "",
            global_prompt: this.timeline.global?.prompt || "",
            total_frames: this.getTotalFrames(),
            frame_rate: this.getFrameRate(),
            width: this.timeline.output?.width || 864,
            height: this.timeline.output?.height || 480,
            ref_max_size: this.refMaxWidget?.value || this.timeline.output?.longEdge || 864,
            workflow_name: getStableWorkflowId(),
            // The backend probes one pass only, so switching the toggle changes
            // what is offered — there is no cross-source fallback.
            source: source === "2nd" ? "2nd" : "1st",
        };
        try {
            const resp = await api.fetchApi("/minimax/director_opt/segment_export_status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            return await resp.json();
        } catch (e) {
            console.error("[MiniMax] segment export status failed", e);
            return { segments: [], error: String(e) };
        }
    },
    async openSegmentExportPicker() {
        this._closeBdModal();
        const n = this.getRunnableSegmentCount();
        const segments = (this.timeline.segments || []).slice(0, n);
        const cfg = this._segmentExportConfig();

        const overlay = document.createElement("div");
        overlay.className = "bd-modal-overlay bd-modal-overlay-fixed";
        const panel = document.createElement("div");
        panel.className = "bd-modal bd-modal-wide";
        panel.innerHTML = `
            <div class="bd-modal-title"></div>
            <div class="bd-modal-body"></div>
            <div class="bd-modal-list"></div>
            <div class="bd-modal-actions"></div>`;
        panel.querySelector(".bd-modal-title").textContent = t("segmentExport.title");
        const bodyEl = panel.querySelector(".bd-modal-body");
        const listEl = panel.querySelector(".bd-modal-list");
        const actionsEl = panel.querySelector(".bd-modal-actions");

        const finish = (val) => {
            this._closeBdModal();
            if (val) this.resolveSegmentExport(val);
            // Cancelling clears a leftover flag: the picker cannot be confirmed
            // with zero segments (OK is disabled), so this is the only way back
            // to "enabled" once it has been persisted.
            else this._clearSegmentExportFlag();
        };

        // Mode — piecewise (one file per segment) or continuous (adjacent
        // checked segments are stitched into one file).
        const modeWrap = document.createElement("div");
        modeWrap.className = "bd-seg-export-mode";
        modeWrap.innerHTML = `<span class="bd-seg-export-mode-label">${t("segmentExport.mode")}</span>
            <label><input type="radio" name="seg-export-mode" value="piecewise"${cfg.mode === "continuous" ? "" : " checked"}> ${t("segmentExport.modePiecewise")}</label>
            <label><input type="radio" name="seg-export-mode" value="continuous"${cfg.mode === "continuous" ? " checked" : ""}> ${t("segmentExport.modeContinuous")}</label>`;
        const modeHint = document.createElement("div");
        modeHint.className = "bd-seg-export-hint";
        const syncModeHint = () => {
            const picked = modeWrap.querySelector('input[name="seg-export-mode"]:checked')?.value;
            modeHint.textContent = t(
                picked === "continuous" ? "segmentExport.modeHintContinuous" : "segmentExport.modeHintPiecewise"
            );
        };
        modeWrap.querySelectorAll('input[name="seg-export-mode"]').forEach((r) => {
            r.onchange = syncModeHint;
        });
        syncModeHint();
        bodyEl.appendChild(modeWrap);
        bodyEl.appendChild(modeHint);

        // Cache source — first pass (seg_*) or second pass (seg2_*). Switching it
        // re-probes the backend and re-renders the list, so the picker only ever
        // shows the selected pass's cache state.
        const currentSource = () =>
            sourceWrap.querySelector('input[name="seg-export-source"]:checked')?.value === "2nd"
                ? "2nd"
                : "1st";
        const sourceWrap = document.createElement("div");
        sourceWrap.className = "bd-seg-export-mode";
        sourceWrap.innerHTML = `<span class="bd-seg-export-mode-label">${t("segmentExport.source")}</span>
            <label><input type="radio" name="seg-export-source" value="1st"${cfg.source === "2nd" ? "" : " checked"}> ${t("segmentExport.sourceFirst")}</label>
            <label><input type="radio" name="seg-export-source" value="2nd"${cfg.source === "2nd" ? " checked" : ""}> ${t("segmentExport.sourceSecond")}</label>`;
        const sourceHint = document.createElement("div");
        sourceHint.className = "bd-seg-export-hint";
        const syncSourceHint = () => {
            sourceHint.textContent = t(
                currentSource() === "2nd"
                    ? "segmentExport.sourceHintSecond"
                    : "segmentExport.sourceHintFirst"
            );
        };
        bodyEl.appendChild(sourceWrap);
        bodyEl.appendChild(sourceHint);

        // Segment list
        const hint = document.createElement("div");
        hint.className = "bd-seg-export-hint";
        hint.textContent = t("segmentExport.desc");
        bodyEl.appendChild(hint);
        listEl.classList.remove("hidden");

        const checkboxes = [];
        const rowEls = [];
        const refreshCount = () => {
            const checked = checkboxes.filter((c) => c.checked && !c.disabled).length;
            countEl.textContent = checked
                ? t("segmentExport.count", { n: checked })
                : t("segmentExport.empty");
            okBtn.disabled = checked === 0;
        };

        let countEl = null;
        const countRow = document.createElement("div");
        countRow.className = "bd-seg-export-count";
        bodyEl.appendChild(countRow);
        countEl = countRow;

        const okBtn = document.createElement("button");
        okBtn.type = "button";
        okBtn.className = "bd-btn bd-btn-primary";
        okBtn.textContent = t("segmentExport.export");
        okBtn.onclick = () => {
            const indices = [];
            checkboxes.forEach((cb, i) => {
                if (cb.checked && !cb.disabled) indices.push(i);
            });
            const mode = modeWrap.querySelector('input[name="seg-export-mode"]:checked')?.value === "continuous"
                ? "continuous"
                : "piecewise";
            finish({ enabled: indices.length > 0, mode, source: currentSource(), indices });
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "bd-btn";
        cancelBtn.textContent = t("dialog.cancel");
        cancelBtn.onclick = () => finish(null);
        actionsEl.appendChild(cancelBtn);
        actionsEl.appendChild(okBtn);

        // Fetch availability, then render rows (disabled where not exportable).
        // Re-runnable: the cache-source toggle re-probes and rebuilds the list.
        const renderRows = async () => {
            const source = currentSource();
            syncSourceHint();
            listEl.textContent = "";
            checkboxes.length = 0;
            rowEls.length = 0;
            listEl.classList.add("bd-loading");
            okBtn.disabled = true;

            const status = await this._fetchSegmentExportStatus(source);
            const rows = (status && status.segments) || [];
            const avail = {};
            for (const r of rows) avail[r.index] = r;
            for (let i = 0; i < segments.length; i++) {
                const info = avail[i] || {};
                const exportable = !!info.exportable;
                const row = document.createElement("div");
                row.className = "bd-modal-item bd-seg-export-item" + (exportable ? "" : " disabled");
                const name = `${i + 1}`;
                const badge = this._segmentExportBadge(info);
                row.innerHTML = `<span class="bd-seg-export-name">#${name}</span><span class="bd-seg-export-badges">${badge}</span>`;
                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.className = "bd-seg-export-cb";
                cb.disabled = !exportable;
                cb.checked = exportable && cfg.indices.includes(i);
                cb.onchange = refreshCount;
                row.prepend(cb);
                row.onclick = (e) => {
                    if (e.target === cb) return;
                    if (!cb.disabled) {
                        cb.checked = !cb.checked;
                        cb.onchange && cb.onchange();
                    }
                };
                listEl.appendChild(row);
                checkboxes.push(cb);
                rowEls.push(row);
            }
            listEl.classList.remove("bd-loading");
            refreshCount();
        };

        sourceWrap.querySelectorAll('input[name="seg-export-source"]').forEach((r) => {
            r.onchange = () => renderRows();
        });
        await renderRows();

        // Escape closes
        const keyHandler = (e) => {
            if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                finish(null);
            }
        };
        window.addEventListener("keydown", keyHandler, true);
        this._modalKeyHandler = keyHandler;

        overlay.onclick = (e) => { if (e.target === overlay) finish(null); };
        panel.onclick = (e) => e.stopPropagation();
        overlay.appendChild(panel);
        document.body.appendChild(overlay);
        // Register with the shared modal owner so _closeBdModal() cleans it up.
        this._modalEl = overlay;
    },
    _secondSampleConfig() {
        const cfg = this.timeline.output?.secondSample || {};
        return {
            enabled: !!cfg.enabled,
            indices: Array.isArray(cfg.indices)
                ? cfg.indices.map((i) => parseInt(i, 10)).filter((i) => i >= 0)
                : [],
        };
    },
    getSecondSampleNodeId() {
        return String(this.node?.id ?? "");
    },
    async _fetchSecondSampleStatus() {
        const payload = {
            node_id: this.getSecondSampleNodeId(),
            timeline_data: this.buildTimelinePayload(),
            task_type: this.globalTask?.value || this.taskTypeWidget?.value || "",
            global_prompt: this.timeline.global?.prompt || "",
            total_frames: this.getTotalFrames(),
            frame_rate: this.getFrameRate(),
            width: this.timeline.output?.width || 864,
            height: this.timeline.output?.height || 480,
            ref_max_size: this.refMaxWidget?.value || this.timeline.output?.longEdge || 864,
            workflow_name: getStableWorkflowId(),
        };
        try {
            const resp = await api.fetchApi("/minimax/director_opt/second_sample_status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            return await resp.json();
        } catch (e) {
            console.error("[MiniMax] second sample status failed", e);
            return { segments: [], error: String(e) };
        }
    },
    async openSecondSamplePicker() {
        this._closeBdModal();
        const n = this.getRunnableSegmentCount();
        const segments = (this.timeline.segments || []).slice(0, n);
        const cfg = this._secondSampleConfig();

        const overlay = document.createElement("div");
        overlay.className = "bd-modal-overlay bd-modal-overlay-fixed";
        const panel = document.createElement("div");
        panel.className = "bd-modal bd-modal-wide";
        panel.innerHTML = `
            <div class="bd-modal-title">${t("secondSample.title")}</div>
            <div class="bd-modal-body"></div>
            <div class="bd-modal-list"></div>
            <div class="bd-modal-actions"></div>`;
        const bodyEl = panel.querySelector(".bd-modal-body");
        const listEl = panel.querySelector(".bd-modal-list");
        const actionsEl = panel.querySelector(".bd-modal-actions");

        const finish = (val) => {
            this._closeBdModal();
            if (val) this.resolveSecondSample(val);
            else this._clearSecondSampleFlag();
        };

        const hint = document.createElement("div");
        hint.className = "bd-seg-export-hint";
        hint.textContent = t("secondSample.hint");
        bodyEl.appendChild(hint);
        const legend = document.createElement("div");
        legend.className = "bd-seg-export-legend";
        legend.innerHTML = `<span class="bd-seg-dot ready"></span>${t("secondSample.legendCan")} <span class="bd-seg-dot partial"></span>${t("secondSample.legendCachedNo")} <span class="bd-seg-dot empty"></span>${t("segmentExport.noCache")} <span class="bd-seg-dot ready bd-seg-dot-done"></span>${t("secondSample.legendDone")}`;
        bodyEl.appendChild(legend);
        listEl.classList.remove("hidden");

        const checkboxes = [];
        let countEl = null;
        const refreshCount = () => {
            const checked = checkboxes.filter((c) => c.checked && !c.disabled).length;
            countEl.textContent = checked
                ? t("secondSample.selectedCount", { count: checked })
                : t("secondSample.noneSelected");
            okBtn.disabled = checked === 0;
        };

        const countRow = document.createElement("div");
        countRow.className = "bd-seg-export-count";
        bodyEl.appendChild(countRow);
        countEl = countRow;

        const okBtn = document.createElement("button");
        okBtn.type = "button";
        okBtn.className = "bd-btn bd-btn-primary";
        okBtn.textContent = t("secondSample.runBtn");
        okBtn.disabled = true;
        okBtn.onclick = () => {
            const indices = [];
            checkboxes.forEach((cb) => {
                if (cb.checked && !cb.disabled) indices.push(...cb._runIndices);
            });
            finish({ enabled: indices.length > 0, indices });
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "bd-btn";
        cancelBtn.textContent = t("dialog.cancel");
        cancelBtn.onclick = () => finish(null);
        actionsEl.appendChild(cancelBtn);
        actionsEl.appendChild(okBtn);

        // Fetch availability, then render rows (disabled where not second-sampleable)
        const status = await this._fetchSecondSampleStatus();
        const rows = (status && status.segments) || [];
        const avail = {};
        for (const r of rows) avail[r.index] = r;

        // 按「引用上段」关系把相邻连续片段合并为一个「段」:一段从「不引用上段」的
        // 片段(或首段)开始,后续「引用上段」的片段并入同一段。段号按时间线顺序 1/2/3…。
        // 勾选一段 = 二采该整段连续区间(不可选子集);某段内有片段不可二采则整段置灰。
        const runs = [];
        let curRun = null;
        for (let i = 0; i < segments.length; i++) {
            const refPrev = !!(avail[i] || {}).continuityFromPrev;
            if (!curRun || !refPrev) {
                curRun = { start: i, end: i, indices: [i] };
                runs.push(curRun);
            } else {
                curRun.end = i;
                curRun.indices.push(i);
            }
        }
        runs.forEach((run, ri) => {
            const runNo = ri + 1;
            const runInfos = run.indices.map((i) => avail[i] || {});
            const sampleable = runInfos.length > 0 && runInfos.every((info) => info.canSecondSample);
            const rangeLabel = run.start === run.end
                ? t("secondSample.segmentOne", { n: run.start + 1 })
                : t("secondSample.segmentRange", { a: run.start + 1, b: run.end + 1 });
            const row = document.createElement("div");
            row.className = "bd-modal-item bd-seg-export-item" + (sampleable ? "" : " disabled");
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.className = "bd-seg-export-cb";
            cb._runIndices = run.indices.slice();
            cb.disabled = !sampleable;
            cb.checked = sampleable && run.indices.every((i) => cfg.indices.includes(i));
            cb.onchange = refreshCount;
            row.prepend(cb);
            const nameSpan = document.createElement("span");
            nameSpan.className = "bd-seg-export-name";
            nameSpan.textContent = t("secondSample.runLabel", { no: runNo, range: rangeLabel });
            row.appendChild(nameSpan);
            const badgeSpan = document.createElement("span");
            badgeSpan.className = "bd-seg-export-badges";
            badgeSpan.innerHTML = this._secondSampleDots(runInfos) + this._secondSampleRunSummary(runInfos);
            row.appendChild(badgeSpan);
            row.onclick = (e) => {
                if (e.target === cb) return;
                if (!cb.disabled) {
                    cb.checked = !cb.checked;
                    cb.onchange && cb.onchange();
                }
            };
            listEl.appendChild(row);
            checkboxes.push(cb);
        });
        refreshCount();

        const keyHandler = (e) => {
            if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                finish(null);
            }
        };
        window.addEventListener("keydown", keyHandler, true);
        this._modalKeyHandler = keyHandler;

        overlay.onclick = (e) => { if (e.target === overlay) finish(null); };
        panel.onclick = (e) => e.stopPropagation();
        overlay.appendChild(panel);
        document.body.appendChild(overlay);
        this._modalEl = overlay;
    },
    // 优雅的缓存状态展示:每段一个点(悬停看明细),整段再加一行摘要,
    // 取代之前一长串文字胶囊。
    _secondSampleBadgeText(info) {
        if (!info) return t("segmentExport.noCache");
        const parts = [];
        parts.push(info.hasLatent ? t("secondSample.badgeHasLatent") : t("secondSample.badgeNoLatent"));
        parts.push(info.hasTextCond ? t("secondSample.badgeHasTextCond") : t("secondSample.badgeNoTextCond"));
        if (info.hasSecondLatent) parts.push(t("secondSample.legendDone"));
        if (!info.canSecondSample) parts.push(t("secondSample.badgeCannot"));
        return parts.join(" · ");
    },
    _secondSampleDots(infos) {
        const dots = (infos || []).map((info) => {
            let cls = "empty";
            if (info) {
                if (info.canSecondSample) cls = "ready";
                else if (info.hasLatent || info.hasTextCond || info.hasSecondLatent) cls = "partial";
            }
            const done = info && info.hasSecondLatent ? " bd-seg-dot-done" : "";
            const title = this._secondSampleBadgeText(info).replace(/"/g, "&quot;");
            return `<span class="bd-seg-dot ${cls}${done}" title="${title}"></span>`;
        }).join("");
        return `<span class="bd-seg-dots">${dots}</span>`;
    },
    _secondSampleRunSummary(infos) {
        const total = (infos || []).length;
        const ready = (infos || []).filter((i) => i && i.canSecondSample).length;
        const done = (infos || []).filter((i) => i && i.hasSecondLatent).length;
        let txt;
        if (ready === total) txt = done === total
            ? t("secondSample.summaryAllDone", { total })
            : t("secondSample.summaryAllReady", { total });
        else if (ready === 0) txt = t("secondSample.summaryNoneReady", { total });
        else txt = t("secondSample.summaryPartial", { ready, total });
        return `<span class="bd-seg-export-summary">${txt}</span>`;
    },
    resolveSecondSample(val) {
        if (!this.timeline.output) this.timeline.output = {};
        this.timeline.output.secondSample = {
            enabled: !!val.enabled,
            indices: [...(val.indices || [])].sort((a, b) => a - b),
        };
        this.commit(false, { syncTimeline: true });
        this.flushTimelineSync();
        this.scheduleRender();
        try {
            if (typeof app?.queuePrompt === "function") {
                const queued = app.queuePrompt();
                const clearAfter = () => {
                    this._clearSecondSampleFlag();
                    this._secondSampleToast(t("secondSample.queued"));
                };
                if (queued && typeof queued.then === "function") {
                    queued.then(clearAfter, clearAfter);
                } else {
                    setTimeout(clearAfter, 0);
                }
            } else {
                this._secondSampleToast(t("secondSample.needRun"));
            }
        } catch (e) {
            console.warn("[MiniMax] second sample queue prompt failed", e);
            this._secondSampleToast(t("secondSample.needRun"));
        }
    },
    _clearSecondSampleFlag() {
        const cfg = this.timeline.output?.secondSample;
        if (!cfg) return;
        this.timeline.output.secondSample = { ...cfg, enabled: false };
        try {
            this.commit(false, { syncTimeline: true });
        } catch (e) {
            /* best-effort */
        }
        this.scheduleRender();
    },
    _secondSampleToast(msg) {
        let el = this.root.querySelector("[data-r='second-sample-toast']");
        if (!el) {
            el = document.createElement("div");
            el.setAttribute("data-r", "second-sample-toast");
            el.className = "bd-seg-export-toast";
            document.body.appendChild(el);
        }
        el.textContent = msg;
        el.classList.add("show");
        clearTimeout(this._secondSampleToastTimer);
        this._secondSampleToastTimer = setTimeout(() => el.classList.remove("show"), 2600);
    },
    _segmentExportBadge(info) {
        if (!info) return `<span class="bd-seg-export-badge muted" title="${t("segmentExport.noCache")}">${t("segmentExport.noCache")}</span>`;
        const bits = [];
        if (info.hasClip) bits.push(`<span class="bd-seg-export-badge ok" title="${t("segmentExport.hasClip")}">${t("segmentExport.hasClip")}</span>`);
        if (info.hasFrames) bits.push(`<span class="bd-seg-export-badge ok" title="${t("segmentExport.hasFrames")}">${t("segmentExport.hasFrames")}</span>`);
        if (info.hasLatent) bits.push(`<span class="bd-seg-export-badge warn" title="${t("segmentExport.hasLatent")}">${t("segmentExport.hasLatent")}</span>`);
        if (!bits.length) bits.push(`<span class="bd-seg-export-badge muted" title="${t("segmentExport.noCache")}">${t("segmentExport.noCache")}</span>`);
        return bits.join(" ");
    },
    resolveSegmentExport(val) {
        // Persist into timeline.output.segmentExport and re-sync the widget so the
        // backend's _parse_segment_export sees it on the next run.
        if (!this.timeline.output) this.timeline.output = {};
        this.timeline.output.segmentExport = {
            enabled: !!val.enabled,
            mode: val.mode === "continuous" ? "continuous" : "piecewise",
            source: val.source === "2nd" ? "2nd" : "1st",
            indices: [...(val.indices || [])].sort((a, b) => a - b),
        };
        this.commit(false, { syncTimeline: true });
        this.flushTimelineSync();
        this.scheduleRender();
        // Push the Director node into ComfyUI's queue so the export runs with the
        // models loaded (latent-only segments need the VAE). The queuePrompt patch
        // flushes every Director's timeline before the prompt is built, so
        // segmentExport reaches the backend. Best-effort: never throw in the picker.
        try {
            if (typeof app?.queuePrompt === "function") {
                // IMPORTANT: do NOT clear the one-shot `enabled` flag until the
                // prompt has actually been built & queued. queuePrompt may be async
                // (it awaits internal validation/prompt collection), so clearing
                // synchronously right after the call races the prompt build and can
                // reset `segmentExport.enabled` to false *before* the backend reads
                // it — which is why exports silently produced nothing. Await the
                // queue, then clear on the next tick so the prompt is already sent.
                const queued = app.queuePrompt();
                const clearAfter = () => {
                    // Re-flush with the flag off so the in-memory timeline widget
                    // reflects the cleared state for any subsequent run.
                    this._clearSegmentExportFlag();
                    this._toast ? this._toast(t("segmentExport.queued")) : this._segExportToast(t("segmentExport.queued"));
                };
                if (queued && typeof queued.then === "function") {
                    queued.then(clearAfter, clearAfter);
                } else {
                    // Synchronous queue: defer clearing to a macrotask so the
                    // synchronous prompt build in queuePrompt has fully completed.
                    setTimeout(clearAfter, 0);
                }
            } else {
                this._segExportToast(t("segmentExport.runToExport"));
            }
        } catch (e) {
            console.warn("[MiniMax] segment export queue prompt failed", e);
            this._segExportToast(t("segmentExport.runToExport"));
        }
    },
    _clearSegmentExportFlag() {
        const cfg = this.timeline.output?.segmentExport;
        if (!cfg) return;
        this.timeline.output.segmentExport = { ...cfg, enabled: false };
        // Persist the cleared state into the timeline widget. This is only ever
        // called *after* the export prompt has been queued (see resolveSegmentExport),
        // so it can no longer race the prompt build.
        try {
            this.commit(false, { syncTimeline: true });
        } catch (e) {
            /* best-effort */
        }
        this.scheduleRender();
    },
    /** Minimal inline toast so the picker gives feedback without other deps. */
    _segExportToast(msg) {
        let el = this.root.querySelector("[data-r='seg-export-toast']");
        if (!el) {
            el = document.createElement("div");
            el.setAttribute("data-r", "seg-export-toast");
            el.className = "bd-seg-export-toast";
            this.root.appendChild(el);
        }
        el.textContent = msg;
        el.classList.add("show");
        clearTimeout(this._segExportToastTimer);
        this._segExportToastTimer = setTimeout(() => el.classList.remove("show"), 4000);
    },
    _runSelectionPayload() {
        // Never leak video-mode「选择运行」into i2v/batch (or vice versa).
        if (!this.supportsRunSelect() || !this.timeline.runSelectEnabled) {
            return { runSelectEnabled: false, runSelection: [] };
        }
        this.normalizeRunSelection();
        return {
            runSelectEnabled: true,
            runSelection: [...(this.timeline.runSelection || [])],
        };
    }
};
