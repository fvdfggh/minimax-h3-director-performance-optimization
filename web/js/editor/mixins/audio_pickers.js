/** audio_pickers mixin for the Director editor (audio_pickers).
 *
 * 「提取音频」: pick segments → the backend writes a standalone WAV plus an audio
 * latent per segment into ``<node dir>/audio_extract/`` → the「音频」tab (and the
 * result view here) plays them back.
 *
 * Unlike「分段导出」this never queues a prompt: everything it needs is already on
 * disk, so the extraction is a plain HTTP call and the artefacts are appended —
 * re-running appends another take instead of replacing one.
 */

import { invalidateR2vClipProbe } from "../../minimax_image_batch.js";
import { t } from "../../minimax_i18n.js";
import { api } from "../../../../scripts/api.js";

export const audio_pickersMixin = {
    /** Cards the picker offers (same trimming the other pickers use). */
    _audioExtractSegments() {
        const all = this.timeline.segments || [];
        const n = typeof this.getRunnableSegmentCount === "function"
            ? this.getRunnableSegmentCount()
            : all.length;
        return all.slice(0, n);
    },

    /** Binding key of every card, in timeline order (mirrors the backend). */
    _audioExtractSegIds() {
        return (this.timeline.segments || []).map((s, i) => String(s?.id || `@${i}`));
    },

    _audioExtractBasePayload(source) {
        return {
            node_id: String(this.node?.id ?? ""),
            timeline_data: this.buildTimelinePayload(),
            task_type: this.globalTask?.value || this.taskTypeWidget?.value || "",
            global_prompt: this.timeline.global?.prompt || "",
            total_frames: typeof this.getTotalFrames === "function" ? this.getTotalFrames() : 0,
            frame_rate: typeof this.getFrameRate === "function" ? this.getFrameRate() : 24,
            width: this.timeline.output?.width || 864,
            height: this.timeline.output?.height || 480,
            ref_max_size: this.refMaxWidget?.value || this.timeline.output?.longEdge || 864,
            workflow_name: this.getWorkflowId?.() || "",
            // Every entry is bound to a segment id, so the backend can re-derive
            // positions (move with the card) and drop lost ones (delete).
            seg_ids: this._audioExtractSegIds(),
            source: source === "2nd" ? "2nd" : "1st",
        };
    },

    async _fetchAudioExtractStatus(source) {
        try {
            const resp = await api.fetchApi("/minimax/director_opt/audio_extract_status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(this._audioExtractBasePayload(source)),
            });
            return await resp.json();
        } catch (e) {
            console.error("[MiniMax] audio extract status failed", e);
            return { segments: [], error: String(e) };
        }
    },

    audioExtractFileUrl(entryId) {
        const q = [
            `node_id=${encodeURIComponent(String(this.node?.id ?? ""))}`,
            `entry=${encodeURIComponent(String(entryId || ""))}`,
        ];
        const wf = this.getWorkflowId?.() || "";
        if (wf) q.push(`workflow_name=${encodeURIComponent(wf)}`);
        return `/minimax/director_opt/audio_extract_file?${q.join("&")}`;
    },

    /** Entries of one card (``index`` omitted → every card). Used by the 音频 tab. */
    async listAudioExtracts(index) {
        try {
            const resp = await api.fetchApi("/minimax/director_opt/audio_extract_list", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(this.node?.id ?? ""),
                    timeline_data: this.buildTimelinePayload(),
                    workflow_name: this.getWorkflowId?.() || "",
                    seg_ids: this._audioExtractSegIds(),
                    index,
                }),
            });
            const data = await resp.json();
            return (data && data.entries) || [];
        } catch (e) {
            console.error("[MiniMax] audio extract list failed", e);
            return [];
        }
    },

    async removeAudioExtract(entryId) {
        try {
            await api.fetchApi("/minimax/director_opt/audio_extract_remove", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(this.node?.id ?? ""),
                    workflow_name: this.getWorkflowId?.() || "",
                    entry_id: String(entryId || ""),
                }),
            });
            return true;
        } catch (e) {
            console.error("[MiniMax] audio extract remove failed", e);
            return false;
        }
    },

    _audioExtractBadge(info) {
        const muted = (key) =>
            `<span class="bd-seg-export-badge muted" title="${t(key)}">${t(key)}</span>`;
        if (!info) return muted("audioExtract.noCache");
        const bits = [];
        const ok = (key) =>
            `<span class="bd-seg-export-badge ok" title="${t(key)}">${t(key)}</span>`;
        const warn = (key) =>
            `<span class="bd-seg-export-badge warn" title="${t(key)}">${t(key)}</span>`;
        if (info.hasWave) bits.push(ok("audioExtract.hasWave"));
        if (info.hasClipAudio) bits.push(ok("audioExtract.hasClipAudio"));
        if (info.latentAvailable) bits.push(ok("audioExtract.hasLatent"));
        if (info.needsVae) bits.push(warn("audioExtract.needsVae"));
        if (!bits.length) bits.push(muted("audioExtract.noCache"));
        return bits.join(" ");
    },

    /** 同一张卡片只能保留一条：勾上新的就把其余条目取消勾选（不重渲染，免得打断播放）。 */
    _syncRetainBoxes(root, pickedId, on) {
        if (!on || !root) return;
        root.querySelectorAll(".bd-audio-retain-cb").forEach((cb) => {
            const row = cb.closest(".bd-audio-item");
            if (row && row.dataset.audioEntry !== String(pickedId)) cb.checked = false;
        });
    },

    /** 这张卡片当前保留的条目 id（"" = 不保留）。 */
    retainedAudioId(index) {
        return String(this.timeline.segments?.[index]?.retainAudioId || "");
    },

    /** 勾选/取消「保留音频」。写进段对象本身，所以随卡片移动、随删除一起走。 */
    setRetainAudio(index, entryId, on) {
        const seg = this.timeline.segments?.[index];
        if (!seg) return;
        seg.retainAudioId = on ? String(entryId || "") : "";
        try {
            this.commit?.(false, { syncTimeline: true });
        } catch (e) {
            /* best-effort */
        }
        try {
            this.flushTimelineSync?.();
        } catch (e) {
            /* best-effort */
        }
    },

    /** One playable row: <audio> + meta + delete. Shared by the result view. */
    _audioExtractRow(entry, { onDeleted, onRetain } = {}) {
        const row = document.createElement("div");
        row.className = "bd-audio-item";
        const dur = (entry.duration_s || 0).toFixed(2);
        const metaKey = entry.has_latent ? "audioExtract.metaLatent" : "audioExtract.meta";
        const meta = t(metaKey, {
            dur,
            sr: entry.sample_rate || "-",
            ch: entry.channels || "-",
        });
        const srcLabel = entry.src_kind === "clip"
            ? t("audioExtract.srcClip")
            : t("audioExtract.srcWave");

        const head = document.createElement("div");
        head.className = "bd-audio-meta";
        head.innerHTML = `<span class="bd-audio-src">${srcLabel}</span><span>${meta}</span>`;

        // 保留音频：把这条提取音频设为该片段的固定音轨。同一张卡片互斥（只有一
        // 个字段），勾上新的会自动顶掉旧的。
        const retainWrap = document.createElement("label");
        retainWrap.className = "bd-audio-retain";
        retainWrap.title = t("audioExtract.retainHint");
        const retainCb = document.createElement("input");
        retainCb.type = "checkbox";
        retainCb.className = "bd-audio-retain-cb";
        retainCb.checked = this.retainedAudioId(entry.index) === String(entry.id);
        retainCb.onchange = () => {
            this.setRetainAudio(entry.index, entry.id, retainCb.checked);
            if (typeof onRetain === "function") onRetain(entry.id, retainCb.checked);
        };
        retainWrap.appendChild(retainCb);
        const retainText = document.createElement("span");
        retainText.textContent = t("audioExtract.retain");
        retainWrap.appendChild(retainText);
        head.appendChild(retainWrap);

        const player = document.createElement("audio");
        player.controls = true;
        player.preload = "metadata";
        player.src = this.audioExtractFileUrl(entry.id);

        const del = document.createElement("button");
        del.type = "button";
        del.className = "bd-btn bd-audio-del";
        del.textContent = t("audioExtract.delete");
        del.onclick = async (e) => {
            e.stopPropagation();
            if (!window.confirm(t("audioExtract.deleteConfirm"))) return;
            del.disabled = true;
            const ok = await this.removeAudioExtract(entry.id);
            if (ok) {
                invalidateR2vClipProbe(null);
                row.remove();
                if (typeof onDeleted === "function") onDeleted();
            } else {
                del.disabled = false;
            }
        };

        row.appendChild(head);
        row.appendChild(player);
        row.appendChild(del);
        return row;
    },

    _audioExtractToast(msg) {
        let el = this.root.querySelector("[data-r='audio-extract-toast']");
        if (!el) {
            el = document.createElement("div");
            el.setAttribute("data-r", "audio-extract-toast");
            el.className = "bd-seg-export-toast";
            this.root.appendChild(el);
        }
        el.textContent = msg;
        el.classList.add("show");
        clearTimeout(this._audioExtractToastTimer);
        this._audioExtractToastTimer = setTimeout(() => el.classList.remove("show"), 4000);
    },

    async openAudioExtractPicker() {
        this._closeBdModal();
        const segments = this._audioExtractSegments();
        if (!segments.length) return;

        const overlay = document.createElement("div");
        overlay.className = "bd-modal-overlay bd-modal-overlay-fixed";
        const panel = document.createElement("div");
        panel.className = "bd-modal bd-modal-wide";
        panel.innerHTML = `
            <div class="bd-modal-title"></div>
            <div class="bd-modal-body"></div>
            <div class="bd-modal-list"></div>
            <div class="bd-modal-actions"></div>`;
        panel.querySelector(".bd-modal-title").textContent = t("audioExtract.title");
        const bodyEl = panel.querySelector(".bd-modal-body");
        const listEl = panel.querySelector(".bd-modal-list");
        const actionsEl = panel.querySelector(".bd-modal-actions");

        const finish = () => this._closeBdModal();

        // Cache source — one pass only, exactly like「分段导出」.
        const currentSource = () =>
            sourceWrap.querySelector('input[name="audio-extract-source"]:checked')?.value === "2nd"
                ? "2nd"
                : "1st";
        const sourceWrap = document.createElement("div");
        sourceWrap.className = "bd-seg-export-mode";
        sourceWrap.innerHTML = `<span class="bd-seg-export-mode-label">${t("audioExtract.source")}</span>
            <label><input type="radio" name="audio-extract-source" value="1st" checked> ${t("audioExtract.sourceFirst")}</label>
            <label><input type="radio" name="audio-extract-source" value="2nd"> ${t("audioExtract.sourceSecond")}</label>`;
        const sourceHint = document.createElement("div");
        sourceHint.className = "bd-seg-export-hint";
        const syncSourceHint = () => {
            sourceHint.textContent = t(
                currentSource() === "2nd"
                    ? "audioExtract.sourceHintSecond"
                    : "audioExtract.sourceHintFirst"
            );
        };
        bodyEl.appendChild(sourceWrap);
        bodyEl.appendChild(sourceHint);

        const hint = document.createElement("div");
        hint.className = "bd-seg-export-hint";
        hint.textContent = t("audioExtract.desc");
        bodyEl.appendChild(hint);
        listEl.classList.remove("hidden");

        const checkboxes = [];
        let countEl = null;
        const refreshCount = () => {
            const checked = checkboxes.filter((c) => c.checked && !c.disabled).length;
            countEl.textContent = checked
                ? t("audioExtract.count", { n: checked })
                : t("audioExtract.empty");
            okBtn.disabled = checked === 0;
        };
        const countRow = document.createElement("div");
        countRow.className = "bd-seg-export-count";
        bodyEl.appendChild(countRow);
        countEl = countRow;

        const okBtn = document.createElement("button");
        okBtn.type = "button";
        okBtn.className = "bd-btn bd-btn-primary";
        okBtn.textContent = t("audioExtract.extract");
        okBtn.onclick = () => {
            const indices = [];
            checkboxes.forEach((cb, i) => {
                if (cb.checked && !cb.disabled) indices.push(i);
            });
            if (!indices.length) return;
            void this._runAudioExtract(indices, currentSource(), {
                panel, listEl, bodyEl, okBtn,
            });
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "bd-btn";
        cancelBtn.textContent = t("dialog.cancel");
        cancelBtn.onclick = finish;
        actionsEl.appendChild(cancelBtn);
        actionsEl.appendChild(okBtn);

        const renderRows = async () => {
            const source = currentSource();
            syncSourceHint();
            listEl.textContent = "";
            checkboxes.length = 0;
            listEl.classList.add("bd-loading");
            okBtn.disabled = true;

            const status = await this._fetchAudioExtractStatus(source);
            const rows = (status && status.segments) || [];
            const avail = {};
            for (const r of rows) avail[r.index] = r;
            for (let i = 0; i < segments.length; i++) {
                const info = avail[i] || {};
                const extractable = !!info.extractable;
                const row = document.createElement("div");
                row.className = "bd-modal-item bd-seg-export-item" + (extractable ? "" : " disabled");
                row.innerHTML = `<span class="bd-seg-export-name">#${i + 1}</span><span class="bd-seg-export-badges">${this._audioExtractBadge(info)}</span>`;
                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.className = "bd-seg-export-cb";
                cb.disabled = !extractable;
                cb.checked = extractable;
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
            }
            listEl.classList.remove("bd-loading");
            refreshCount();
        };

        sourceWrap.querySelectorAll('input[name="audio-extract-source"]').forEach((r) => {
            r.onchange = () => void renderRows();
        });
        await renderRows();

        const keyHandler = (e) => {
            if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                finish();
            }
        };
        window.addEventListener("keydown", keyHandler, true);
        this._modalKeyHandler = keyHandler;
        overlay.onclick = (e) => { if (e.target === overlay) finish(); };
        panel.onclick = (e) => e.stopPropagation();
        overlay.appendChild(panel);
        document.body.appendChild(overlay);
        this._modalEl = overlay;
    },

    async _runAudioExtract(indices, source, { listEl, bodyEl, okBtn }) {
        okBtn.disabled = true;
        okBtn.textContent = t("audioExtract.extracting");
        listEl.classList.add("bd-loading");
        try {
            const resp = await api.fetchApi("/minimax/director_opt/audio_extract", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...this._audioExtractBasePayload(source),
                    indices,
                }),
            });
            const data = await resp.json();
            if (data && data.error) throw new Error(data.error);
            const entries = (data && data.entries) || [];
            const skipped = (data && data.skipped) || [];
            // 音频 tab 的条目探测作废，否则新提取的音频最长要等一个 TTL 才出现。
            invalidateR2vClipProbe(null);
            this._audioExtractToast(t("audioExtract.done", { n: entries.length }));
            this._renderAudioExtractResult(
                { listEl, bodyEl, okBtn },
                entries,
                skipped.length,
            );
        } catch (e) {
            okBtn.disabled = false;
            okBtn.textContent = t("audioExtract.extract");
            listEl.classList.remove("bd-loading");
            this._audioExtractToast(t("audioExtract.failed", { msg: String(e?.message || e) }));
        }
    },

    _renderAudioExtractResult({ listEl, bodyEl, okBtn }, entries, skippedCount) {
        listEl.classList.remove("bd-loading");
        listEl.textContent = "";
        okBtn.textContent = t("audioExtract.extract");
        okBtn.disabled = true;

        bodyEl.textContent = "";
        const title = document.createElement("div");
        title.className = "bd-seg-export-hint";
        title.textContent = t("audioExtract.resultTitle");
        const hint = document.createElement("div");
        hint.className = "bd-seg-export-hint";
        hint.textContent = t("audioExtract.resultHint");
        bodyEl.appendChild(title);
        bodyEl.appendChild(hint);
        if (skippedCount > 0) {
            const skip = document.createElement("div");
            skip.className = "bd-seg-export-hint";
            skip.textContent = t("audioExtract.skipped", { n: skippedCount });
            bodyEl.appendChild(skip);
        }

        if (!entries.length) {
            listEl.textContent = t("audioExtract.noCache");
            return;
        }
        const syncRetain = (pickedId, on) => this._syncRetainBoxes(listEl, pickedId, on);
        for (const entry of entries) {
            const row = this._audioExtractRow(entry, { onRetain: syncRetain });
            row.dataset.audioEntry = String(entry.id);
            listEl.appendChild(row);
        }
    },
};
