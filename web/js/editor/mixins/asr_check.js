/** asr_check mixin — 「音频有效性校验」按钮的选片段弹窗。
 *
 * 校验本身在后端（``director/routes_asr.py``）：前端只负责挑片段并把**当前**时间轴
 * 发过去，后端重建 plan、读取这些片段已缓存的音轨、用此刻提示词里的台词块打分。
 * 整个过程不生成任何东西，所以核对一次不会重跑视频。
 */

import { getStableWorkflowId } from "../../core/graph_refs.js";
import { t } from "../../minimax_i18n.js";
import { api } from "../../../../scripts/api.js";

export const asrCheckMixin = {
    /** 可校验的片段（与「分段导出」「二次采样」同一套口径）。 */
    _asrCheckSegments() {
        const n = Math.max(0, this.getRunnableSegmentCount?.() || 0);
        return (this.timeline?.segments || []).slice(0, n);
    },

    /** 选中片段的有效提示词 = 公共提示词 + 空行 + 本段提示词（与后端拼接口径一致）。 */
    _asrEffectivePrompt(seg) {
        const common = String(this.timeline?.global?.prompt || "").trim();
        const own = String(seg?.prompt || "").trim();
        if (common && own) return `${common}\n\n${own}`;
        return common || own;
    },

    async openAsrCheckPicker() {
        this._closeBdModal();
        const segments = this._asrCheckSegments();
        if (!segments.length) {
            await this.showBdMessage(t("asr.title"), t("asr.noSegments"));
            return;
        }

        const overlay = document.createElement("div");
        overlay.className = "bd-modal-overlay bd-modal-overlay-fixed";
        const panel = document.createElement("div");
        panel.className = "bd-modal bd-modal-wide";
        panel.innerHTML = `
            <div class="bd-modal-title"></div>
            <div class="bd-modal-body"></div>
            <div class="bd-modal-list"></div>
            <div class="bd-modal-actions"></div>`;
        panel.querySelector(".bd-modal-title").textContent = t("asr.title");
        const bodyEl = panel.querySelector(".bd-modal-body");
        const listEl = panel.querySelector(".bd-modal-list");
        const actionsEl = panel.querySelector(".bd-modal-actions");

        const finish = () => this._closeBdModal();

        const hint = document.createElement("div");
        hint.className = "bd-seg-export-hint";
        hint.textContent = t("asr.hint");
        bodyEl.appendChild(hint);

        const countEl = document.createElement("div");
        countEl.className = "bd-seg-export-count";
        bodyEl.appendChild(countEl);

        const okBtn = document.createElement("button");
        okBtn.type = "button";
        okBtn.className = "bd-btn bd-btn-primary";
        okBtn.textContent = t("asr.runBtn");

        const checkboxes = [];
        const refreshCount = () => {
            const checked = checkboxes.filter((cb) => cb.checked).length;
            countEl.textContent = checked
                ? t("asr.selectedCount", { count: checked })
                : t("asr.noneSelected");
            okBtn.disabled = checked === 0;
        };

        segments.forEach((seg, i) => {
            const effective = this._asrEffectivePrompt(seg);
            const row = document.createElement("div");
            row.className = "bd-modal-item bd-seg-export-item";
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.className = "bd-seg-export-cb";
            cb.checked = true;
            cb.onchange = refreshCount;
            row.prepend(cb);

            const nameSpan = document.createElement("span");
            nameSpan.className = "bd-seg-export-name";
            nameSpan.textContent = t("asr.segmentLabel", { n: i + 1 });
            row.appendChild(nameSpan);

            const preview = document.createElement("span");
            preview.className = "bd-seg-export-badges";
            preview.textContent = effective
                ? effective.replace(/\s+/g, " ").slice(0, 80)
                : t("asr.noPrompt");
            preview.title = effective;
            row.appendChild(preview);

            row.onclick = (e) => {
                if (e.target === cb) return;
                cb.checked = !cb.checked;
                cb.onchange();
            };
            listEl.appendChild(row);
            checkboxes.push(cb);
        });
        refreshCount();

        okBtn.onclick = async () => {
            const indices = [];
            checkboxes.forEach((cb, i) => {
                if (cb.checked) indices.push(i);
            });
            finish();
            await this.runAsrCheck(indices);
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "bd-btn";
        cancelBtn.textContent = t("dialog.cancel");
        cancelBtn.onclick = () => finish();
        actionsEl.appendChild(cancelBtn);
        actionsEl.appendChild(okBtn);

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

    /** 把选中片段交给后端核对，并把报告弹出来。 */
    async runAsrCheck(indices) {
        const payload = {
            node_id: String(this.node?.id ?? ""),
            timeline_data: this.buildTimelinePayload(),
            task_type: this.globalTask?.value || this.taskTypeWidget?.value || "",
            global_prompt: this.timeline?.global?.prompt || "",
            total_frames: this.getTotalFrames(),
            frame_rate: this.getFrameRate(),
            width: this.timeline?.output?.width || 864,
            height: this.timeline?.output?.height || 480,
            ref_max_size: this.refMaxWidget?.value || this.timeline?.output?.longEdge || 864,
            workflow_name: getStableWorkflowId(),
            indices,
        };
        let data = null;
        try {
            const resp = await api.fetchApi("/minimax/director_opt/asr_check", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            data = resp.ok ? await resp.json() : { error: (await resp.text()).slice(0, 300) };
        } catch (e) {
            console.error("[MiniMax] ASR check failed", e);
            data = { error: String(e) };
        }
        if (!data?.success) {
            await this.showBdMessage(t("asr.title"), data?.error || t("asr.failed"));
            return;
        }
        const missing = Array.isArray(data.missing) && data.missing.length
            ? `\n\n${t("asr.missing", { list: data.missing.join(", ") })}`
            : "";
        await this.showBdMessage(
            t("asr.title"),
            `${String(data.report || "").trim()}${missing}`,
        );
    },
};
