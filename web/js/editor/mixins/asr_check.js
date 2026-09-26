/** asr_check mixin — 「音频有效性校验」按钮的选片段弹窗。
 *
 * 校验本身在后端（``director/routes_asr.py``）：前端只负责挑片段并把**当前**时间轴
 * 发过去，后端重建 plan、读取这些片段已缓存的音轨、用此刻提示词里的台词块打分。
 * 整个过程不生成任何东西，所以核对一次不会重跑视频。
 *
 * 打开弹窗时先问一次 ``/asr_check_status``：没有音轨缓存的片段直接置灰不可选，
 * 免得「选了才发现跑不了」。探测与真正校验走的是后端同一个读取函数，不会打架。
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

    /** ASR 模型口是否接了线：``true`` / ``false`` / ``null``（无法判断）。
     *
     * 用来把「模型口空着」和「接了但后端还没句柄」分开——两者都要再跑一次节点，
     * 但前者要去接线，后者只需排队运行。
     */
    _asrModelSocketLinked() {
        const inputs = this.node?.inputs;
        if (!Array.isArray(inputs)) return null;
        const slot = inputs.find((i) => i?.name === "asr_model");
        if (!slot) return null;
        return slot.link != null;
    },

    /** 两个 ASR 接口的公共字段（与「提取音频」「分段导出」一致）。 */
    _asrCheckPayload(source) {
        return {
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
            source: source === "2nd" ? "2nd" : "1st",
        };
    },

    /** 哪些片段在该缓存来源下有可读音轨，以及 ASR 模型是否已就绪。
     *
     * 探测不到（路由没生效 / 时间轴重建失败 / 网络异常）时返回 ``error``，调用方
     * 会「失败即放开」——宁可让用户自己挑，也不要把弹窗变成点不动的死界面。
     */
    async _fetchAsrCheckStatus(source) {
        const empty = {
            hasAudio: {}, hasModel: false, autoModel: false, withAudio: 0,
            knownNodes: [], modelState: "",
        };
        let resp;
        try {
            resp = await api.fetchApi("/minimax/director_opt/asr_check_status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(this._asrCheckPayload(source)),
            });
        } catch (e) {
            console.error("[MiniMax] ASR 状态探测失败", e);
            return { ...empty, error: t("asr.statusUnreachable", { detail: String(e) }) };
        }
        if (resp.status === 404) {
            // 新增的后端路由必须重启 ComfyUI 才会生效，这是最常见的一次性原因。
            return { ...empty, error: t("asr.statusNoRoute") };
        }
        let data = null;
        try {
            data = await resp.json();
        } catch (e) {
            return { ...empty, error: t("asr.statusUnreachable", { detail: `HTTP ${resp.status}` }) };
        }
        if (!resp.ok || data?.error) {
            return { ...empty, error: data?.error || `HTTP ${resp.status}` };
        }
        const hasAudio = {};
        for (const row of data.segments || []) {
            hasAudio[Number(row.index)] = !!row.hasAudio;
        }
        return {
            hasAudio,
            hasModel: !!data.hasModel,
            // 后端将用「自己构造的默认句柄」而不是你接线的配置。
            autoModel: !!data.autoModel,
            withAudio: Number(data.withAudio) || 0,
            knownNodes: (Array.isArray(data.knownNodes) ? data.knownNodes : []).map(String),
            // ready / ran-without / never-ran
            modelState: String(data.modelState || ""),
        };
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

        // 缓存来源（一采 / 二采）——与「提取音频」「分段导出」同一套写法。
        const currentSource = () =>
            sourceWrap.querySelector('input[name="asr-check-source"]:checked')?.value === "2nd"
                ? "2nd"
                : "1st";
        const sourceWrap = document.createElement("div");
        sourceWrap.className = "bd-seg-export-mode";
        sourceWrap.innerHTML = `<span class="bd-seg-export-mode-label">${t("asr.source")}</span>
            <label><input type="radio" name="asr-check-source" value="1st" checked> ${t("asr.sourceFirst")}</label>
            <label><input type="radio" name="asr-check-source" value="2nd"> ${t("asr.sourceSecond")}</label>`;
        const sourceHint = document.createElement("div");
        sourceHint.className = "bd-seg-export-hint";
        const syncSourceHint = () => {
            sourceHint.textContent = t(
                currentSource() === "2nd" ? "asr.sourceHintSecond" : "asr.sourceHintFirst",
            );
        };
        bodyEl.appendChild(sourceWrap);
        bodyEl.appendChild(sourceHint);

        const statusLine = document.createElement("div");
        statusLine.className = "bd-seg-export-hint";
        bodyEl.appendChild(statusLine);

        // 模型没就绪时**不禁用**按钮：点下去后端会给出「接线模型口并运行一次节点」
        // 的明确提示，比一个灰按钮可解释得多。这一行只是提前告知。
        const modelLine = document.createElement("div");
        modelLine.className = "bd-seg-export-hint warn";
        bodyEl.appendChild(modelLine);

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
        okBtn.disabled = true;

        let availability = { hasAudio: {}, hasModel: false, withAudio: 0, error: "" };
        const checkboxes = [];
        /** 勾选状态跨「切来源」保留：能用就默认勾上，用不了强制取消。 */
        const checkedByIndex = new Map();

        const refreshCount = () => {
            const usable = checkboxes.filter((cb) => !cb.disabled);
            const checked = usable.filter((cb) => cb.checked).length;
            countEl.textContent = checked
                ? t("asr.selectedCount", { count: checked })
                : t("asr.noneSelected");
            // 只有「没勾任何可校验片段」才禁用——与「提取音频」「分段导出」一致。
            okBtn.disabled = checked === 0;
            okBtn.title = okBtn.disabled ? t("asr.disabledHint") : "";
        };

        const renderRows = () => {
            listEl.innerHTML = "";
            checkboxes.length = 0;
            segments.forEach((seg, i) => {
                // 探测失败时放开全部（fail-open）：探测不到不等于没有缓存。
                const hasAudio = availability.error ? true : availability.hasAudio[i] === true;
                const effective = this._asrEffectivePrompt(seg);
                const row = document.createElement("div");
                row.className = "bd-modal-item bd-seg-export-item" + (hasAudio ? "" : " disabled");

                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.className = "bd-seg-export-cb";
                cb.disabled = !hasAudio;
                if (!hasAudio) checkedByIndex.set(i, false);
                else if (!checkedByIndex.has(i)) checkedByIndex.set(i, true);
                cb.checked = checkedByIndex.get(i) === true;
                cb.onchange = () => {
                    checkedByIndex.set(i, cb.checked);
                    refreshCount();
                };
                row.prepend(cb);

                const nameSpan = document.createElement("span");
                nameSpan.className = "bd-seg-export-name";
                nameSpan.textContent = t("asr.segmentLabel", { n: i + 1 });
                row.appendChild(nameSpan);

                const badge = document.createElement("span");
                badge.className = "bd-seg-export-badge " + (hasAudio ? "ok" : "muted");
                badge.textContent = t(hasAudio ? "asr.hasAudio" : "asr.noAudio");
                row.appendChild(badge);

                const preview = document.createElement("span");
                preview.className = "bd-seg-export-badges";
                preview.textContent = effective
                    ? effective.replace(/\s+/g, " ").slice(0, 60)
                    : t("asr.noPrompt");
                preview.title = effective;
                row.appendChild(preview);

                row.onclick = (e) => {
                    if (e.target === cb || cb.disabled) return;
                    cb.checked = !cb.checked;
                    cb.onchange();
                };
                listEl.appendChild(row);
                checkboxes.push(cb);
            });
            refreshCount();
        };

        const applyStatus = () => {
            const failed = !!availability.error;
            statusLine.classList.toggle("warn", failed || !availability.withAudio);
            if (failed) {
                statusLine.textContent = t("asr.statusFailed", { detail: availability.error });
            } else if (!availability.withAudio) {
                statusLine.textContent = t("asr.noCacheAny");
            } else {
                statusLine.textContent = t("asr.statusCount", { count: availability.withAudio });
            }
            // 只有「探测成功且确实没有模型」才提示；探测失败时我们对模型状态一无所知。
            // 没有可用模型时给「怎么修」；用得上自动句柄时只说明「用的是默认参数」。
            const showModel = !failed && !availability.hasModel;
            const useAuto = !failed && !showModel && availability.autoModel
                && availability.modelState !== "ready";
            if (showModel) {
                // 四种情况对应四种修法，宁可啰嗦也要说准：
                //   接在别的节点 → 改接线；口空着 → 接线；跑过但没拿到 → 查加载器；没跑过 → 排队跑一次。
                const known = availability.knownNodes || [];
                const mine = String(this.node?.id ?? "");
                if (known.length && mine && !known.includes(mine)) {
                    modelLine.textContent = t("asr.modelOtherNode", {
                        nodes: known.join("、"),
                        mine,
                    });
                } else if (this._asrModelSocketLinked() === false) {
                    modelLine.textContent = t("asr.modelNotWired");
                } else if (availability.modelState === "ran-without") {
                    modelLine.textContent = t("asr.modelRanWithout");
                } else if (availability.modelState === "never-ran") {
                    modelLine.textContent = t("asr.modelNeverRan");
                } else {
                    modelLine.textContent = t("asr.modelLine");
                }
            } else if (useAuto) {
                modelLine.textContent = t("asr.modelAuto");
            }
            modelLine.classList.toggle("warn", showModel);
            modelLine.classList.toggle("hidden", !(showModel || useAuto));
        };

        const refresh = async () => {
            listEl.innerHTML = `<div class="bd-seg-export-hint">${t("asr.checking")}</div>`;
            okBtn.disabled = true;
            availability = await this._fetchAsrCheckStatus(currentSource());
            // 换来源等于重新挑一次：可用集合变了，之前的勾选没有意义。
            checkedByIndex.clear();
            applyStatus();
            renderRows();
        };

        okBtn.onclick = async () => {
            const indices = [];
            checkboxes.forEach((cb, i) => {
                if (cb.checked && !cb.disabled) indices.push(i);
            });
            const source = currentSource();
            finish();
            await this.runAsrCheck(indices, source);
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "bd-btn";
        cancelBtn.textContent = t("dialog.cancel");
        cancelBtn.onclick = () => finish();
        actionsEl.appendChild(cancelBtn);
        actionsEl.appendChild(okBtn);

        sourceWrap.addEventListener("change", () => {
            syncSourceHint();
            void refresh();
        });
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

        syncSourceHint();
        await refresh();
    },

    /** POST 一个 JSON 接口，**两边的响应都先当 JSON 解析**。
     *
     * 后端连错误体也是 ``{"success": false, "error": "…"}``，而且是非 ASCII 转义的，
     * 所以「非 2xx 就切原文」会把真正的说明变成一串 ``\uXXXX`` 再被截断——诊断信息
     * 恰恰全在被截掉的那一段里。只有在响应根本不是 JSON 时才退回原文。
     */
    async _postJson(url, payload) {
        try {
            const resp = await api.fetchApi(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const text = await resp.text();
            let data = null;
            try {
                data = text ? JSON.parse(text) : null;
            } catch {
                data = null;
            }
            if (data && typeof data === "object") {
                if (!resp.ok && !data.error) data.error = `HTTP ${resp.status}`;
                return data;
            }
            return { success: false, error: (text || `HTTP ${resp.status}`).slice(0, 600) };
        } catch (e) {
            console.error("[MiniMax] ASR request failed", e);
            return { success: false, error: String(e) };
        }
    },

    /** 最近一次校验结果，渲染到预览的「识别」页签里（``null`` = 还没跑过）。
     *
     * ``segments`` 是后端按时间窗归属好的**每段**明细：本段提示词台词 + 本段识别到
     * 的段落。整批报告（``text``）也在，但那是跨片段合并计分的结论，作为附属展示。
     */
    _asrReport: null,

    /** 把最近一次结果画进「识别」页签的容器（每张卡片显示**它自己**那一段）。 */
    renderAsrReportInto(body, index) {
        if (!body) return;
        body.innerHTML = "";
        const wrap = document.createElement("div");
        wrap.className = "bd-asr-report";
        const report = this._asrReport;
        if (!report) {
            wrap.textContent = t("asr.reportEmpty");
            body.appendChild(wrap);
            return;
        }
        const div = (cls, text) => {
            const el = document.createElement("div");
            el.className = cls;
            if (text != null) el.textContent = text;
            return el;
        };
        const sourceLabel = t(report.source === "2nd" ? "asr.sourceSecond" : "asr.sourceFirst");
        const detail = (report.segments || []).find((s) => Number(s.index) === Number(index));

        if (!detail) {
            // 这张卡片不在本次核对范围内：说清楚范围就好，别再把整批报告贴一遍。
            wrap.appendChild(div("bd-asr-scope", t("asr.reportMeta", {
                list: (report.indices || []).map((i) => i + 1).join("、") || "—",
                source: sourceLabel,
                time: new Date(report.at).toLocaleTimeString(),
            })));
            wrap.appendChild(div("bd-asr-none", t("asr.reportExcluded", { n: index + 1 })));
            body.appendChild(wrap);
            return;
        }

        const span = (detail.start != null && detail.end != null)
            ? t("asr.segMeta", {
                n: index + 1, source: sourceLabel,
                time: new Date(report.at).toLocaleTimeString(),
                start: detail.start.toFixed(2), end: detail.end.toFixed(2),
            })
            : t("asr.reportMeta", {
                list: String(index + 1), source: sourceLabel,
                time: new Date(report.at).toLocaleTimeString(),
            });
        wrap.appendChild(div("bd-asr-scope", span));

        const line = (spk, arrow, text, warn) => {
            const row = div(`bd-asr-line${warn ? " warn" : ""}`);
            row.appendChild(div("bd-asr-spk", arrow ? `${spk} → ${arrow}` : spk));
            row.appendChild(div("bd-asr-text", text));
            return row;
        };

        // 1) 本段提示词要求的台词
        wrap.appendChild(div("bd-asr-head", t("asr.segPrompt", { n: detail.prompt.length })));
        if (!detail.prompt.length) {
            wrap.appendChild(div("bd-asr-none", t("asr.segNoPrompt")));
        }
        for (const item of detail.prompt) {
            wrap.appendChild(line(`S${item.speaker}`, null, item.text, false));
        }

        // 2) 本段时间窗里识别到的内容（含「提示词里没有」的说话人）
        wrap.appendChild(div("bd-asr-head", t("asr.segHeard", { n: detail.heard.length })));
        if (!detail.heard.length) {
            wrap.appendChild(div("bd-asr-none", t("asr.segNoHeard")));
        }
        for (const item of detail.heard) {
            const unknown = item.speaker === "S00";
            const unmapped = !unknown && !(item.mapped > 0);
            const note = unknown
                ? t("asr.segUnknown")
                : (unmapped ? t("asr.segUnmapped") : "");
            wrap.appendChild(line(
                item.speaker,
                item.mapped > 0 ? `S${item.mapped}` : null,
                `${item.text}${note ? `　（${note}）` : ""}`,
                unknown || unmapped,
            ));
        }

        // 3) 本段该说、却没听到的说话人（「漏识」在这里一眼可见）
        const heardSpeakers = new Set(
            detail.heard.map((h) => Number(h.mapped)).filter((n) => n > 0),
        );
        const silent = detail.prompt
            .map((l) => `S${l.speaker}`)
            .filter((_, i, all) => all.indexOf(all[i]) === i)
            .filter((spk) => !heardSpeakers.has(Number(String(spk).slice(1))));
        if (silent.length) {
            wrap.appendChild(div("bd-asr-none warn", t("asr.segMissing", { list: silent.join("、") })));
        }

        // 4) 整批判定（说话人跨片段合并，所以数字只在这里给）
        wrap.appendChild(div("bd-asr-head", t("asr.batchHead")));
        wrap.appendChild(div(
            "bd-asr-batch",
            report.failed ? `${t("asr.reportFailed")}\n${report.text}` : report.text,
        ));
        body.appendChild(wrap);
    },

    /** 记录结果并切到「识别」页签，让结论就地出现（弹窗只是选片段用的）。
     *
     * 返回 ``true`` 表示报告真的出现在了 DOM 里；缩略图视图下预览区没渲染，
     * 调用方据此退回弹窗，避免结论无处可看。
     */
    _showAsrReport({ text, segments, indices, source, failed }) {
        this._asrReport = {
            text,
            segments: Array.isArray(segments) ? segments : [],
            indices,
            source,
            at: Date.now(),
            failed: !!failed,
        };
        this.r2vPreviewTab = "asr";
        try {
            if (this.isImageBatch?.()) this.renderImageBatchGroups?.();
        } catch (e) {
            console.error("[MiniMax] ASR 报告渲染失败", e);
        }
        return !!this.root?.querySelector?.(".bd-asr-report");
    },

    /** 把选中片段交给后端核对，并把报告弹出来。``source`` 为 "1st" / "2nd"。 */
    async runAsrCheck(indices, source = "1st") {
        const payload = { ...this._asrCheckPayload(source), indices };
        // 弹窗此刻已经关闭，而识别要跑一会儿：先给一个「开始了」的信号。
        this._audioExtractToast?.(t("asr.running"));
        const data = await this._postJson("/minimax/director_opt/asr_check", payload);
        const missing = Array.isArray(data?.missing) && data.missing.length
            ? `\n\n${t("asr.missing", { list: data.missing.join(", ") })}`
            : "";
        if (!data?.success) {
            // 失败也要留在页签里（弹窗一关就没了），同时弹一次——这一步是要人去改的。
            const error = data?.error || t("asr.failed");
            this._showAsrReport({ text: error, segments: [], indices, source, failed: true });
            await this.showBdMessage(t("asr.title"), error);
            return;
        }
        const text = `${String(data.report || "").trim()}${missing}`;
        const shown = this._showAsrReport({
            text,
            segments: data.segments,
            indices,
            source,
            failed: false,
        });
        if (!shown) await this.showBdMessage(t("asr.title"), text);
    },
};
