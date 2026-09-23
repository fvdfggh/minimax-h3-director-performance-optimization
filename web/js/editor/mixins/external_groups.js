/** external_groups mixin for the Director editor (external_groups).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { collectExternalGroupNodes, collectExternalGroupSpecs, imageRefFromPath } from "../../core/graph_refs.js";
import { flushFl2vPromptDraft, newFl2vShot, setFl2vToolbar, syncFl2vFromShots, updateFl2vDetailUI, updateFl2vToolbarBtns } from "../../minimax_fl2v.js";
import { defaultDurationSec, isPromptBatchTask, newBatchSegment, resolveTaskKey } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
import { flushBatchPromptInputs, normalizeImageBatchSegments, setR2vToolbar, updateR2vToolbarBtns } from "../../minimax_image_batch.js";
export const external_groupsMixin = {
    _inputLinkConnected(name) {
        const inp = this.node?.inputs?.find((i) => i?.name === name);
        return inp != null && inp.link != null;
    },
    hasExternalI2vGroups() {
        return this._inputLinkConnected("i2v_groups");
    },
    hasExternalR2vGroups() {
        return this._inputLinkConnected("r2v_groups");
    },
    updateExternalGroupsBanner() {
        const el = this.externalGroupsMsgEl || this.root?.querySelector('[data-r="external-groups-msg"]');
        if (!el) return;
        const i2v = this.hasExternalI2vGroups();
        const r2v = this.hasExternalR2vGroups();
        const active = i2v || r2v;
        el.classList.toggle("hidden", !active);
        this.root?.classList.toggle("bd-external-groups", active);
        // Refresh add/delete visibility when external wiring toggles.
        if (this.isR2vBatch?.()) setR2vToolbar(this, true);
        else if (this.isFl2vMode?.()) setFl2vToolbar(this, true);
        else {
            updateR2vToolbarBtns(this);
            updateFl2vToolbarBtns(this);
        }
        if (!active) {
            el.textContent = "";
            return;
        }
        const specs = collectExternalGroupSpecs(this);
        const n = specs?.length || 0;
        const base = i2v ? t("external.i2vActive") : t("external.r2vActive");
        const count = n > 0 ? ` (${t("external.groupCount", { n })})` : "";
        el.textContent = `${base}${count} ${t("external.durationHint")}`;
    },
    writeExternalGroupPrompt(segIndex, prompt) {
        if (!this.hasExternalI2vGroups?.() && !this.hasExternalR2vGroups?.()) return;
        const nodes = collectExternalGroupNodes(this);
        const node = nodes?.[segIndex];
        if (!node) return;
        const w = (node.widgets || []).find((x) => x?.name === "prompt");
        if (!w) return;
        const next = String(prompt ?? "");
        if (String(w.value ?? "") === next) return;
        // Avoid feedback loop: our widget callback triggers syncExternalGroupsTimeline.
        w._mmxSkipExternalSync = true;
        try {
            w.value = next;
            // ComfyUI V3 / custom widgets may need callback for persistence.
            w.callback?.(next);
        } finally {
            queueMicrotask(() => { w._mmxSkipExternalSync = false; });
        }
    },
    /** Mirror graph-wired Group count/duration into the Director timeline UI. */
    syncExternalGroupsTimeline() {
        this.updateExternalGroupsBanner();
        // Keep any in-progress Director textarea edits before rebuilding from graph.
        if (this.isImageBatch?.()) flushBatchPromptInputs(this);
        if (this.isFl2vMode?.()) flushFl2vPromptDraft(this);
        const specs = collectExternalGroupSpecs(this);
        if (!specs?.length) {
            this._externalGroupsSyncSig = null;
            return;
        }

        const mode = this.getDirectorMode?.() || this._directorMode;
        const taskKey = resolveTaskKey(this.getTaskKey?.() || this.taskTypeWidget?.value);
        const sig = JSON.stringify(specs.map((s) => [
            s.nodeId ?? "",
            Number(s.durationSec) || 0,
            s.prompt || "",
            s.firstImageFile || "",
            s.lastImageFile || "",
            (s.refImages || []).map((r) => `${r.index}:${r.imageFile || ""}`).join(","),
            (s.refVideos || []).map((r) => [
                r.index,
                r.videoFile || "",
                r.previewImageFile || "",
                r.previewImageUrl || "",
                r.pairedAudioFile || "",
                r.linked ? 1 : 0,
            ].join(":")).join(","),
            (s.refAudios || []).map((r) => `${r.index}:${r.audioFile || ""}`).join(","),
        ]));
        if (this._externalGroupsSyncSig === sig) return;
        this._externalGroupsSyncSig = sig;

        if (mode === "fl2v") {
            const prev = this.timeline.shots || [];
            const prevByNode = new Map(
                prev.filter((s) => s?.externalNodeId != null)
                    .map((s) => [String(s.externalNodeId), s]),
            );
            const allowIndexFallback = !prev.some((s) => s?.externalNodeId != null);
            this.timeline.shots = specs.map((spec, i) => {
                const matched = (spec.nodeId != null && prevByNode.get(String(spec.nodeId)))
                    || (allowIndexFallback ? (prev[i] || null) : null);
                // Same Group node → keep Director draft if widget briefly empty.
                // Different/new node at this index → never inherit another shot's prompt.
                const specPrompt = String(spec.prompt ?? "").trim();
                const prompt = specPrompt
                    || (matched ? String(matched.prompt || "").trim() : "");
                return newFl2vShot({
                    id: matched?.id,
                    durationSec: spec.durationSec ?? defaultDurationSec("fl2v"),
                    prompt,
                    externalNodeId: spec.nodeId ?? null,
                    // External graph is source of truth for media previews.
                    startImage: imageRefFromPath(spec.firstImageFile),
                    endImage: imageRefFromPath(spec.lastImageFile),
                });
            });
            syncFl2vFromShots(this);
            this.selectedIndex = Math.min(this.selectedIndex ?? 0, Math.max(0, this.timeline.shots.length - 1));
            updateFl2vDetailUI?.(this);
            this.scheduleRender?.();
            this.commit?.(false, { syncTimeline: true });
            this.updateVideoNameLabel?.();
            this.updateDomWidgetHeight?.();
            this.updateRunSelectUI?.();
            return;
        }

        if (mode === "prompt_batch" || mode === "image_batch" || isPromptBatchTask(taskKey)) {
            const prev = this.timeline.segments || [];
            const prevByNode = new Map(
                prev.filter((s) => s?.externalNodeId != null)
                    .map((s) => [String(s.externalNodeId), s]),
            );
            // First wire / pre-nodeId eras: allow index align once. After segments are
            // tagged, never inherit prompt from a different Group at the same index.
            const allowIndexFallback = !prev.some((s) => s?.externalNodeId != null);
            const isR2v = taskKey === "r2v" || this.hasExternalR2vGroups?.();
            const promptWriteBack = [];
            const activePromptIndex = (() => {
                const el = typeof document !== "undefined" ? document.activeElement : null;
                if (!el?.getAttribute) return -1;
                const n = parseInt(el.getAttribute("data-batch-prompt-index"), 10);
                return Number.isFinite(n) ? n : -1;
            })();
            this.timeline.segments = specs.map((spec, i) => {
                const matched = (spec.nodeId != null && prevByNode.get(String(spec.nodeId)))
                    || (allowIndexFallback ? (prev[i] || null) : null);
                const firstRef = imageRefFromPath(spec.firstImageFile);
                const genImage = firstRef
                    || (isR2v ? (matched?.genImage || { imageFile: "" }) : { imageFile: "" });
                // External graph is source of truth for r2v media (do not keep stale UI uploads).
                const refs = isR2v
                    ? (spec.refImages || []).map((r) => ({
                        index: r.index,
                        imageFile: r.imageFile || "",
                        imageB64: "",
                    }))
                    : (matched?.refs || []);
                const refVideos = isR2v
                    ? (spec.refVideos || []).map((r) => ({
                        index: r.index,
                        videoFile: r.videoFile || "",
                        fileName: r.fileName || "",
                        type: r.type || "input",
                        subfolder: r.subfolder || "",
                        pairedAudioFile: r.pairedAudioFile || "",
                        previewImageFile: r.previewImageFile || "",
                        previewImageUrl: r.previewImageUrl || "",
                        linked: !!r.linked || !!(r.videoFile || r.previewImageFile || r.previewImageUrl),
                    }))
                    : (matched?.refVideos || []);
                const refAudios = isR2v
                    ? (spec.refAudios || []).map((r) => ({
                        index: r.index,
                        audioFile: r.audioFile || "",
                        fileName: r.fileName || "",
                        type: r.type || "input",
                        subfolder: r.subfolder || "",
                    }))
                    : (matched?.refAudios || []);
                const specPrompt = String(spec.prompt ?? "").trim();
                const draftPrompt = matched ? String(matched.prompt || "").trim() : "";
                // Priority: focused Director textarea > Group widget > same-node draft.
                // Prevents a just-pasted Group-3 prompt from being replaced by stale
                // widget text from a previous short film during an incidental sync.
                let prompt = specPrompt;
                if (activePromptIndex === i && draftPrompt) {
                    prompt = draftPrompt;
                } else if (!specPrompt) {
                    prompt = draftPrompt;
                }
                if (prompt && prompt !== specPrompt) {
                    promptWriteBack.push({ index: i, prompt });
                }
                return newBatchSegment({
                    ...(matched?.id ? { id: matched.id } : {}),
                    durationSec: spec.durationSec ?? defaultDurationSec(taskKey),
                    prompt,
                    negativePrompt: matched?.negativePrompt ?? "",
                    externalNodeId: spec.nodeId ?? null,
                    refs,
                    refAudios,
                    refVideos,
                    genImage: genImage?.imageFile ? genImage : { imageFile: "" },
                    imageFile: genImage?.imageFile || "",
                    // Preserve preview frames for the same Group node across syncs.
                    previewB64: matched?.previewB64 || "",
                    previewFrames: matched?.previewFrames || [],
                    previewFps: matched?.previewFps,
                    refImageSize: matched?.refImageSize ?? matched?.ref_image_size,
                    ...(matched?.runEnabled != null ? { runEnabled: matched.runEnabled } : {}),
                });
            });
            for (const item of promptWriteBack) {
                this.writeExternalGroupPrompt(item.index, item.prompt);
            }
            normalizeImageBatchSegments(this);
            this.selectedIndex = Math.min(this.selectedIndex ?? 0, Math.max(0, this.timeline.segments.length - 1));
            this.renderImageBatchGroups?.();
            this.scheduleRender?.();
            this.commit?.(false, { syncTimeline: true });
            this.updateVideoNameLabel?.();
            this.updateDomWidgetHeight?.();
            this.updateRunSelectUI?.();
            this.updateSelectionUI?.();
        }
    }
};
