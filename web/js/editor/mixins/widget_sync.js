/** widget_sync mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { resolveOutputDimensions, snapDim } from "../../core/dims.js";
import { MIN_SEG } from "../../core/layout_spec.js";
import { formatProbeFps } from "../../core/ruler.js";
import { DEFAULT_CONTINUITY_FRAMES, isContinuityEligible, isContinuityEnabled, normalizeAudioMode, snapContinuityFrames } from "../../core/timeline_sanitize.js";
import { clamp, uid } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { getFl2vSampleFrames, normalizeFl2vSegments, updateFl2vDetailUI } from "../../minimax_fl2v.js";
import { DEFAULT_ASPECT_RATIO, DEFAULT_MEGAPIXELS, MINIMAX_CANVAS_MULTIPLE, NO_VIDEO_UPLOAD_TASKS, clampMegapixels, isCustomAspectRatio, normalizeRefImageSize, taskUsesReferenceAudios, taskUsesReferenceImages } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const widget_syncMixin = {
    _exportPreviewSuffix() {
        const cap = this.getMaxExportFrames();
        const exportMode = this.timeline.output?.exportMode === "segments"
            ? t("output.preview.segmentExport")
            : "";
        const dur = this.getTimelineDurationSec().toFixed(2);
        const fps = formatProbeFps(this.getFrameRate());
        const timeHint = t("output.preview.timeFps", { dur, fps });
        if (cap <= 0) return `${timeHint}${exportMode}`;
        const total = this.getTotalFrames();
        const exportTotal = this.getExportFrameTotal();
        if (exportTotal >= total) {
            return `${timeHint}${t("output.preview.exportFrames", { n: exportTotal })}${exportMode}`;
        }
        return `${timeHint}${t("output.preview.exportFramesPartial", { n: exportTotal, total })}${exportMode}`;
    },
    onOutputField(key, value) {
        this.timeline.output = this.timeline.output || {
            mode: "long_edge",
            aspectRatio: DEFAULT_ASPECT_RATIO,
            megapixels: DEFAULT_MEGAPIXELS,
            multiple: MINIMAX_CANVAS_MULTIPLE,
            longEdge: 848, width: 848, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        };
        if (key === "aspectRatio") {
            if (isCustomAspectRatio(value)) {
                // Keep current computed size when entering custom mode.
                this.applyCustomResolution(
                    this.timeline.output.width ?? this.outW?.value,
                    this.timeline.output.height ?? this.outH?.value,
                );
            } else {
                this.applyResolutionSelector(value, null);
            }
        } else if (key === "megapixels") {
            const mp = clampMegapixels(value);
            if (!isCustomAspectRatio(this.timeline.output.aspectRatio)) {
                this.applyResolutionSelector(null, mp);
            } else {
                this.timeline.output.megapixels = mp;
            }
        } else if (key === "mode") {
            this.timeline.output.mode = value;
        } else if (key === "longEdge") {
            // Long-edge is a size budget, not a canvas dim — do not snap to 32
            // (848 would become 864). Final W/H still snap via resolveOutputDimensions.
            const n = Math.round(Number(value) || 864);
            this.timeline.output.longEdge = Math.max(32, n);
        } else if (key === "width") {
            const useSelector = this.isImageBatch() || this.isGenMode() || this.isFl2vMode()
                || NO_VIDEO_UPLOAD_TASKS.has(this.getTaskKey());
            if (useSelector) {
                this.applyCustomResolution(value, this.timeline.output.height ?? this.outH?.value);
            } else {
                this.timeline.output.width = snapDim(value || 864);
            }
        } else if (key === "height") {
            const useSelector = this.isImageBatch() || this.isGenMode() || this.isFl2vMode()
                || NO_VIDEO_UPLOAD_TASKS.has(this.getTaskKey());
            if (useSelector) {
                this.applyCustomResolution(this.timeline.output.width ?? this.outW?.value, value);
            } else {
                this.timeline.output.height = snapDim(value || 480);
            }
        } else if (key === "maxExportFrames") {
            const n = parseInt(value, 10);
            this.timeline.output.maxExportFrames = Number.isFinite(n) && n > 0 ? n : 0;
        } else if (key === "exportMode") {
            this.timeline.output.exportMode = value === "segments" ? "segments" : "all";
        } else if (key === "audioMode") {
            this.timeline.output.audioMode = normalizeAudioMode(value);
        } else if (key === "continuityEnabled") {
            this.timeline.output.continuityEnabled = !!value;
        } else if (key === "continuityOverlapFrames") {
            this.timeline.output.continuityOverlapFrames = snapContinuityFrames(value);
        }
        this.syncOutputUIFromTimeline();
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
        // Refresh per-segment「引用上段」checkboxes when master toggle changes.
        if (key === "continuityEnabled") {
            if (this.isImageBatch()) this.renderImageBatchGroups?.();
            this.syncSegmentContinuityFromPrevUI?.();
        }
        this.commit();
        this.flushTimelineSync();
    },
    syncOutputToWidgets() {
        if (this.isImageBatch() && (this.getTaskKey() === "i2i" || this.getTaskKey() === "i2v")) {
            const out = this.timeline.output || {};
            const mode = (out.mode || "long_edge").toLowerCase();
            if (mode === "long_edge") {
                const src = this.getI2iSourceDimensions();
                const resolved = resolveOutputDimensions(src.width, src.height, out, {
                    width: this.widthWidget?.value,
                    height: this.heightWidget?.value,
                    refMaxSize: this.refMaxWidget?.value,
                });
                this.timeline.output = {
                    ...out,
                    mode: "long_edge",
                    longEdge: out.longEdge ?? resolved.refMaxSize,
                    width: resolved.width,
                    height: resolved.height,
                };
                if (this.widthWidget) this.widthWidget.value = resolved.width;
                if (this.heightWidget) this.heightWidget.value = resolved.height;
                if (this.refMaxWidget) this.refMaxWidget.value = resolved.refMaxSize;
                this.timeline.width = resolved.width;
                this.timeline.height = resolved.height;
                this.timeline.refMaxSize = resolved.refMaxSize;
            } else {
                const w = snapDim(+(out.width ?? this.widthWidget?.value ?? 864));
                const h = snapDim(+(out.height ?? this.heightWidget?.value ?? 480));
                this.timeline.output = { ...out, mode: "fixed", width: w, height: h };
                if (this.widthWidget) this.widthWidget.value = w;
                if (this.heightWidget) this.heightWidget.value = h;
                this.timeline.width = w;
                this.timeline.height = h;
            }
            this.updateOutputPreview();
            return;
        }
        if (this.isGenBlank() || this.isImageBatch() || this.isFl2vMode()) {
            const out = this.timeline.output || {};
            const resolved = isCustomAspectRatio(out.aspectRatio)
                ? this.applyCustomResolution(out.width, out.height)
                : this.applyResolutionSelector();
            this.timeline.width = resolved.width;
            this.timeline.height = resolved.height;
            this.timeline.refMaxSize = Math.max(resolved.width, resolved.height);
            this.updateOutputPreview();
            return;
        }
        const src = this.getSourceDimensions();
        const prevOut = this.timeline.output || {};
        const resolved = resolveOutputDimensions(src.width, src.height, prevOut, {
            width: this.timeline.width,
            height: this.timeline.height,
            refMaxSize: this.timeline.refMaxSize,
        });
        // Preserve audioMode / aspect / megapixels etc. — do not rebuild a bare object.
        this.timeline.output = {
            ...prevOut,
            mode: resolved.mode,
            longEdge: prevOut.longEdge ?? resolved.refMaxSize,
            width: resolved.width,
            height: resolved.height,
            maxExportFrames: prevOut.maxExportFrames ?? 0,
            exportMode: prevOut.exportMode ?? "all",
            audioMode: normalizeAudioMode(prevOut.audioMode),
            refImageSize: normalizeRefImageSize(prevOut.refImageSize ?? prevOut.ref_image_size),
            continuityEnabled: isContinuityEnabled(prevOut),
            continuityOverlapFrames: snapContinuityFrames(
                prevOut.continuityOverlapFrames ?? DEFAULT_CONTINUITY_FRAMES,
            ),
        };
        if (this.widthWidget) this.widthWidget.value = resolved.width;
        if (this.heightWidget) this.heightWidget.value = resolved.height;
        if (this.refMaxWidget) this.refMaxWidget.value = resolved.refMaxSize;
        this.timeline.width = resolved.width;
        this.timeline.height = resolved.height;
        this.timeline.refMaxSize = resolved.refMaxSize;
        this._refreshVideoStorageDimensions(resolved);
        this.updateOutputPreview();
    },
    syncFromWidgets() {
        this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {}, continuousReference: false };
        this.timeline.global.taskType = this.globalTask?.value || this.taskTypeWidget?.value || "";
        // r2v：公共提示词在素材组「公共素材页」里编辑，面板那个 textarea 已随
        // bd-split 隐藏（值永远是旧的）。这里再读它会把公共页的输入回滚掉。
        if (!this.usesR2vCommonPanel?.()) {
            this.timeline.global.prompt = this.globalPrompt?.value ?? this.globalPromptWidget?.value ?? "";
        }
        if (this.continuousRefCb) {
            this.timeline.global.continuousReference = !!this.continuousRefCb.checked;
        }
        // fl2v: totalFrames stores the sampling window (总时长), not visual overflow length.
        this.timeline.totalFrames = this.isFl2vMode()
            ? getFl2vSampleFrames(this)
            : this.getTotalFrames();
        this.timeline.frameRate = this.getFrameRate();
        this.timeline.output = this.timeline.output || {
            mode: "long_edge", longEdge: 864, width: 864, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        };
        if (this.timeline.output.audioMode == null) {
            this.timeline.output.audioMode = "generate";
        }
        // Sync from DOM when task+segments are eligible — do not rely on CSS
        // "hidden" class (can lag behind equal-split / task changes at queue time).
        const continuityEligible = isContinuityEligible(this);
        if (continuityEligible && this.segmentContinuityCb) {
            this.timeline.output.continuityEnabled = !!this.segmentContinuityCb.checked;
        } else {
            // Normalize stored flag without clearing preference while ineligible.
            this.timeline.output.continuityEnabled = isContinuityEnabled(this.timeline.output);
        }
        if (continuityEligible && this.segmentContinuityOverlap) {
            this.timeline.output.continuityOverlapFrames = snapContinuityFrames(
                this.segmentContinuityOverlap.value
                    ?? this.timeline.output.continuityOverlapFrames
                    ?? DEFAULT_CONTINUITY_FRAMES,
            );
        } else if (this.timeline.output.continuityOverlapFrames != null) {
            this.timeline.output.continuityOverlapFrames = snapContinuityFrames(
                this.timeline.output.continuityOverlapFrames,
            );
        }
        this.syncOutputToWidgets();
    },
    commit(skipRender = false, { syncTimeline = true } = {}) {
        this.syncFromWidgets();
        this.normalizeSegments();
        if (this.isRunSelectEnabled()) this.normalizeRunSelection();
        this.updateRunSelectUI();
        this.updateSegmentContinuityUI();
        if (this.taskTypeWidget) this.taskTypeWidget.value = this.timeline.global.taskType;
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt;
        if (this.negativePromptWidget) {
            const neg = this.globalNegative?.value ?? this.segNegative?.value ?? this.negativePromptWidget.value ?? "";
            this.negativePromptWidget.value = neg;
        }
        if (this.totalFramesWidget) {
            this.totalFramesWidget.value = Math.max(
                0,
                this.isFl2vMode() ? getFl2vSampleFrames(this) : this.getTotalFrames(),
            );
        }
        this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
        if (syncTimeline) this.scheduleTimelineSync();
        if (!skipRender) this.scheduleRender();
        if (this.usesGlobalRefPanel() && taskUsesReferenceImages(this.getTaskKey())) {
            this.renderRefSlots(this.timeline.global.refs, this.globalRefsBox, true);
        }
        if (this.usesGlobalRefPanel() && taskUsesReferenceAudios(this.getTaskKey())) {
            this.renderRefAudioSlots();
        }
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.updateSelectionUI();
    },
    normalizeSegments() {
        if (this.isImageBatch()) {
            this.normalizeImageBatchSegments();
            return;
        }
        if (this.isFl2vMode()) {
            normalizeFl2vSegments(this);
            const n = this.timeline.segments?.length || 0;
            this.selectedIndex = clamp(this.selectedIndex, 0, Math.max(0, n - 1));
            return;
        }
        if (this.isGenMode()) {
            this.normalizeGenSegments();
            return;
        }
        const total = this.getTotalFrames();
        let segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        if (!total) {
            this.timeline.segments = [];
            this.timeline.totalFrames = 0;
            return;
        }
        if (!segs.length) segs = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        const fixed = [];
        let cursor = 0;
        for (const seg of segs) {
            const start = clamp(seg.start, cursor, total);
            let length = Math.max(MIN_SEG, seg.length ?? (total - start));
            if (start + length > total) length = total - start;
            if (length < MIN_SEG) continue;
            fixed.push({ ...seg, start, length, refs: seg.refs || [] });
            cursor = start + length;
        }
        if (fixed.length && cursor < total) fixed[fixed.length - 1].length += total - cursor;
        this.timeline.segments = fixed;
        this.timeline.totalFrames = total;
        this.selectedIndex = clamp(this.selectedIndex, 0, Math.max(0, fixed.length - 1));
        this.updateSegmentContinuityUI();
    }
};
