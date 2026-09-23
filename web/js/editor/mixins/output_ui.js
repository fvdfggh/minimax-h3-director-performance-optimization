/** output_ui mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { resolveOutputDimensions, snapDim } from "../../core/dims.js";
import { DEFAULT_CONTINUITY_FRAMES, isContinuityEligible, isContinuityEnabled, normalizeAudioMode, snapContinuityFrames } from "../../core/timeline_sanitize.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { CUSTOM_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, DEFAULT_MEGAPIXELS, MINIMAX_CANVAS_MULTIPLE, NO_VIDEO_UPLOAD_TASKS, isContinuityMasterEnabled, isCustomAspectRatio, isSegmentContinuityFromPrev, normalizeAspectRatioLabel, resolutionFromSelector, resolveSegmentRefImageSize, snapResolutionDim } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const output_uiMixin = {
    getI2iSourceDimensions() {
        for (const seg of this.timeline.segments || []) {
            const gi = seg.genImage || {};
            const w = +(gi.width || 0);
            const h = +(gi.height || 0);
            if (w > 0 && h > 0) return { width: w, height: h };
        }
        const out = this.timeline.output || {};
        if (+(out.sourceWidth || 0) > 0 && +(out.sourceHeight || 0) > 0) {
            return { width: +out.sourceWidth, height: +out.sourceHeight };
        }
        return { width: 0, height: 0 };
    },
    getSourceDimensions() {
        const clips = this.getVideoClips?.() || [];
        const video = clips[0] || this.timeline.video || {};
        // Prefer native clip/source size — never fall back to output canvas W×H
        // (that makes long_edge look like a no-op and keeps a cropped 16:9).
        if (+(video.width || 0) > 0 && +(video.height || 0) > 0) {
            return { width: +video.width, height: +video.height };
        }
        for (const clip of clips) {
            if (+(clip?.width || 0) > 0 && +(clip?.height || 0) > 0) {
                return { width: +clip.width, height: +clip.height };
            }
        }
        return { width: 0, height: 0 };
    },
    _refreshVideoStorageDimensions(resolved) {
        if (!resolved?.width || !resolved?.height) return;
        this._storageWidth = resolved.width;
        this._storageHeight = resolved.height;
        if (this.timeline.video) {
            this.timeline.video.storageWidth = resolved.width;
            this.timeline.video.storageHeight = resolved.height;
        }
        for (const clip of this.getVideoClips()) {
            clip.storageWidth = resolved.width;
            clip.storageHeight = resolved.height;
        }
    },
    syncOutputUIFromTimeline() {
        const out = this.timeline.output || {
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
        // Prefer ResolutionSelector fields; backfill from width/height when missing.
        // Custom keeps explicit width/height and does not recompute from megapixels.
        if (!isCustomAspectRatio(out.aspectRatio) && (out.aspectRatio == null || out.megapixels == null)) {
            const resolved = resolutionFromSelector(
                out.aspectRatio || DEFAULT_ASPECT_RATIO,
                out.megapixels ?? DEFAULT_MEGAPIXELS,
                out.multiple ?? MINIMAX_CANVAS_MULTIPLE,
            );
            if (resolved) {
                out.aspectRatio = resolved.aspectRatio;
                out.megapixels = resolved.megapixels;
                out.multiple = resolved.multiple;
                if (out.width == null) out.width = resolved.width;
                if (out.height == null) out.height = resolved.height;
                this.timeline.output = { ...out };
            }
        }
        if (this.outMode) this.outMode.value = out.mode || "long_edge";
        if (this.outAspect) {
            const ar = isCustomAspectRatio(out.aspectRatio)
                ? CUSTOM_ASPECT_RATIO
                : normalizeAspectRatioLabel(out.aspectRatio || DEFAULT_ASPECT_RATIO);
            this.outAspect.value = ar;
            if (out.aspectRatio !== ar) {
                out.aspectRatio = ar;
                this.timeline.output = { ...out };
            }
        }
        if (this.outMp) this.outMp.value = String(out.megapixels ?? DEFAULT_MEGAPIXELS);
        if (this.outLong) this.outLong.value = String(out.longEdge ?? 864);
        if (this.outW) this.outW.value = String(out.width ?? 864);
        if (this.outH) this.outH.value = String(out.height ?? 480);
        if (this.outMaxFrames) this.outMaxFrames.value = String(out.maxExportFrames ?? 0);
        if (this.outExportMode) this.outExportMode.value = out.exportMode === "segments" ? "segments" : "all";
        if (this.outAudioMode) {
            const am = normalizeAudioMode(out.audioMode);
            this.outAudioMode.value = am;
            if (out.audioMode !== am) {
                out.audioMode = am;
                this.timeline.output = { ...out };
            }
        }
        if (this.segmentContinuityCb) this.segmentContinuityCb.checked = isContinuityEnabled(out);
        if (this.segmentContinuityOverlap) {
            this.segmentContinuityOverlap.value = String(
                snapContinuityFrames(out.continuityOverlapFrames ?? DEFAULT_CONTINUITY_FRAMES),
            );
        }
        this.syncFrameRateUI(this.timeline.frameRate);
        this.updateOutputModeUI();
        this.updateSegmentContinuityUI();
        this.updateOutputPreview();
    },
    updateSegmentContinuityUI() {
        const show = isContinuityEligible(this);
        if (this.segmentContinuityWrap) {
            this.segmentContinuityWrap.classList.toggle("hidden", !show);
            this.segmentContinuityWrap.hidden = !show;
            this.segmentContinuityWrap.setAttribute("aria-hidden", show ? "false" : "true");
            this.segmentContinuityWrap.title = show
                ? t("tooltip.segmentContinuity")
                : "";
        }
        if (!show && this.timeline?.output) {
            // Hide only — keep saved preference so it returns when multi-segment again.
        }
        if (this.segmentContinuityOverlap && this.timeline?.output) {
            const frames = snapContinuityFrames(
                this.timeline.output.continuityOverlapFrames ?? DEFAULT_CONTINUITY_FRAMES,
            );
            this.segmentContinuityOverlap.value = String(frames);
            this.timeline.output.continuityOverlapFrames = frames;
        }
        if (this.segmentContinuityCb && this.timeline?.output) {
            // Keep DOM aligned with timeline; eligibility only gates visibility.
            this.segmentContinuityCb.checked = isContinuityEnabled(this.timeline.output);
        }
        this.syncSegmentContinuityFromPrevUI();
        this.syncSegmentRefImageSizeUI();
    },
    /** Per-segment「引用上段」on v2v/rv2v segment panel (index>0 + master on). */
    syncSegmentContinuityFromPrevUI() {
        const wrap = this.segContinuityFromPrevWrap;
        const cb = this.segContinuityFromPrevCb;
        if (!wrap || !cb) return;
        const idx = this.selectedIndex ?? 0;
        const masterOn = isContinuityEligible(this)
            && isContinuityMasterEnabled(this.timeline?.output);
        const show = masterOn && idx > 0 && !this.isImageBatch() && !this.isFl2vMode();
        wrap.classList.toggle("hidden", !show);
        wrap.hidden = !show;
        if (!show) return;
        const seg = this.timeline.segments?.[idx];
        cb.checked = isSegmentContinuityFromPrev(seg, idx);
        wrap.title = t("tooltip.segmentContinuityFromPrev");
    },
    /** Per-segment ref_image_size for rv2v (r2v uses the group card control). */
    syncSegmentRefImageSizeUI() {
        const wrap = this.segRefImageSizeWrap;
        const sel = this.segRefImageSize;
        if (!wrap || !sel) return;
        const show = this.getTaskKey() === "rv2v" && !this.isImageBatch() && !this.isFl2vMode();
        wrap.classList.toggle("hidden", !show);
        wrap.hidden = !show;
        if (!show) return;
        const seg = this.timeline.segments?.[this.selectedIndex ?? 0];
        const value = resolveSegmentRefImageSize(seg, this.timeline.output);
        sel.value = value;
        if (seg && seg.refImageSize !== value) seg.refImageSize = value;
        wrap.title = t("tooltip.refImageSize");
    },
    /** Apply ResolutionSelector → fixed width/height on timeline + node widgets. */
    applyResolutionSelector(aspectRatio = null, megapixels = null) {
        const out = this.timeline.output || {};
        const ar = aspectRatio ?? out.aspectRatio ?? this.outAspect?.value ?? DEFAULT_ASPECT_RATIO;
        if (isCustomAspectRatio(ar)) {
            return this.applyCustomResolution(out.width, out.height);
        }
        const resolved = resolutionFromSelector(
            ar,
            megapixels ?? out.megapixels ?? this.outMp?.value ?? DEFAULT_MEGAPIXELS,
            out.multiple ?? MINIMAX_CANVAS_MULTIPLE,
        );
        if (!resolved) {
            return this.applyCustomResolution(out.width, out.height);
        }
        this.timeline.output = {
            ...out,
            mode: "fixed",
            aspectRatio: resolved.aspectRatio,
            megapixels: resolved.megapixels,
            multiple: resolved.multiple,
            width: resolved.width,
            height: resolved.height,
            longEdge: Math.max(resolved.width, resolved.height),
        };
        if (this.widthWidget) this.widthWidget.value = resolved.width;
        if (this.heightWidget) this.heightWidget.value = resolved.height;
        if (this.refMaxWidget) this.refMaxWidget.value = Math.max(resolved.width, resolved.height);
        if (this.outW) this.outW.value = String(resolved.width);
        if (this.outH) this.outH.value = String(resolved.height);
        if (this.outAspect) this.outAspect.value = resolved.aspectRatio;
        // Keep the in-progress typed text while the field is focused.
        if (this.outMp && document.activeElement !== this.outMp) {
            this.outMp.value = String(resolved.megapixels);
        }
        return resolved;
    },
    /** Apply explicit custom width × height (snapped to canvas multiple). */
    applyCustomResolution(width = null, height = null) {
        const out = this.timeline.output || {};
        const mult = out.multiple ?? MINIMAX_CANVAS_MULTIPLE;
        const w = snapResolutionDim(width ?? out.width ?? this.outW?.value ?? this.widthWidget?.value ?? 864, mult);
        const h = snapResolutionDim(height ?? out.height ?? this.outH?.value ?? this.heightWidget?.value ?? 480, mult);
        this.timeline.output = {
            ...out,
            mode: "fixed",
            aspectRatio: CUSTOM_ASPECT_RATIO,
            megapixels: out.megapixels ?? DEFAULT_MEGAPIXELS,
            multiple: mult,
            width: w,
            height: h,
            longEdge: Math.max(w, h),
        };
        if (this.widthWidget) this.widthWidget.value = w;
        if (this.heightWidget) this.heightWidget.value = h;
        if (this.refMaxWidget) this.refMaxWidget.value = Math.max(w, h);
        if (this.outW) this.outW.value = String(w);
        if (this.outH) this.outH.value = String(h);
        if (this.outAspect) this.outAspect.value = CUSTOM_ASPECT_RATIO;
        if (this.outMp) this.outMp.value = String(this.timeline.output.megapixels);
        return {
            width: w,
            height: h,
            megapixels: this.timeline.output.megapixels,
            aspectRatio: CUSTOM_ASPECT_RATIO,
            multiple: mult,
        };
    },
    updateOutputModeUI() {
        const taskKey = this.getTaskKey();
        const useSelector = this.isImageBatch() || this.isGenMode() || this.isFl2vMode()
            || NO_VIDEO_UPLOAD_TASKS.has(taskKey);
        // Gen / batch / fl2v: aspect + megapixels, or Custom width/height.
        // Video edit (v2v): long_edge / fixed — must toggle .hidden (CSS uses !important).
        if (this.outAspect) this.outAspect.classList.toggle("hidden", !useSelector);
        if (this.outMode) this.outMode.classList.toggle("hidden", useSelector);
        if (this.outLongWrap) this.outLongWrap.style.display = "";
        if (useSelector) {
            const custom = isCustomAspectRatio(this.timeline.output?.aspectRatio ?? this.outAspect?.value);
            if (this.outMpWrap) this.outMpWrap.classList.toggle("hidden", custom);
            if (this.outLongWrap) this.outLongWrap.classList.add("hidden");
            if (this.outFixedWrap) this.outFixedWrap.classList.toggle("hidden", !custom);
            if (custom) this.applyCustomResolution();
            else this.applyResolutionSelector();
            return;
        }
        if (this.outMpWrap) this.outMpWrap.classList.add("hidden");
        const mode = this.timeline.output?.mode || "long_edge";
        const isFixed = mode === "fixed";
        if (this.outLongWrap) this.outLongWrap.classList.toggle("hidden", isFixed);
        if (this.outFixedWrap) this.outFixedWrap.classList.toggle("hidden", !isFixed);
    },
    updateOutputPreview() {
        if (!this.outPreview) return;
        if (this.isImageBatch() && (this.getTaskKey() === "i2i" || this.getTaskKey() === "i2v")) {
            const out = this.timeline.output || {};
            if ((out.mode || "long_edge") === "long_edge") {
                const src = this.getI2iSourceDimensions();
                const resolved = resolveOutputDimensions(src.width, src.height, out, {
                    refMaxSize: this.refMaxWidget?.value,
                });
                const note = src.width > 0 ? "" : t("output.preview.needSourceForLongEdge");
                this.outPreview.textContent = `→ ${resolved.width}×${resolved.height}${note}${this._exportPreviewSuffix()}`;
            } else {
                const w = snapDim(+(out.width ?? this.outW?.value ?? 864));
                const h = snapDim(+(out.height ?? this.outH?.value ?? 480));
                this.outPreview.textContent = `→ ${w}×${h}${this._exportPreviewSuffix()}`;
            }
            return;
        }
        if (this.isGenBlank() || this.isImageBatch() || this.isFl2vMode()) {
            const out = this.timeline.output || {};
            if (isCustomAspectRatio(out.aspectRatio)) {
                const w = snapResolutionDim(out.width ?? this.outW?.value ?? 864, out.multiple ?? MINIMAX_CANVAS_MULTIPLE);
                const h = snapResolutionDim(out.height ?? this.outH?.value ?? 480, out.multiple ?? MINIMAX_CANVAS_MULTIPLE);
                this.outPreview.textContent = t("output.preview.custom", { w, h }) + this._exportPreviewSuffix();
                return;
            }
            const resolved = resolutionFromSelector(
                out.aspectRatio || DEFAULT_ASPECT_RATIO,
                out.megapixels ?? DEFAULT_MEGAPIXELS,
                out.multiple ?? MINIMAX_CANVAS_MULTIPLE,
            );
            if (!resolved) {
                const w = snapResolutionDim(out.width ?? 864);
                const h = snapResolutionDim(out.height ?? 480);
                this.outPreview.textContent = `→ ${w}×${h}${this._exportPreviewSuffix()}`;
                return;
            }
            const w = resolved.width;
            const h = resolved.height;
            const ar = resolved.aspectRatio.split(" ")[0];
            this.outPreview.textContent = `→ ${w}×${h} · ${ar} · ${resolved.megapixels}MP${this._exportPreviewSuffix()}`;
            return;
        }
        const src = this.getSourceDimensions();
        const out = this.timeline.output || {};
        const resolved = resolveOutputDimensions(src.width, src.height, out, {
            width: this.widthWidget?.value,
            height: this.heightWidget?.value,
            refMaxSize: this.refMaxWidget?.value,
        });
        if (src.width > 0 && src.height > 0) {
            const mode = (out.mode || "long_edge").toLowerCase();
            const note = mode === "long_edge"
                ? t("output.preview.scaleKeepAspect")
                : t("output.preview.fixedCrop");
            this.outPreview.textContent = `${src.width}×${src.height} → ${resolved.width}×${resolved.height}${note}${this._exportPreviewSuffix()}`;
        } else {
            this.outPreview.textContent = `→ ${resolved.width}×${resolved.height}${t("output.preview.needSourceForLongEdge")}${this._exportPreviewSuffix()}`;
        }
    }
};
