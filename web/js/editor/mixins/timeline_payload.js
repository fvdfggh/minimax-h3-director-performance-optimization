/** timeline_payload mixin for the Director editor (timeline_payload).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { resolveOutputDimensions } from "../../core/dims.js";
import { deletedSourceRanges } from "../../core/frame_map.js";
import { TIMELINE_SYNC_DEBOUNCE_MS } from "../../core/layout_spec.js";
import { normalizeOutputContinuity, sanitizeSegmentForPayload, stripTimelineContinuityRootFields, stripTimelineEphemeralFields } from "../../core/timeline_sanitize.js";
import { buildFl2vPayloadFields, flushFl2vPromptDraft } from "../../minimax_fl2v.js";
import { imageBatchRequiresFixedOutput, isSegmentContinuityFromPrev, isVideoBatchTask, resolveSegmentRefImageSize, sumFrameCounts } from "../../minimax_gen_timeline.js";
import { flushBatchPromptInputs } from "../../minimax_image_batch.js";
import { refreshPromptTokenEditors } from "../../minimax_prompt_mentions.js";
export const timeline_payloadMixin = {
    buildTimelinePayload() {
        if (this.isFl2vMode()) {
            const fl = buildFl2vPayloadFields(this);
            const outMode = this.timeline.output?.mode || "long_edge";
            const output = normalizeOutputContinuity({
                ...(this.timeline.output || {}),
                mode: outMode,
            });
            const body = { ...this.timeline };
            stripTimelineContinuityRootFields(body);
            stripTimelineEphemeralFields(body);
            return {
                ...body,
                version: 5,
                ...fl,
                frameRate: this.getFrameRate(),
                global: {
                    ...(this.timeline.global || {}),
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                },
                output,
                ...this._runSelectionPayload(),
                ...this._segmentExportPayload(),
            };
        }
        if (this.isImageBatch()) {
            this._persistCurrentBatchWorkspace();
            const taskKey = this.getTaskKey();
            const i2iSrc = (taskKey === "i2i" || taskKey === "i2v") ? this.getI2iSourceDimensions() : null;
            const outMode = imageBatchRequiresFixedOutput(taskKey)
                ? "fixed"
                : (this.timeline.output?.mode || "long_edge");
            const output = normalizeOutputContinuity({
                ...this.timeline.output,
                mode: outMode,
            });
            if (!isVideoBatchTask(taskKey)) {
                output.exportMode = "all";
            }
            if (i2iSrc?.width > 0 && i2iSrc?.height > 0) {
                output.sourceWidth = i2iSrc.width;
                output.sourceHeight = i2iSrc.height;
            }
            const batchBody = { ...this.timeline };
            stripTimelineContinuityRootFields(batchBody);
            stripTimelineEphemeralFields(batchBody);
            return {
                ...batchBody,
                version: 5,
                timelineMode: "prompt_batch",
                editMode: "segment",
                totalFrames: sumFrameCounts(this.timeline.segments),
                frameRate: this.getFrameRate(),
                width: this.timeline.output?.width,
                height: this.timeline.output?.height,
                global: {
                    ...this.timeline.global,
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                    ...(i2iSrc?.width > 0 ? { sourceWidth: i2iSrc.width, sourceHeight: i2iSrc.height } : {}),
                },
                output,
                segments: this.timeline.segments.map((s, i) => {
                    const clean = sanitizeSegmentForPayload(s);
                    return {
                        id: clean.id,
                        start: clean.start,
                        length: clean.frameCount ?? clean.length ?? 1,
                        frameCount: clean.frameCount ?? clean.length ?? 1,
                        durationSec: clean.durationSec,
                        prompt: clean.prompt || "",
                        negativePrompt: clean.negativePrompt || "",
                        taskType: clean.taskType || "",
                        refs: clean.refs || [],
                        refAudios: clean.refAudios || [],
                        refVideos: clean.refVideos || [],
                        genImage: clean.genImage || { imageFile: "" },
                        // Persist per-segment「引用上段」(default true when unset).
                        continuityFromPrev: isSegmentContinuityFromPrev(clean, i),
                        // 「对齐下段」is opt-in (default false) and cache-driven.
                        continuityToNext: clean.continuityToNext === true,
                        refImageSize: resolveSegmentRefImageSize(clean, this.timeline.output),
                    };
                }),
                ...this._runSelectionPayload(),
                ...this._segmentExportPayload(),
            };
        }
        if (this.isGenMode()) {
            const mode = this.getDirectorMode();
            const genBody = { ...this.timeline };
            stripTimelineContinuityRootFields(genBody);
            stripTimelineEphemeralFields(genBody);
            return {
                ...genBody,
                version: 5,
                timelineMode: mode,
                totalFrames: sumFrameCounts(this.timeline.segments),
                frameRate: this.getFrameRate(),
                width: this.timeline.output?.width,
                height: this.timeline.output?.height,
                refMaxSize: this.timeline.output?.longEdge,
                global: {
                    ...this.timeline.global,
                    taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                    prompt: this.timeline.global?.prompt || "",
                },
                output: normalizeOutputContinuity({ ...this.timeline.output }),
                segments: this.timeline.segments.map((s) => {
                const clean = sanitizeSegmentForPayload(s);
                return {
                    ...clean,
                    frameCount: clean.frameCount ?? clean.length,
                };
            }),
            ...this._runSelectionPayload(),
            ...this._segmentExportPayload(),
            };
        }
        this._persistCurrentVideoWorkspace();
        const video = { ...(this.timeline.video || {}) };
        const frameMap = video.frameMap?.length ? video.frameMap : [];
        const src = this.getSourceDimensions();
        const resolved = resolveOutputDimensions(src.width, src.height, this.timeline.output || {}, {
            refMaxSize: this.refMaxWidget?.value,
        });
        const storageW = resolved.width || video.storageWidth || this._storageWidth;
        const storageH = resolved.height || video.storageHeight || this._storageHeight;
        const clips = this.getVideoClips().map((c) => ({
            ...c,
            storageWidth: storageW,
            storageHeight: storageH,
        }));
        const { referenceVideo: _legacyRefVideo, reference_video: _legacyRefVideo2, ...timelineBody } = this.timeline;
        stripTimelineContinuityRootFields(timelineBody);
        stripTimelineEphemeralFields(timelineBody);
        const clipSourceTotal = clips.reduce(
            (s, c) => s + (parseInt(c.sourceFrameCount, 10) || 0),
            0,
        );
        const sourceFrameCount = parseInt(video.sourceFrameCount, 10)
            || clipSourceTotal
            || (frameMap.length ? 0 : this.getTotalFrames());
        return {
            ...timelineBody,
            version: 4,
            timelineMode: "video",
            totalFrames: this.getTotalFrames(),
            frameRate: this.getFrameRate(),
            videoClips: clips,
            global: {
                ...(this.timeline.global || {}),
                taskType: this.globalTask?.value || this.taskTypeWidget?.value || "",
                prompt: this.timeline.global?.prompt || "",
                referenceVideo: this.timeline.global?.referenceVideo || {},
                continuousReference: !!this.timeline.global?.continuousReference,
            },
            segments: (this.timeline.segments || []).map((s) => {
                const clean = sanitizeSegmentForPayload(s);
                return {
                    ...clean,
                    referenceVideo: clean.referenceVideo || {},
                };
            }),
            video: {
                ...video,
                frameMap,
                sourceFrameCount,
                deletedSourceRanges: frameMap.length ? [] : (video.deletedSourceRanges || []),
                frames: (video.videoFile || video.fileName)
                    ? []
                    : (this._legacyFrames.length ? this._legacyFrames : []),
                storageWidth: storageW,
                storageHeight: storageH,
            },
            output: normalizeOutputContinuity({ ...this.timeline.output }),
            ...this._runSelectionPayload(),
            ...this._segmentExportPayload(),
        };
    },
    flushTimelineSync() {
        clearTimeout(this._syncTimer);
        this._syncTimer = null;
        // Refresh visibility first so queue flush can pull checkbox state reliably.
        this.updateSegmentContinuityUI();
        this._writeTimelineWidget();
    },
    scheduleTimelineSync() {
        clearTimeout(this._syncTimer);
        this._syncTimer = setTimeout(() => this._writeTimelineWidget(), TIMELINE_SYNC_DEBOUNCE_MS);
    },
    _flushPromptTokenEditors() {
        this.globalPrompt?.__bdTokenApi?.sync?.();
        this.segPrompt?.__bdTokenApi?.sync?.();
        refreshPromptTokenEditors(this.root || document);
    },
    _writeTimelineWidget() {
        if (!this.timelineWidget) return;
        // Token editors keep textarea.value in sync on blur/input; force-flush
        // before serialize so a focused editor cannot drop the latest draft.
        this._flushPromptTokenEditors();
        // Batch prompt textareas can lag behind segment objects after duration
        // normalize — always harvest DOM drafts before serializing timeline_data.
        if (this.isImageBatch?.()) flushBatchPromptInputs(this);
        if (this.isFl2vMode?.()) flushFl2vPromptDraft(this);
        this.syncFromWidgets();
        this.timelineWidget.value = JSON.stringify(this.buildTimelinePayload());
        this.node.setDirtyCanvas(true, false);
    },
    _markNodeDirtyLight() {
        this.node.setDirtyCanvas(true, false);
    }
};
