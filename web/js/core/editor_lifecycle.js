/** Mounting, sizing and tearing down the Director editor's DOM.
 *
 * ComfyUI hands a node a DOM widget whose height it does not manage, so the plugin
 * has to: measure the canvas the timeline needs, keep the widget in step with the
 * node's size, move it after the other widgets, prune the leftovers of an older
 * install, and clean up when the node goes away. These helpers do exactly that and
 * nothing else.
 *
 * One piece deliberately stays behind: ``initDirectorEditor`` — the only function
 * that must know the concrete editor class, because it constructs it. Keeping the
 * construction site next to the class means this module never imports
 * minimax_timeline.js, so the two are not circular.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { app } from "../../../scripts/app.js";
import { coerceTimelineFps } from "./dims.js";
import { DIRECTOR_DOM_WIDGET_NAME, DIRECTOR_MIN_WIDTH, DIRECTOR_UI_RUNAWAY_ABS_H, DIRECTOR_UI_RUNAWAY_EXTRA_H, HIDDEN_WIDGETS, LIVE_SAMPLE_PREVIEW_H, MIN_SEG, RULER_H, SEG_LABEL_H, STAGE_PREVIEW_H, TRACK_H } from "./layout_spec.js";
import { DEFAULT_CONTINUITY_FRAMES, normalizeAudioMode, normalizeOutputContinuity, sanitizeRefVideo, stripTimelineContinuityRootFields, stripTimelineEphemeralFields } from "./timeline_sanitize.js";
import { uid } from "./utils.js";
import { getFl2vUiHeight } from "../minimax_fl2v.js";
import { CUSTOM_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, DEFAULT_MEGAPIXELS, MINIMAX_CANVAS_MULTIPLE, RESOLUTION_ASPECTS, defaultFrameCount, isPromptBatchTask, normalizeAspectRatioLabel, normalizeRefImageSize, resolveTaskKey, sumFrameCounts } from "../minimax_gen_timeline.js";
import { DIRECTOR_UI_MAX_EXTRA_H, bindDomWidgetContentComputeSize, getImageBatchUiHeight, syncBatchPanelFillHeight } from "../minimax_image_batch.js";

export function getDirectorUiHeight(editor) {
    if (editor?.getDirectorMode?.() === "prompt_batch") {
        const batchH = getImageBatchUiHeight(editor);
        // t2v / i2v / r2v show the main timeline track above batch cards.
        if (editor?.usesBatchTimeline?.()) {
            const track = editor?.canvasHeight || RULER_H + SEG_LABEL_H + TRACK_H;
            // toolbar + track + batch panel (batchH already includes list max-height cap)
            return batchH + track + 100;
        }
        return batchH + 100;
    }
    if (editor?.getDirectorMode?.() === "fl2v") {
        let h = getFl2vUiHeight(editor) + 110;
        if (editor?.needsLiveSamplePanel?.()) h += LIVE_SAMPLE_PREVIEW_H + 12;
        return h;
    }
    let h = (editor?.canvasHeight || RULER_H + SEG_LABEL_H + TRACK_H) + 370 + 52;
    if (
        editor?.hasVideo?.()
        && !editor?.isImageBatch?.()
        && !editor?.isGenMode?.()
        && !editor?.isFl2vMode?.()
    ) {
        h += STAGE_PREVIEW_H + 10;
    }
    // v2v live preview sits beside the prompt (no extra vertical stack).
    if (editor?.needsLiveSamplePanel?.() && !editor?.usesV2vPromptStyle?.()) {
        h += LIVE_SAMPLE_PREVIEW_H + 12;
    }
    return h;
}

export function hookTaskTypeWidget(node) {
    const tw = node.widgets?.find((w) => w.name === "task_type");
    if (!tw || tw._berniniTaskHooked) return;
    tw._berniniTaskHooked = true;
    const orig = tw.callback;
    tw.callback = function (...args) {
        const r = orig?.apply(this, args);
        const ed = node._minimaxEditor;
        if (ed?.globalTask) ed.globalTask.value = tw.value;
        ed?.onTaskTypeChanged?.(tw.value);
        return r;
    };
}

export function healOversizedDirectorNode(node, editor) {
    if (!node?.size || !node.computeSize) return false;
    bindDomWidgetContentComputeSize(editor);
    const ideal = node.computeSize()?.[1];
    if (ideal == null) return false;
    const curH = node.size[1] || 0;
    const runaway = curH > DIRECTOR_UI_RUNAWAY_ABS_H
        || curH > ideal + DIRECTOR_UI_RUNAWAY_EXTRA_H;
    if (!runaway) return false;
    // Keep a modest stretch so heal does not feel like a hard snap to content min.
    const safeH = Math.max(ideal, Math.min(curH, ideal + DIRECTOR_UI_MAX_EXTRA_H));
    node.setSize([node.size[0], safeH]);
    node.setDirtyCanvas?.(true, true);
    return true;
}

export function scheduleDirectorLayoutSettle(editor) {
    if (!editor) return;
    const run = () => {
        if (editor.isPlaying || editor._pauseSettling) return;
        bindDomWidgetContentComputeSize(editor);
        // On reload the node kept its old saved size; grow it to the (possibly larger)
        // content min so the batch list actually gets the taller slot. Only grows,
        // never shrinks a user-enlarged node — safe for workflow size preservation.
        // 所有模式通用：i2v/t2v 多组提示词/参考图也需要把节点撑高到内容高度，
        // 否则列表被 .bd-wrap overflow:hidden 裁掉（之前只给 r2v 兜底，导致非 r2v 显示异常）。
        ensureDirectorNodeFitsContent(editor?.node, editor);
        syncBatchPanelFillHeight(editor, { settle: true });
    };
    requestAnimationFrame(() => {
        run();
        requestAnimationFrame(run);
        setTimeout(run, 80);
        setTimeout(run, 250);
    });
}

export function ensureDirectorNodeFitsContent(node, editor) {
    if (!node?.size || !node.computeSize) return false;
    // Progress ticks must not grow the node — status text / rebuild noise used to ratchet.
    if (editor?.runStatusEl?.classList?.contains("active")) return false;
    bindDomWidgetContentComputeSize(editor);
    const ideal = node.computeSize()?.[1];
    if (ideal == null) return false;
    if ((node.size[1] || 0) >= ideal - 2) return false;
    const maxOk = ideal + DIRECTOR_UI_MAX_EXTRA_H;
    node.setSize([node.size[0], Math.min(ideal, maxOk)]);
    node.setDirtyCanvas?.(true, true);
    return true;
}

export function syncDirectorNodeSize(node, editor) {
    if (editor?.isPlaying) return;
    // Update CSS/content min. Avoid stretch bookkeeping + Vue RO feedback loops.
    // User-dragged height is preserved by never shrinking (#7).
    editor?.updateDomWidgetHeight?.();
}

export function ensureDirectorDomWidgetWidth(node) {
    const widget = node?._minimaxDomWidget;
    const fullW = node?.size?.[0];
    if (!widget || !fullW) return false;
    if (widget.width === fullW) return false;
    widget.width = fullW;
    return true;
}

export function moveDirectorDomWidgetToEnd(node) {
    const widget = node?._minimaxDomWidget;
    if (!widget || !node?.widgets?.length) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1 || idx === node.widgets.length - 1) return;
    node.widgets.splice(idx, 1);
    node.widgets.push(widget);
}

export function finalizeDirectorWidgetOrder(node) {
    moveDirectorDomWidgetToEnd(node);
}

export function listDirectorDomWidgets(node) {
    return (node?.widgets || []).filter((w) => w?.name === DIRECTOR_DOM_WIDGET_NAME);
}

export function pruneDirectorDomWidgets(node) {
    if (!node) return null;
    const dups = listDirectorDomWidgets(node);
    const keep = dups.find((w) => w.element?.querySelector?.(":scope > .bd-wrap"))
        || dups.find((w) => w === node._minimaxDomWidget && w?.element)
        || dups.find((w) => w?.element)
        || node._minimaxDomWidget
        || null;
    if (dups.length > 1) {
        for (const w of dups) {
            if (w === keep) continue;
            const idx = node.widgets.indexOf(w);
            if (idx !== -1) node.widgets.splice(idx, 1);
            try { w.onRemove?.(); } catch { /* ignore */ }
            w.element?.remove?.();
        }
    }
    if (keep) node._minimaxDomWidget = keep;
    return keep || null;
}

export function destroyDirectorEditor(node) {
    const ed = node?._minimaxEditor;
    if (!ed) {
        if (node) node._minimaxEditor = null;
        return;
    }
    try { ed.destroy(); } catch { /* ignore */ }
    if (node._minimaxEditor === ed) node._minimaxEditor = null;
    if (ed.domWidget?._minimaxEditor === ed) ed.domWidget._minimaxEditor = null;
}

export function bindDirectorDomWidgetSizing(node, widget, getEditor) {
    const editor = getEditor?.();
    const minHeight = () => getDirectorUiHeight(getEditor?.());
    // Do not set computeSize — fixed-size widgets never receive resize free space.
    try {
        delete widget.computeSize;
    } catch {
        widget.computeSize = undefined;
    }
    widget.computeLayoutSize = () => ({
        minHeight: minHeight(),
        maxHeight: undefined,
        minWidth: DIRECTOR_MIN_WIDTH,
    });
    if (widget.options) {
        widget.options.getMinHeight = minHeight;
        delete widget.options.getMaxHeight;
    }
    const el = widget.element;
    if (el) {
        el.style.minHeight = `${minHeight()}px`;
        el.style.setProperty("--comfy-widget-min-height", `${minHeight()}px`);
    }
    if (editor) bindDomWidgetContentComputeSize(editor);
}

export function patchDirectorDomWidgetLayout() {
    const canvas = app.canvas;
    if (!canvas || canvas._minimaxDirectorLayoutPatch) return;
    canvas._minimaxDirectorLayoutPatch = true;
    const prev = canvas.onDrawForeground;
    canvas.onDrawForeground = function (ctx) {
        const graph = app.graph ?? canvas.graph;
        for (const node of graph?._nodes ?? graph?.nodes ?? []) {
            if (node._minimaxEditor?.isPlaying) continue;
            ensureDirectorDomWidgetWidth(node);
        }
        return prev?.apply(this, arguments);
    };
}

export function stopDomEvent(e) {
    e.stopPropagation();
}

export function directorEditableFromEventTarget(target) {
    let node = target;
    if (node?.nodeType === Node.TEXT_NODE) node = node.parentElement;
    if (!node?.closest) return null;
    if (!node.closest(".mmx-host")) return null;
    return node.closest("input, textarea, select, [contenteditable='true'], .bd-token-editor");
}

export function installDirectorClipboardGuard() {
    if (typeof document === "undefined" || document.__mmxDirectorClipboardGuard) return;
    document.__mmxDirectorClipboardGuard = true;

    const blockBubbleToCanvas = (e) => {
        if (!directorEditableFromEventTarget(e.target)
            && !directorEditableFromEventTarget(document.activeElement)) {
            return;
        }
        e.stopImmediatePropagation();
    };

    for (const type of ["paste", "copy", "cut"]) {
        document.addEventListener(type, blockBubbleToCanvas, true);
    }
    document.addEventListener("keydown", (e) => {
        if (!(e.ctrlKey || e.metaKey)) return;
        const k = e.key?.toLowerCase?.();
        if (k !== "v" && k !== "c" && k !== "x") return;
        blockBubbleToCanvas(e);
    }, true);

    const patchPaste = () => {
        const canvas = app.canvas;
        if (!canvas) return;
        const wrap = (obj, key) => {
            if (!obj || typeof obj[key] !== "function" || obj[key].__mmxDirectorPatched) return;
            const orig = obj[key];
            const patched = function (...args) {
                if (directorEditableFromEventTarget(document.activeElement)) return null;
                return orig.apply(this, args);
            };
            patched.__mmxDirectorPatched = true;
            obj[key] = patched;
        };
        wrap(canvas, "pasteFromClipboard");
        wrap(canvas.constructor?.prototype, "pasteFromClipboard");
        // Some frontend builds expose paste on the LiteGraph canvas proto only.
        try {
            const LG = globalThis.LiteGraph?.LGraphCanvas?.prototype;
            wrap(LG, "pasteFromClipboard");
        } catch {
            /* ignore */
        }
    };
    patchPaste();
    queueMicrotask(patchPaste);
    setTimeout(patchPaste, 0);
    setTimeout(patchPaste, 500);
}

export function hideWidget(w) {
    if (!w) return;
    // Group headers in HIDDEN_WIDGETS duplicate timeline panel sections — hide them too.
    if (w._bdGroupHeader && !HIDDEN_WIDGETS.includes(w.name)) return;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    w.computeSize = () => [0, 0];
    if (w.element) w.element.style.display = "none";
}

export function parseTimeline(raw, totalFrames, fps) {
    const total = totalFrames || 124;
    const base = {
        version: 4,
        editMode: "global",
        totalFrames: total,
        frameRate: coerceTimelineFps(fps || 24),
        video: {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
        },
        videoClips: [],
        global: {
            taskType: "", prompt: "", refs: [], refAudios: [], referenceVideo: {},
            continuousReference: false, commonEnabled: false, commonCollapsed: false,
        },
        output: {
            // v2v/rv2v default: scale by long edge (preserve aspect). Fixed = center-crop.
            mode: "long_edge",
            aspectRatio: DEFAULT_ASPECT_RATIO,
            megapixels: DEFAULT_MEGAPIXELS,
            multiple: MINIMAX_CANVAS_MULTIPLE,
            longEdge: 848, width: 848, height: 480,
            maxExportFrames: 0, exportMode: "all",
            audioMode: "generate",
            refImageSize: "match",
            continuityEnabled: false, continuityOverlapFrames: DEFAULT_CONTINUITY_FRAMES,
        },
        runSelectEnabled: false,
        runSelection: [],
        liveTaePreview: true,
        batchDetailMode: "solo",
        segments: [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], refAudios: [], referenceVideo: {} }],
    };
    if (!raw?.trim()) return base;
    try {
        const data = JSON.parse(raw);
        data.version = data.version || 4;
        data.editMode = data.editMode || "global";
        data.frameRate = coerceTimelineFps(data.frameRate ?? fps ?? 24);
        data.video = data.video || { fileName: "", frames: [] };
        if (!data.video.videoFile && data.video.fileName) {
            data.video.videoFile = data.video.fileName;
        }
        data.video.type = data.video.type || "input";
        data.video.subfolder = data.video.subfolder || "";
        data.video.frames = data.video.frames || [];
        data.global = data.global || {
            refs: [], refAudios: [], referenceVideo: {},
            continuousReference: false, commonEnabled: false, commonCollapsed: false,
        };
        data.global.refs = data.global.refs || [];
        data.global.refAudios = data.global.refAudios || data.global.ref_audios || [];
        data.global.refVideos = data.global.refVideos || data.global.ref_videos || [];
        if (Array.isArray(data.global.refVideos)) {
            data.global.refVideos = data.global.refVideos.map(sanitizeRefVideo);
        }
        data.global.referenceVideo = data.global.referenceVideo || data.global.reference_video || {};
        data.global.continuousReference = !!data.global.continuousReference || !!data.global.continuous_reference;
        // r2v shared params: default OFF unless explicitly enabled.
        data.global.commonEnabled = !!(
            data.global.commonEnabled ?? data.global.common_enabled
        );
        // UI fold only — does not affect runtime merge when commonEnabled is true.
        data.global.commonCollapsed = !!(
            data.global.commonCollapsed ?? data.global.common_collapsed
        );
        const legacyRef = data.referenceVideo || data.reference_video;
        if (legacyRef && (legacyRef.videoFile || legacyRef.fileName)
            && !(data.global.referenceVideo.videoFile || data.global.referenceVideo.fileName)) {
            data.global.referenceVideo = { ...legacyRef };
        }
        delete data.referenceVideo;
        delete data.reference_video;
        data.output = normalizeOutputContinuity({
            mode: data.output?.mode || "long_edge",
            // Keep ResolutionSelector fields across reload (were previously dropped → always 16:9).
            aspectRatio: data.output?.aspectRatio != null
                ? normalizeAspectRatioLabel(data.output.aspectRatio)
                : undefined,
            megapixels: data.output?.megapixels ?? data.output?.megaPixels ?? undefined,
            multiple: data.output?.multiple ?? MINIMAX_CANVAS_MULTIPLE,
            longEdge: data.output?.longEdge ?? data.output?.long_edge ?? data.refMaxSize ?? 848,
            width: data.output?.width ?? data.width ?? 864,
            height: data.output?.height ?? data.height ?? 480,
            maxExportFrames: data.output?.maxExportFrames ?? data.output?.max_export_frames ?? 0,
            exportMode: data.output?.exportMode ?? data.output?.export_mode ?? "all",
            audioMode: normalizeAudioMode(data.output?.audioMode ?? data.output?.audio_mode),
            refImageSize: normalizeRefImageSize(data.output?.refImageSize ?? data.output?.ref_image_size),
            continuityEnabled: data.output?.continuityEnabled ?? data.output?.continuity_enabled,
            continuityOverlapFrames: data.output?.continuityOverlapFrames ?? data.output?.continuity_overlap_frames,
        });
        // Infer aspectRatio from saved width/height when older payloads omitted the label.
        if (!data.output.aspectRatio && data.output.width > 0 && data.output.height > 0) {
            const rw = data.output.width;
            const rh = data.output.height;
            const match = RESOLUTION_ASPECTS.find(([, aw, ah]) => Math.abs(rw / rh - aw / ah) < 0.02);
            data.output.aspectRatio = match ? match[0] : CUSTOM_ASPECT_RATIO;
        }
        if (!data.output.aspectRatio) data.output.aspectRatio = DEFAULT_ASPECT_RATIO;
        if (data.output.megapixels == null) data.output.megapixels = DEFAULT_MEGAPIXELS;
        stripTimelineContinuityRootFields(data);
        stripTimelineEphemeralFields(data);
        const legacyFrames = data.video.frames?.length || 0;
        if (!data.video.frameMap?.length) {
            const n = data.totalFrames || data.video.sourceFrameCount || legacyFrames || total;
            data.totalFrames = n;
            data.video.sourceFrameCount = data.video.sourceFrameCount || n;
            data.video.deletedSourceRanges = data.video.deletedSourceRanges || [];
            data.video.frameMap = [];
        }
        if (!data.segments?.length) {
            const n = data.totalFrames || data.video.sourceFrameCount || legacyFrames || total;
            data.segments = [{ id: uid(), start: 0, length: Math.max(MIN_SEG, n), prompt: "", taskType: "", refs: [], refAudios: [], referenceVideo: {} }];
        }
        for (const seg of data.segments) {
            if (!seg.id) seg.id = uid();
            if (seg.length == null && seg.end != null) seg.length = seg.end - seg.start;
            if (seg.frameCount == null && seg.length != null) seg.frameCount = seg.length;
            seg.refs = seg.refs || [];
            seg.refAudios = seg.refAudios || seg.ref_audios || [];
            seg.referenceVideo = seg.referenceVideo || seg.reference_video || {};
            seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
            seg.negativePrompt = seg.negativePrompt ?? "";
        }
        data.gen = data.gen || { defaultFrameCount: 124 };
        if (data.global) {
            data.global.genImage = data.global.genImage || { imageFile: data.global.imageFile || "" };
        }
        data.runSelectEnabled = !!data.runSelectEnabled;
        data.runSelection = Array.isArray(data.runSelection) ? data.runSelection.map((i) => parseInt(i, 10)).filter((i) => i >= 0) : [];
        // Default on when missing (older timelines).
        data.liveTaePreview = data.liveTaePreview !== false && data.live_tae_preview !== false;
        const detailMode = data.batchDetailMode ?? data.batch_detail_mode;
        data.batchDetailMode = detailMode === "all" ? "all" : "solo";
        if (data.timelineMode === "fl2v" || resolveTaskKey(data.global?.taskType || "") === "fl2v") {
            data.timelineMode = "fl2v";
            data.editMode = "segment";
            data.keyframes = Array.isArray(data.keyframes) ? data.keyframes : [];
            data.shots = Array.isArray(data.shots) ? data.shots : [];
            const stored = parseInt(data.totalFrames, 10);
            const farthest = Math.max(
                0,
                ...(data.segments || []).map((s) => (parseInt(s.start, 10) || 0) + (parseInt(s.length ?? s.frameCount, 10) || 0)),
                ...(data.keyframes || []).map((k) => (parseInt(k.start, 10) || 0) + (parseInt(k.frameCount ?? k.length, 10) || 0)),
            );
            data.totalFrames = (Number.isFinite(stored) && stored > 0)
                ? stored
                : Math.max(farthest, total, 240);
            return data;
        }
        if (data.timelineMode === "image_batch" || data.timelineMode === "prompt_batch") {
            data.timelineMode = "prompt_batch";
            data.editMode = "segment";
            data.totalFrames = sumFrameCounts(data.segments) || data.totalFrames || total;
            return data;
        }
        if (data.timelineMode === "gen_blank" || data.timelineMode === "gen_image") {
            const gkey = resolveTaskKey(data.global?.taskType || "");
            if (isPromptBatchTask(gkey)) {
                data.timelineMode = "prompt_batch";
                data.editMode = "segment";
            }
            data.totalFrames = sumFrameCounts(data.segments) || data.totalFrames || total;
            return data;
        }
        if (!data.videoClips?.length && data.video?.videoFile) {
            data.videoClips = [{
                id: data.video.id || uid(),
                fileName: data.video.fileName || "",
                videoFile: data.video.videoFile || data.video.fileName || "",
                subfolder: data.video.subfolder || "",
                type: data.video.type || "input",
                width: data.video.width || 0,
                height: data.video.height || 0,
                duration: data.video.duration || 0,
                nativeFps: data.video.nativeFps || data.video.native_fps || 0,
                nativeFrameCount: data.video.nativeFrameCount || data.video.native_frame_count || 0,
                sourceFrameCount: data.video.sourceFrameCount || data.video.frameMap?.length || 0,
                storageWidth: data.video.storageWidth,
                storageHeight: data.video.storageHeight,
            }];
        }
        data.videoClips = data.videoClips || [];
        data.totalFrames = data.totalFrames || data.video.sourceFrameCount || data.video.frameMap?.length || total;
        return data;
    } catch {
        return base;
    }
}
