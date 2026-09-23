/** Narrowing the live editor timeline into the payload the backend receives.
 *
 * The in-memory timeline carries far more than the node needs: per-frame maps,
 * cached thumbnails, run status, transient UI flags, and paths that only make
 * sense on this machine. Every helper here is a *pure* narrowing step — no DOM, no
 * globals, no i18n — so the payload rules can be read and changed on their own.
 *
 * Two invariants are easy to break by accident:
 *
 * * media references must survive a save / reload round-trip, so an absent ref is
 *   dropped rather than written as a null;
 * * nothing ephemeral may reach the payload — a run's thumbnails or a stale
 *   selection must never be persisted into the workflow.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { normalizeRefImageSize, resolveTaskKey } from "../minimax_gen_timeline.js";
import { t } from "../minimax_i18n.js";
export function isContinuityEnabled(output) {
    if (!output) return false;
    const raw = output.continuityEnabled ?? output.continuity_enabled;
    if (raw === true || raw === 1) return true;
    if (typeof raw === "string") {
        const s = raw.trim().toLowerCase();
        return s === "true" || s === "1" || s === "yes" || s === "on";
    }
    return false;
}

/** Whether段间引导 controls apply for the current task + segment count. */
export function isContinuityEligible(editor) {
    if (!editor) return false;
    const taskKey = resolveTaskKey(
        editor.getTaskKey?.() || editor.taskTypeWidget?.value || editor.globalTask?.value || "",
    );
    if (!CONTINUITY_TASKS.has(taskKey)) return false;
    const segCount = Array.isArray(editor.timeline?.segments) ? editor.timeline.segments.length : 0;
    const fl2vCount = Array.isArray(editor.timeline?.fl2vGroups) ? editor.timeline.fl2vGroups.length : 0;
    const r2vCount = Array.isArray(editor.timeline?.r2vGroups) ? editor.timeline.r2vGroups.length : 0;
    return Math.max(segCount, fl2vCount, r2vCount) >= 2;
}

export function normalizeAudioMode(value) {
    const raw = String(value || "generate").trim().toLowerCase();
    if (raw === "source" || raw === "original" || raw === "passthrough") return "source";
    if (raw === "mute" || raw === "silent" || raw === "silence") return "mute";
    return "generate";
}

const CONTINUITY_FRAME_CHOICES = [5, 22, 39, 56];
/** Official Motion Context baseline recommendation. */
export const DEFAULT_CONTINUITY_FRAMES = 22;
const CONTINUITY_TASKS = new Set(["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]);

export function snapContinuityFrames(raw) {
    const n = parseInt(raw, 10);
    const value = Number.isFinite(n) ? n : DEFAULT_CONTINUITY_FRAMES;
    let best = DEFAULT_CONTINUITY_FRAMES;
    let bestDist = Infinity;
    for (const choice of CONTINUITY_FRAME_CHOICES) {
        const dist = Math.abs(choice - value);
        if (dist < bestDist || (dist === bestDist && choice > best)) {
            best = choice;
            bestDist = dist;
        }
    }
    return best;
}

export function normalizeOutputContinuity(output = {}) {
    const rawOverlap = output.continuityOverlapFrames ?? output.continuity_overlap_frames ?? DEFAULT_CONTINUITY_FRAMES;
    return {
        ...output,
        continuityEnabled: isContinuityEnabled(output),
        continuityOverlapFrames: snapContinuityFrames(rawOverlap),
        audioMode: normalizeAudioMode(output.audioMode ?? output.audio_mode),
        refImageSize: normalizeRefImageSize(output.refImageSize ?? output.ref_image_size),
    };
}

export function stripTimelineContinuityRootFields(timeline) {
    if (!timeline || typeof timeline !== "object") return;
    delete timeline.continuityEnabled;
    delete timeline.continuity_enabled;
    delete timeline.continuityOverlapFrames;
    delete timeline.continuity_overlap_frames;
}

/** Drop ephemeral UI-only fields so they never persist in timeline_data. */
export function sanitizeRefImage(ref) {
    if (!ref || typeof ref !== "object") return ref;
    const index = Number(ref.index ?? ref.slot);
    return {
        index: Number.isFinite(index) ? index : 0,
        imageFile: ref.imageFile || "",
        fileName: ref.fileName || "",
        type: ref.type || "input",
        subfolder: ref.subfolder || "",
    };
}

export function sanitizeRefAudio(ref) {
    if (!ref || typeof ref !== "object") return ref;
    const index = Number(ref.index ?? ref.slot);
    return {
        index: Number.isFinite(index) ? index : 0,
        audioFile: ref.audioFile || "",
        fileName: ref.fileName || "",
        type: ref.type || "input",
        subfolder: ref.subfolder || "",
        durationSec: ref.durationSec,
    };
}

export function sanitizeRefVideo(ref) {
    if (!ref || typeof ref !== "object") return ref;
    const index = Number(ref.index ?? ref.slot);
    return {
        index: Number.isFinite(index) ? index : 0,
        videoFile: ref.videoFile || "",
        fileName: ref.fileName || "",
        type: ref.type || "input",
        subfolder: ref.subfolder || "",
        durationSec: ref.durationSec,
        pairedAudioFile: ref.pairedAudioFile || "",
        previewImageFile: ref.previewImageFile || "",
        previewImageUrl: ref.previewImageUrl || "",
        linked: !!ref.linked || !!(ref.videoFile || ref.previewImageFile || ref.previewImageUrl),
    };
}

export function sanitizeSegmentForPayload(seg) {
    if (!seg || typeof seg !== "object") return seg;
    const {
        previewB64,
        previewFrames,
        imageB64,
        ...rest
    } = seg;
    return {
        ...rest,
        refs: Array.isArray(rest.refs) ? rest.refs.map(sanitizeRefImage) : [],
        refAudios: Array.isArray(rest.refAudios) ? rest.refAudios.map(sanitizeRefAudio) : [],
        refVideos: Array.isArray(rest.refVideos) ? rest.refVideos.map(sanitizeRefVideo) : [],
        genImage: rest.genImage
            ? { imageFile: rest.genImage.imageFile || "", fileName: rest.genImage.fileName || "" }
            : undefined,
        referenceVideo: rest.referenceVideo
            ? {
                videoFile: rest.referenceVideo.videoFile || "",
                fileName: rest.referenceVideo.fileName || "",
                type: rest.referenceVideo.type || "input",
                subfolder: rest.referenceVideo.subfolder || "",
            }
            : undefined,
    };
}

export function cloneJson(value, fallback) {
    try {
        if (value == null) return fallback;
        return JSON.parse(JSON.stringify(value));
    } catch {
        return fallback;
    }
}

export function sanitizeBatchGlobalCommon(gc) {
    const src = gc && typeof gc === "object" ? gc : {};
    return {
        commonEnabled: !!src.commonEnabled,
        commonCollapsed: !!src.commonCollapsed,
        prompt: src.prompt || "",
        refs: Array.isArray(src.refs) ? src.refs.map(sanitizeRefImage) : [],
        refAudios: Array.isArray(src.refAudios) ? src.refAudios.map(sanitizeRefAudio) : [],
        refVideos: Array.isArray(src.refVideos)
            ? src.refVideos.map(sanitizeRefVideo)
            : (Array.isArray(src.ref_videos) ? src.ref_videos.map(sanitizeRefVideo) : []),
    };
}

/** Persistable t2v/i2v/r2v snapshot (no preview frames). */
export function sanitizeBatchWorkspace(ws) {
    if (!ws || typeof ws !== "object" || !Array.isArray(ws.segments) || !ws.segments.length) {
        return null;
    }
    return {
        selectedIndex: Number.isFinite(Number(ws.selectedIndex)) ? Number(ws.selectedIndex) : 0,
        editMode: ws.editMode || "segment",
        runSelectEnabled: !!ws.runSelectEnabled,
        runSelection: Array.isArray(ws.runSelection) ? [...ws.runSelection] : [],
        segments: ws.segments.map(sanitizeSegmentForPayload),
        globalCommon: sanitizeBatchGlobalCommon(ws.globalCommon),
    };
}

export function sanitizeVideoMedia(video) {
    if (!video || typeof video !== "object") return video;
    const hasFile = !!(video.videoFile || video.fileName);
    const dropFrames = hasFile
        || (Array.isArray(video.frames) && video.frames.length > 8);
    return { ...video, frames: dropFrames ? [] : (video.frames || []) };
}

/** Persistable v2v/rv2v snapshot (no decoded frame blobs). */
export function sanitizeVideoWorkspace(ws) {
    if (!ws || typeof ws !== "object") return null;
    return {
        selectedIndex: Number.isFinite(Number(ws.selectedIndex)) ? Number(ws.selectedIndex) : 0,
        currentFrame: Math.max(0, Number(ws.currentFrame) || 0),
        editMode: ws.editMode || "global",
        runSelectEnabled: !!ws.runSelectEnabled,
        runSelection: Array.isArray(ws.runSelection) ? [...ws.runSelection] : [],
        totalFrames: ws.totalFrames,
        frameRate: ws.frameRate,
        storageWidth: ws.storageWidth || 0,
        storageHeight: ws.storageHeight || 0,
        segments: Array.isArray(ws.segments) ? ws.segments.map(sanitizeSegmentForPayload) : [],
        video: sanitizeVideoMedia(ws.video || {}),
        videoClips: Array.isArray(ws.videoClips) ? ws.videoClips.map(sanitizeVideoMedia) : [],
        globalCommon: sanitizeBatchGlobalCommon(ws.globalCommon),
    };
}

export function stripTimelineEphemeralFields(timeline) {
    if (!timeline || typeof timeline !== "object") return;
    delete timeline.videoWorkspace;
    delete timeline.batchWorkspace;
    delete timeline.fl2vWorkspace;
    if (timeline.batchWorkspaces && typeof timeline.batchWorkspaces === "object") {
        const cleaned = {};
        for (const [key, ws] of Object.entries(timeline.batchWorkspaces)) {
            const safe = sanitizeBatchWorkspace(ws);
            if (safe) cleaned[key] = safe;
        }
        timeline.batchWorkspaces = cleaned;
    }
    if (timeline.videoWorkspaces && typeof timeline.videoWorkspaces === "object") {
        const cleaned = {};
        for (const [key, ws] of Object.entries(timeline.videoWorkspaces)) {
            const safe = sanitizeVideoWorkspace(ws);
            if (safe) cleaned[key] = safe;
        }
        timeline.videoWorkspaces = cleaned;
    }
    // Shallow-cloned payloads still share nested refs with live state — reassign, don't mutate.
    if (Array.isArray(timeline.segments)) {
        timeline.segments = timeline.segments.map(sanitizeSegmentForPayload);
    }
    if (timeline.global && typeof timeline.global === "object") {
        timeline.global = {
            ...timeline.global,
            refs: Array.isArray(timeline.global.refs)
                ? timeline.global.refs.map(sanitizeRefImage)
                : [],
            refAudios: Array.isArray(timeline.global.refAudios)
                ? timeline.global.refAudios.map(sanitizeRefAudio)
                : [],
            refVideos: Array.isArray(timeline.global.refVideos)
                ? timeline.global.refVideos.map(sanitizeRefVideo)
                : (Array.isArray(timeline.global.ref_videos)
                    ? timeline.global.ref_videos.map(sanitizeRefVideo)
                    : []),
            referenceVideo: timeline.global.referenceVideo
                ? {
                    videoFile: timeline.global.referenceVideo.videoFile || "",
                    fileName: timeline.global.referenceVideo.fileName || "",
                    type: timeline.global.referenceVideo.type || "input",
                    subfolder: timeline.global.referenceVideo.subfolder || "",
                }
                : timeline.global.referenceVideo,
        };
    }
    if (timeline.video && typeof timeline.video === "object") {
        const hasFile = !!(timeline.video.videoFile || timeline.video.fileName);
        const dropFrames = hasFile
            || (Array.isArray(timeline.video.frames) && timeline.video.frames.length > 8);
        timeline.video = {
            ...timeline.video,
            frames: dropFrames ? [] : (timeline.video.frames || []),
        };
    }
    if (Array.isArray(timeline.videoClips)) {
        timeline.videoClips = timeline.videoClips.map((clip) => {
            if (!clip || typeof clip !== "object") return clip;
            const dropFrames = !!(clip.videoFile || clip.fileName)
                || (Array.isArray(clip.frames) && clip.frames.length > 8);
            return dropFrames ? { ...clip, frames: [] } : { ...clip };
        });
    }
}
