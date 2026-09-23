/** workspaces mixin for the Director editor (workspaces).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { cloneJson, sanitizeBatchWorkspace, sanitizeVideoWorkspace } from "../../core/timeline_sanitize.js";
import { clamp } from "../../core/utils.js";
import { defaultDurationSec, getDirectorMode, isVideoBatchTask, newBatchSegment, resolveTaskKey } from "../../minimax_gen_timeline.js";
import { flushBatchPromptInputs } from "../../minimax_image_batch.js";
export const workspacesMixin = {
    /** Snapshot v2v/rv2v workspace for a specific task key (session + persist). */
    _captureVideoWorkspace() {
        const video = this.timeline.video || {};
        const clips = this.timeline.videoClips || [];
        const g = this.timeline.global || {};
        return {
            segments: cloneJson(this.timeline.segments || [], []),
            selectedIndex: this.selectedIndex,
            currentFrame: this.currentFrame,
            editMode: this.timeline.editMode || "global",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : [],
            video: cloneJson(video, {}),
            videoClips: cloneJson(clips, []),
            totalFrames: this.timeline.totalFrames ?? this.getTotalFrames(),
            frameRate: this.timeline.frameRate ?? this.getFrameRate(),
            legacyFrames: this._legacyFrames?.length ? [...this._legacyFrames] : [],
            storageWidth: this._storageWidth || 0,
            storageHeight: this._storageHeight || 0,
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: cloneJson(g.refs, []),
                refAudios: cloneJson(g.refAudios || g.ref_audios, []),
                refVideos: cloneJson(g.refVideos || g.ref_videos, []),
            },
        };
    },
    _applyVideoWorkspace(ws) {
        if (!ws || typeof ws !== "object") return false;
        if (ws.video && typeof ws.video === "object") {
            this.timeline.video = cloneJson(ws.video, {});
        } else {
            this.timeline.video = {
                fileName: "", videoFile: "", subfolder: "", type: "input", frames: [], frameMap: [],
            };
        }
        this.timeline.videoClips = Array.isArray(ws.videoClips) ? cloneJson(ws.videoClips, []) : [];
        this.timeline.segments = Array.isArray(ws.segments) ? cloneJson(ws.segments, []) : [];
        if (ws.totalFrames != null) this.timeline.totalFrames = ws.totalFrames;
        if (ws.frameRate != null) this.timeline.frameRate = ws.frameRate;
        this.timeline.editMode = ws.editMode || "global";
        this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        this.timeline.runSelection = Array.isArray(ws.runSelection) ? [...ws.runSelection] : [];
        const gc = ws.globalCommon || {};
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.commonEnabled = !!gc.commonEnabled;
        this.timeline.global.commonCollapsed = !!gc.commonCollapsed;
        this.timeline.global.prompt = gc.prompt || "";
        this.timeline.global.refs = cloneJson(gc.refs, []);
        this.timeline.global.refAudios = cloneJson(gc.refAudios, []);
        this.timeline.global.refVideos = cloneJson(gc.refVideos, []);
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt || "";
        this.selectedIndex = clamp(
            ws.selectedIndex ?? 0,
            0,
            Math.max(0, (this.timeline.segments?.length || 1) - 1),
        );
        this.currentFrame = Math.max(0, ws.currentFrame ?? 0);
        if (Array.isArray(ws.legacyFrames) && ws.legacyFrames.length) {
            this._legacyFrames = [...ws.legacyFrames];
        } else {
            this._legacyFrames = [];
        }
        if (ws.storageWidth) this._storageWidth = ws.storageWidth;
        if (ws.storageHeight) this._storageHeight = ws.storageHeight;

        this.normalizeSegments();
        this.restoreVideoFromTimeline();
        const total = this.getTotalFrames();
        this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, total - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = this.currentFrame;
        }
        if (this.totalFramesWidget) this.totalFramesWidget.value = total;
        this.updateVideoNameLabel();
        this.updateStageVisibility();
        return true;
    },
    _resetVideoWorkspaceLive() {
        this._clearVideoState();
        this.timeline.segments = [];
        this.timeline.editMode = "global";
        this.selectedIndex = 0;
        this.currentFrame = 0;
        this._clearLiveRunSelection();
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.prompt = "";
        this.timeline.global.refs = [];
        this.timeline.global.refAudios = [];
        this.timeline.global.refVideos = [];
        this.timeline.global.referenceVideo = {};
        this.timeline.global.continuousReference = false;
        this.timeline.global.commonEnabled = false;
        this.timeline.global.commonCollapsed = false;
        if (this.globalPrompt) this.globalPrompt.value = "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = "";
        this.updateVideoNameLabel();
    },
    _stashVideoWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        if (getDirectorMode(key) !== "video") return;
        this._videoWsMem = this._videoWsMem || {};
        const full = this._captureVideoWorkspace();
        this._videoWsMem[key] = full;
        this.timeline.videoWorkspaces = this.timeline.videoWorkspaces || {};
        const safe = sanitizeVideoWorkspace(full);
        if (safe) this.timeline.videoWorkspaces[key] = safe;
    },
    _persistCurrentVideoWorkspace() {
        if (this.getDirectorMode() !== "video") return;
        const key = this.getTaskKey();
        if (getDirectorMode(key) !== "video") return;
        const g = this.timeline.global || {};
        const safe = sanitizeVideoWorkspace({
            segments: this.timeline.segments || [],
            selectedIndex: this.selectedIndex,
            currentFrame: this.currentFrame,
            editMode: this.timeline.editMode || "global",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: this.timeline.runSelection,
            video: this.timeline.video || {},
            videoClips: this.timeline.videoClips || [],
            totalFrames: this.timeline.totalFrames ?? this.getTotalFrames(),
            frameRate: this.timeline.frameRate ?? this.getFrameRate(),
            storageWidth: this._storageWidth || 0,
            storageHeight: this._storageHeight || 0,
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: g.refs,
                refAudios: g.refAudios || g.ref_audios,
                refVideos: g.refVideos || g.ref_videos,
            },
        });
        if (!safe) return;
        this.timeline.videoWorkspaces = this.timeline.videoWorkspaces || {};
        this.timeline.videoWorkspaces[key] = safe;
    },
    _restoreVideoWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        const mem = this._videoWsMem?.[key];
        const persisted = this.timeline.videoWorkspaces?.[key];
        const ws = mem || persisted;
        return this._applyVideoWorkspace(ws);
    },
    _switchToVideoTaskWorkspace(prevTaskKey, currentKey) {
        const prevIds = new Set(this._liveVideoFileIdentities());
        if (prevTaskKey && getDirectorMode(prevTaskKey) === "video" && prevTaskKey !== currentKey) {
            this._stashVideoWorkspace(prevTaskKey);
            this._clearLiveRunSelection();
        }
        const nextWs = this._videoWsMem?.[currentKey] || this.timeline.videoWorkspaces?.[currentKey];
        const nextIds = new Set(this._videoIdentityFromParts(nextWs?.video, nextWs?.videoClips));
        const sameFiles = prevIds.size === nextIds.size && [...prevIds].every((id) => nextIds.has(id));
        if (!sameFiles) this._clearPreviewVideos?.(true);
        if (this._restoreVideoWorkspace(currentKey)) return;
        this._resetVideoWorkspaceLive();
    },
    _captureBatchWorkspace() {
        const g = this.timeline.global || {};
        return {
            segments: cloneJson(this.timeline.segments || [], []),
            selectedIndex: this.selectedIndex,
            editMode: this.timeline.editMode || "segment",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : [],
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: cloneJson(g.refs, []),
                refAudios: cloneJson(g.refAudios || g.ref_audios, []),
                refVideos: cloneJson(g.refVideos || g.ref_videos, []),
            },
        };
    },
    _applyBatchWorkspace(ws) {
        if (!ws || !Array.isArray(ws.segments) || !ws.segments.length) return false;
        this.timeline.segments = cloneJson(ws.segments, []);
        this.timeline.editMode = ws.editMode || "segment";
        this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        this.timeline.runSelection = Array.isArray(ws.runSelection) ? [...ws.runSelection] : [];
        const gc = ws.globalCommon || {};
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.commonEnabled = !!gc.commonEnabled;
        this.timeline.global.commonCollapsed = !!gc.commonCollapsed;
        this.timeline.global.prompt = gc.prompt || "";
        this.timeline.global.refs = cloneJson(gc.refs, []);
        this.timeline.global.refAudios = cloneJson(gc.refAudios, []);
        this.timeline.global.refVideos = cloneJson(gc.refVideos, []);
        this.selectedIndex = clamp(
            ws.selectedIndex ?? 0,
            0,
            Math.max(0, this.timeline.segments.length - 1),
        );
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = this.timeline.global.prompt || "";
        return true;
    },
    _resetBatchWorkspaceLive(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        this.timeline.segments = [newBatchSegment({ durationSec: defaultDurationSec(key) })];
        this.timeline.editMode = "segment";
        this.selectedIndex = 0;
        this._clearLiveRunSelection();
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.commonEnabled = false;
        this.timeline.global.commonCollapsed = false;
        this.timeline.global.prompt = "";
        this.timeline.global.refs = [];
        this.timeline.global.refAudios = [];
        this.timeline.global.refVideos = [];
        if (this.globalPrompt) this.globalPrompt.value = "";
        if (this.globalPromptWidget) this.globalPromptWidget.value = "";
    },
    /** Snapshot t2v / i2v / r2v groups for a specific task key (session + persist). */
    _stashBatchWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        if (!isVideoBatchTask(key)) return;
        if (this.isImageBatch?.()) flushBatchPromptInputs(this);
        const segs = this.timeline.segments || [];
        if (!segs.length) return;
        this._batchWsMem = this._batchWsMem || {};
        const full = this._captureBatchWorkspace();
        this._batchWsMem[key] = full;
        this.timeline.batchWorkspaces = this.timeline.batchWorkspaces || {};
        const safe = sanitizeBatchWorkspace(full);
        if (safe) this.timeline.batchWorkspaces[key] = safe;
    },
    _persistCurrentBatchWorkspace() {
        if (!this.isImageBatch?.()) return;
        const key = this.getTaskKey();
        if (!isVideoBatchTask(key)) return;
        const g = this.timeline.global || {};
        const safe = sanitizeBatchWorkspace({
            segments: this.timeline.segments || [],
            selectedIndex: this.selectedIndex,
            editMode: this.timeline.editMode || "segment",
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: this.timeline.runSelection,
            globalCommon: {
                commonEnabled: !!g.commonEnabled,
                commonCollapsed: !!g.commonCollapsed,
                prompt: g.prompt || "",
                refs: g.refs,
                refAudios: g.refAudios || g.ref_audios,
                refVideos: g.refVideos || g.ref_videos,
            },
        });
        if (!safe) return;
        this.timeline.batchWorkspaces = this.timeline.batchWorkspaces || {};
        this.timeline.batchWorkspaces[key] = safe;
    },
    _restoreBatchWorkspace(taskKey) {
        const key = resolveTaskKey(taskKey || this.getTaskKey());
        const mem = this._batchWsMem?.[key];
        const persisted = this.timeline.batchWorkspaces?.[key];
        const ws = (mem?.segments?.length ? mem : null) || persisted;
        return this._applyBatchWorkspace(ws);
    },
    _switchToBatchTaskWorkspace(prevTaskKey, currentKey) {
        if (prevTaskKey && isVideoBatchTask(prevTaskKey) && prevTaskKey !== currentKey) {
            this._stashBatchWorkspace(prevTaskKey);
            this._clearLiveRunSelection();
        }
        if (this._restoreBatchWorkspace(currentKey)) return;
        const externalLocked = (currentKey === "i2v" && this.hasExternalI2vGroups?.())
            || (currentKey === "r2v" && this.hasExternalR2vGroups?.());
        if (!externalLocked) this._resetBatchWorkspaceLive(currentKey);
    }
};
