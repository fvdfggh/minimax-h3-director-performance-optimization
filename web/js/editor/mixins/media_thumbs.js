/** media_thumbs mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { uid } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { inputViewUrl } from "../urls.js";
export const media_thumbsMixin = {
    _videoIdentityFromParts(video, clips) {
        const list = Array.isArray(clips) && clips.length ? clips : [];
        if (list.length) {
            return list
                .map((c) => `${c?.type || "input"}:${c?.videoFile || c?.fileName || ""}`)
                .filter((id) => id && id !== "input:");
        }
        const v = video || {};
        const id = `${v.type || "input"}:${v.videoFile || v.fileName || ""}`;
        return id && id !== "input:" ? [id] : [];
    },
    _clipThumbIdentity(clipIndex = 0) {
        const clips = this.getVideoClips();
        const c = clips[clipIndex] || clips[0] || this.timeline?.video || {};
        const id = `${c.type || "input"}:${c.videoFile || c.fileName || ""}`;
        return id === "input:" ? "" : id;
    },
    _videoThumbIdentity() {
        return this._videoIdentityFromParts(this.timeline?.video, this.timeline?.videoClips).join("|");
    },
    _liveVideoFileIdentities() {
        return this._videoIdentityFromParts(this.timeline?.video, this.timeline?.videoClips);
    },
    _knownVideoFileIdentities() {
        const ids = new Set(this._liveVideoFileIdentities());
        for (const ws of Object.values(this._videoWsMem || {})) {
            for (const id of this._videoIdentityFromParts(ws?.video, ws?.videoClips)) ids.add(id);
        }
        for (const ws of Object.values(this.timeline?.videoWorkspaces || {})) {
            for (const id of this._videoIdentityFromParts(ws?.video, ws?.videoClips)) ids.add(id);
        }
        return ids;
    },
    _frameThumbKey(logicalFrame) {
        const entry = this.getFrameMapEntry(logicalFrame);
        const id = this._clipThumbIdentity(entry.clip) || this._videoThumbIdentity() || "none";
        if (this._legacyFrames.length) return `${id}#legacy:${logicalFrame}`;
        return `${id}#${entry.clip}:${entry.frame}`;
    },
    _dropThumbsForIdentity(identity) {
        if (!identity) return;
        const prefix = `${identity}#`;
        for (const key of [...this._thumbCache.keys()]) {
            if (key === identity || String(key).startsWith(prefix)) this._thumbCache.delete(key);
        }
        for (const key of [...this._thumbPending]) {
            if (key === identity || String(key).startsWith(prefix)) this._thumbPending.delete(key);
        }
    },
    _dropThumbsIfUnused(identities) {
        const list = Array.isArray(identities) ? identities : [identities];
        const used = this._knownVideoFileIdentities();
        for (const id of list) {
            if (!id || used.has(id)) continue;
            this._dropThumbsForIdentity(id);
        }
    },
    _flushPendingThumbDrops() {
        this._dropThumbsIfUnused(this._thumbIdsPendingDrop);
        this._thumbIdsPendingDrop = [];
    },
    _invalidateVideoThumbs() {
        this._thumbCache.clear();
        this._thumbPending.clear();
    },
    _usesSourceVideoThumbs() {
        return this.getDirectorMode() === "video";
    },
    hasVideo() {
        const v = this.timeline?.video || {};
        return !!(this.getVideoClips().length || v.videoFile || this._legacyFrames.length || v.frames?.length);
    },
    /** v2v / rv2v empty canvas: placeholder says click to upload. */
    needsSourceVideoUpload() {
        return this.getDirectorMode() === "video" && !this.hasVideo();
    },
    getVideoClips() {
        if (this.timeline.videoClips?.length) return this.timeline.videoClips;
        const v = this.timeline?.video || {};
        if (v.videoFile || v.fileName) {
            return [{
                id: v.id || "c0",
                fileName: v.fileName || "",
                videoFile: v.videoFile || v.fileName || "",
                subfolder: v.subfolder || "",
                type: v.type || "input",
                width: v.width || 0,
                height: v.height || 0,
                duration: v.duration || 0,
                nativeFps: v.nativeFps || v.native_fps || 0,
                nativeFrameCount: v.nativeFrameCount || v.native_frame_count || 0,
                sourceFrameCount: v.sourceFrameCount || this.getFrameMap().length,
                storageWidth: v.storageWidth,
                storageHeight: v.storageHeight,
            }];
        }
        return [];
    },
    _ensureVideoClipsArray() {
        if (!this.timeline.videoClips?.length) {
            const v = this.timeline?.video || {};
            if (v.videoFile || v.fileName) {
                this.timeline.videoClips = [{
                    id: v.id || uid(),
                    fileName: v.fileName || "",
                    videoFile: v.videoFile || v.fileName || "",
                    subfolder: v.subfolder || "",
                    type: v.type || "input",
                    width: v.width || 0,
                    height: v.height || 0,
                    duration: v.duration || 0,
                    nativeFps: v.nativeFps || v.native_fps || 0,
                    nativeFrameCount: v.nativeFrameCount || v.native_frame_count || 0,
                    sourceFrameCount: v.sourceFrameCount || this.getFrameMap().length,
                    storageWidth: v.storageWidth,
                    storageHeight: v.storageHeight,
                }];
            } else {
                this.timeline.videoClips = [];
            }
        }
    },
    getClipViewUrl(clipIndex) {
        const clip = this.getVideoClips()[clipIndex];
        if (!clip?.videoFile) return "";
        return inputViewUrl(clip.videoFile, clip.type || "input");
    },
    getRefVideoTarget() {
        if (this.isGlobalMode()) {
            this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
            if (!this.timeline.global.referenceVideo) this.timeline.global.referenceVideo = {};
            return this.timeline.global;
        }
        const seg = this.timeline.segments[this.selectedIndex];
        if (seg) {
            if (!seg.referenceVideo) seg.referenceVideo = {};
            return seg;
        }
        this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
        return this.timeline.global;
    },
    getReferenceVideoViewUrl(ref) {
        const block = ref || {};
        const file = block.videoFile || block.fileName;
        if (!file) return "";
        return inputViewUrl(file, block.type || "input");
    },
    _stopRefVideoPreviews(onlyEls = null) {
        const targets = onlyEls || [this.globalRefVideo, this.segRefVideo];
        for (const el of targets) {
            const v = el?.querySelector("video");
            if (v) {
                v.pause();
                v.removeAttribute("src");
                v.load();
            }
        }
    }
};
