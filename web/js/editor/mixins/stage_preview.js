/** stage_preview mixin for the Director editor (stage_preview).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { syncDirectorNodeSize } from "../../core/editor_lifecycle.js";
import { normalizeFrameMapEntry, sourceToLogicalFrame } from "../../core/frame_map.js";
import { clamp } from "../../core/utils.js";
import { t } from "../../minimax_i18n.js";
export const stage_previewMixin = {
    getVideoViewUrl() {
        return this.getClipViewUrl(0);
    },
    getSourceFrameIndex(logicalFrame) {
        return this.getFrameMapEntry(logicalFrame).frame;
    },
    _previewUrlEquals(videoEl, url) {
        if (!videoEl || !url) return false;
        const src = videoEl.currentSrc || videoEl.getAttribute("src") || videoEl.src || "";
        if (!src) return false;
        try {
            return new URL(src, location.href).href === new URL(url, location.href).href;
        } catch {
            return src === url;
        }
    },
    _assignPreviewSrc(videoEl, url) {
        if (!videoEl || !url) return;
        if (this._previewUrlEquals(videoEl, url)) return;
        videoEl.pause();
        videoEl.src = url;
        videoEl.load();
    },
    _getPreviewVideoForClip(clipIndex) {
        const url = this.getClipViewUrl(clipIndex);
        if (!this._previewVideos) this._previewVideos = new Map();
        if (clipIndex === 0 && this._previewVideo && !this._previewVideos.has(0)) {
            this._previewVideos.set(0, this._previewVideo);
        }
        if (!url) return this._previewVideos.get(clipIndex) || (clipIndex === 0 ? this._previewVideo : null);
        let v = this._previewVideos.get(clipIndex);
        if (!v) {
            v = document.createElement("video");
            v.crossOrigin = "anonymous";
            v.muted = true;
            v.playsInline = true;
            v.preload = "auto";
            v.style.cssText = "position:fixed;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
            document.body.appendChild(v);
            this._previewVideos.set(clipIndex, v);
        }
        this._assignPreviewSrc(v, url);
        return v;
    },
    async _ensurePreviewReady(clipIndex, timeoutMs = 8000) {
        const v = this._getPreviewVideoForClip(clipIndex);
        if (!v) return null;
        if (v.videoWidth && v.readyState >= 2) return v;
        await new Promise((resolve) => {
            let done = false;
            const finish = () => {
                if (done) return;
                done = true;
                v.removeEventListener("loadeddata", finish);
                v.removeEventListener("canplay", finish);
                resolve();
            };
            v.addEventListener("loadeddata", finish);
            v.addEventListener("canplay", finish);
            if (v.videoWidth && v.readyState >= 2) finish();
            else setTimeout(finish, timeoutMs);
        });
        return v.videoWidth ? v : null;
    },
    _restorePreviewVideos() {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        for (let i = 0; i < clips.length; i++) this._getPreviewVideoForClip(i);
        this._previewVideo = this._previewVideos.get(0) || this._previewVideo;
    },
    _clearPreviewVideos(removeExtra = true) {
        if (!this._previewVideos) return;
        for (const [idx, v] of this._previewVideos.entries()) {
            v.pause();
            if (idx === 0 && v === this._previewVideo) {
                v.removeAttribute("src");
                v.load();
                continue;
            }
            if (removeExtra) {
                v.removeAttribute("src");
                v.load();
                v.remove();
            }
        }
        const keep = this._previewVideo;
        this._previewVideos.clear();
        if (keep) this._previewVideos.set(0, keep);
    },
    async _seekPreviewVideo(timeSec, clipIndex = 0) {
        this._seekChain = this._seekChain.then(() => new Promise((resolve) => {
            const v = this._getPreviewVideoForClip(clipIndex);
            if (!v || !(v.currentSrc || v.getAttribute("src"))) { resolve(); return; }
            const target = Math.max(0, timeSec);
            let settled = false;
            const finish = () => {
                if (settled) return;
                settled = true;
                v.removeEventListener("seeked", finish);
                resolve();
            };
            v.addEventListener("seeked", finish);
            try {
                v.currentTime = target;
            } catch {
                finish();
                return;
            }
            if (Math.abs(v.currentTime - target) < 0.02 && v.readyState >= 2) {
                finish();
            } else {
                setTimeout(finish, 1500);
            }
        }));
        return this._seekChain;
    },
    updateStageVisibility() {
        if (!this.stageEl) return;
        const show = this.hasVideo()
            && !this.isImageBatch()
            && !this.isGenMode()
            && !this.isFl2vMode();
        this.stageEl.classList.toggle("hidden", !show);
        if (!show) {
            if (this.stageVideo) {
                this.stageVideo.pause();
                this.stageVideo.classList.add("hidden");
            }
            this.stageImg?.classList.add("hidden");
            this.stageEmpty?.classList.remove("hidden");
            this.stageBadge?.classList.add("hidden");
            this._stageClipIndex = -1;
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }
        this.updateDomWidgetHeight();
        syncDirectorNodeSize(this.node, this);
    },
    _updateStageBadge(logicalFrame) {
        if (!this.stageBadge) return;
        const total = this.getTotalFrames();
        const frame = clamp(logicalFrame | 0, 0, Math.max(0, total - 1));
        const clips = this.getVideoClips();
        const entry = this.getFrameMapEntry(frame);
        const clipHint = clips.length > 1 ? t("canvas.clipHint", { n: entry.clip + 1 }) : "";
        this.stageBadge.textContent = t("player.frameOf", { cur: frame + 1, total, clip: clipHint });
        this.stageBadge.classList.remove("hidden");
    },
    _logicalRangeForClip(clipIndex) {
        const map = this.getFrameMap();
        let start = -1;
        let end = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip !== clipIndex) {
                if (start >= 0) break;
                continue;
            }
            if (start < 0) start = i;
            end = i + 1;
        }
        if (start < 0) return { start: 0, end: this.getTotalFrames() };
        return { start, end };
    },
    _logicalFromStageTime(clipIndex, timeSec) {
        const fps = Math.max(0.001, this.getFrameRate());
        const srcFrame = Math.max(0, Math.round(Number(timeSec) * fps));
        const map = this.getFrameMap();
        if (!map.length) {
            const logical = sourceToLogicalFrame(srcFrame, this.timeline.video || {});
            if (logical < 0) return -1; // source lands in a deleted gap
            return clamp(logical, 0, Math.max(0, this.getTotalFrames() - 1));
        }
        let first = -1;
        let best = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip !== clipIndex) continue;
            if (first < 0) first = i;
            if (e.frame === srcFrame) return i;
            if (e.frame <= srcFrame) best = i;
        }
        if (best >= 0) return best;
        if (first >= 0) return first;
        return 0;
    },
    /** Next logical index whose source frame is strictly after srcFrame (same clip). */
    _nextLogicalAfterSourceFrame(clipIndex, srcFrame) {
        const map = this.getFrameMap();
        if (!map.length) {
            // Sparse: walk forward until source maps to a kept logical frame.
            const total = this.getTotalFrames();
            const startLogical = sourceToLogicalFrame(srcFrame, this.timeline.video || {});
            const from = startLogical < 0 ? 0 : startLogical;
            for (let i = from; i < total; i++) {
                if (this.logicalToSourceFrame(i) > srcFrame) return i;
            }
            return -1;
        }
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.clip === clipIndex && e.frame > srcFrame) return i;
        }
        return -1;
    },
    _syncStagePreview(logicalFrame, { force = false } = {}) {
        if (!this.stageEl || this.stageEl.classList.contains("hidden")) return;
        if (!this.hasVideo()) {
            this.stageEmpty?.classList.remove("hidden");
            this.stageVideo?.classList.add("hidden");
            this.stageImg?.classList.add("hidden");
            return;
        }

        // During native playback, do not seek every tick (that causes stutter).
        // Only refresh the badge; playhead is driven from video.currentTime.
        if (this.isPlaying && !force && !this._legacyFrames.length) {
            this._updateStageBadge(logicalFrame);
            return;
        }

        const frame = clamp(logicalFrame | 0, 0, Math.max(0, this.getTotalFrames() - 1));
        const fps = Math.max(0.001, this.getFrameRate());

        if (this._legacyFrames.length) {
            const dataUrl = this._legacyFrames[frame];
            if (this.stageVideo) {
                this.stageVideo.pause();
                this.stageVideo.classList.add("hidden");
            }
            if (this.stageImg && dataUrl) {
                this.stageImg.src = dataUrl;
                this.stageImg.classList.remove("hidden");
                this.stageEmpty?.classList.add("hidden");
            }
            this._updateStageBadge(frame);
            return;
        }

        const entry = this.getFrameMapEntry(frame);
        const url = this.getClipViewUrl(entry.clip);
        const v = this.stageVideo;
        if (!v || !url) {
            this.stageEmpty?.classList.remove("hidden");
            return;
        }

        this.stageImg?.classList.add("hidden");
        this.stageEmpty?.classList.add("hidden");
        v.classList.remove("hidden");

        let sameSrc = false;
        if (v.src && url) {
            try {
                sameSrc = new URL(v.src, location.href).href === new URL(url, location.href).href;
            } catch {
                sameSrc = v.src === url;
            }
        }
        // Must reload when the file changes even if clip index stays 0 (replace upload).
        if (this._stageClipIndex !== entry.clip || !sameSrc) {
            this._stageClipIndex = entry.clip;
            if (!sameSrc) {
                v.pause();
                v.src = url;
                v.load();
            }
        }

        const target = Math.max(0, entry.frame / fps);
        if (force || Math.abs(v.currentTime - target) > 0.035) {
            try {
                v.currentTime = target;
            } catch {
                /* ignore seek races while loading */
            }
        }
        if (this.isPlaying && force) {
            v.play().catch(() => {});
        }
        this._updateStageBadge(frame);
    },
    async _ensureStageReadyForFrame(logicalFrame) {
        this._syncStagePreview(logicalFrame, { force: true });
        const v = this.stageVideo;
        if (!v || this._legacyFrames.length) return false;
        if (v.readyState >= 2) return true;
        await new Promise((resolve) => {
            const done = () => {
                v.removeEventListener("loadeddata", done);
                v.removeEventListener("canplay", done);
                resolve();
            };
            v.addEventListener("loadeddata", done);
            v.addEventListener("canplay", done);
            setTimeout(done, 800);
        });
        return true;
    }
};
