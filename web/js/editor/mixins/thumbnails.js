/** thumbnails mixin for the Director editor (thumbnails).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { buildIdentityFrameMap, deletedSourceRanges } from "../../core/frame_map.js";
import { THUMB_JPEG_Q, THUMB_MAX_W, THUMB_PREFETCH_BATCH } from "../../core/layout_spec.js";
import { UPLOAD_SOFT_LIMIT, formatUploadError, uploadToInputSmart } from "../../core/upload.js";
import { relPath, uid } from "../../core/utils.js";
import { inputViewUrl, videoRelativePath } from "../urls.js";
import { openFl2vUpload } from "../../minimax_fl2v.js";
import { taskUsesReferenceVideo } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const thumbnailsMixin = {
    _queueThumbPrefetch(logicalFrame) {
        if (this.isPlaying) return;
        if (!this._usesSourceVideoThumbs()) return;
        const cacheKey = this._frameThumbKey(logicalFrame);
        if (this._thumbCache.has(cacheKey) || this._thumbPending.has(cacheKey)) return;
        if (!this.hasVideo() && !this._legacyFrames.length) return;
        this._thumbPending.add(cacheKey);
        this._fetchThumb(logicalFrame).then((img) => {
            this._thumbPending.delete(cacheKey);
            if (!img) return;
            if (this._frameThumbKey(logicalFrame) !== cacheKey) return;
            this._thumbCache.set(cacheKey, img);
            this.scheduleRender();
        }).catch(() => {
            this._thumbPending.delete(cacheKey);
        });
    },
    /** Capture a still from an r2v reference video for the timeline strip. */
    _queueR2vVideoThumb(cacheKey, videoFile, type = "input") {
        if (!cacheKey || !videoFile) return;
        if (this._thumbCache.has(cacheKey) || this._thumbPending.has(cacheKey)) return;
        this._thumbPending.add(cacheKey);
        const url = inputViewUrl(videoFile, type || "input");
        const v = document.createElement("video");
        v.muted = true;
        v.playsInline = true;
        v.preload = "auto";
        v.crossOrigin = "anonymous";
        let done = false;
        const finish = (img) => {
            if (done) return;
            done = true;
            this._thumbPending.delete(cacheKey);
            try {
                v.removeAttribute("src");
                v.load();
            } catch (_) { /* ignore */ }
            if (img) this._thumbCache.set(cacheKey, img);
            this.scheduleRender();
        };
        const capture = () => {
            try {
                if (!v.videoWidth) {
                    finish(null);
                    return;
                }
                if (!this._thumbCanvas) {
                    this._thumbCanvas = document.createElement("canvas");
                    this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });
                }
                const ratio = v.videoWidth > THUMB_MAX_W ? THUMB_MAX_W / v.videoWidth : 1;
                const tw = Math.max(1, Math.round(v.videoWidth * ratio));
                const th = Math.max(1, Math.round(v.videoHeight * ratio));
                this._thumbCanvas.width = tw;
                this._thumbCanvas.height = th;
                this._thumbCtx.drawImage(v, 0, 0, tw, th);
                const img = new Image();
                img.onload = () => finish(img);
                img.onerror = () => finish(null);
                img.src = this._thumbCanvas.toDataURL("image/jpeg", THUMB_JPEG_Q);
            } catch (_) {
                finish(null);
            }
        };
        v.addEventListener("loadeddata", () => {
            const seekTo = Math.min(0.15, Math.max(0, (v.duration || 1) * 0.05));
            const onSeeked = () => {
                v.removeEventListener("seeked", onSeeked);
                capture();
            };
            v.addEventListener("seeked", onSeeked);
            try {
                v.currentTime = seekTo;
            } catch (_) {
                capture();
            }
            setTimeout(() => {
                if (!done) capture();
            }, 700);
        }, { once: true });
        v.onerror = () => finish(null);
        v.src = url;
    },
    async _fetchThumb(logicalFrame) {
        if (this._legacyFrames.length) {
            const dataUrl = this._legacyFrames[logicalFrame];
            if (!dataUrl) return null;
            return this._decodeThumb(dataUrl);
        }
        const entry = this.getFrameMapEntry(logicalFrame);
        const v = await this._ensurePreviewReady(entry.clip);
        if (!v?.videoWidth) return null;
        const t = Math.max(0, entry.frame / this.getFrameRate());
        await this._seekPreviewVideo(t, entry.clip);
        if (!v.videoWidth) return null;
        try {
            const ratio = v.videoWidth > THUMB_MAX_W ? THUMB_MAX_W / v.videoWidth : 1;
            const tw = Math.max(1, Math.round(v.videoWidth * ratio));
            const th = Math.max(1, Math.round(v.videoHeight * ratio));
            if (!this._thumbCanvas) {
                this._thumbCanvas = document.createElement("canvas");
                this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });
            }
            this._thumbCanvas.width = tw;
            this._thumbCanvas.height = th;
            this._thumbCtx.drawImage(v, 0, 0, tw, th);
            const dataUrl = this._thumbCanvas.toDataURL("image/jpeg", THUMB_JPEG_Q);
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = dataUrl;
            });
        } catch {
            return null;
        }
    },
    _clearVideoState({ dropUnusedThumbs = false } = {}) {
        const oldIds = dropUnusedThumbs ? this._liveVideoFileIdentities() : [];
        this._legacyFrames = [];
        this.timeline.videoClips = [];
        this.timeline.videoWorkspace = null;
        // Wipe video identity BEFORE visibility sync — otherwise hasVideo() stays
        // true via the old videoFile and stage reloads the previous clip.
        this.timeline.video = {
            fileName: "",
            videoFile: "",
            subfolder: "",
            type: "input",
            frames: [],
            frameMap: [],
            deletedSourceRanges: [],
            sourceFrameCount: 0,
            width: 0,
            height: 0,
        };
        this.timeline.totalFrames = 0;
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._clearPreviewVideos(true);
        if (this._previewVideo) {
            this._previewVideo.pause();
            this._previewVideo.removeAttribute("src");
            this._previewVideo.load();
        }
        if (this.stageVideo) {
            this.stageVideo.pause();
            this.stageVideo.removeAttribute("src");
            this.stageVideo.load();
            this.stageVideo.classList.add("hidden");
        }
        this.stageImg?.classList.add("hidden");
        if (this.stageImg) this.stageImg.removeAttribute("src");
        this.stageEmpty?.classList.remove("hidden");
        this.stageBadge?.classList.add("hidden");
        this._stageClipIndex = -1;
        if (dropUnusedThumbs) this._dropThumbsIfUnused(oldIds);
        this.updateStageVisibility();
    },
    _resetTimelineForReplaceUpload() {
        const ids = this._liveVideoFileIdentities();
        if (ids.length) this._thumbIdsPendingDrop = ids;
        this._clearVideoState();
        this.timeline.segments = [];
        this.selectedIndex = 0;
        this.currentFrame = 0;
        if (this.seekBar) {
            this.seekBar.value = 0;
            this.seekBar.max = 0;
        }
    },
    _setSingleSegment(totalFrames) {
        const total = Math.max(0, totalFrames);
        this.timeline.segments = total > 0
            ? [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }]
            : [];
        this.selectedIndex = 0;
        this.currentFrame = 0;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = 0;
        }
    },
    restoreVideoFromTimeline() {
        const video = this.timeline.video || {};
        this._storageWidth = video.storageWidth || 0;
        this._storageHeight = video.storageHeight || 0;

        const legacy = video.frames || [];
        if (legacy.length && !video.videoFile) {
            this._legacyFrames = legacy;
            this.setFrameMap(buildIdentityFrameMap(legacy.length));
            this.videoNameEl.textContent = t("videoName.legacy", {
                name: video.fileName || t("videoName.defaultVideo"),
                frames: legacy.length,
            });
            this._prefetchSegmentThumbs(0, legacy.length);
            this.updateStageVisibility();
            return;
        }

        if (!video.videoFile) {
            this._clearVideoState();
            return;
        }

        this._restorePreviewVideos();
        const n = this.getTotalFrames();
        this._prefetchSegmentThumbs(0, Math.min(n, THUMB_PREFETCH_BATCH * 4));
        this.updateVideoNameLabel();
        if (taskUsesReferenceVideo(this.getTaskKey()) && this.getReferenceVideoViewUrl(this.timeline.global?.referenceVideo)) {
            this.renderRefVideoSlot();
        }
        this.updateStageVisibility();
    },
    _prefetchSegmentThumbs(from, to) {
        if (!this._usesSourceVideoThumbs()) return;
        for (let f = from; f < to; f++) this._queueThumbPrefetch(f);
    },
    _decodeThumb(dataUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                if (!img.naturalWidth || img.naturalWidth <= THUMB_MAX_W) {
                    resolve(img);
                    return;
                }
                const ratio = THUMB_MAX_W / img.naturalWidth;
                const w = THUMB_MAX_W;
                const h = Math.max(1, Math.round(img.naturalHeight * ratio));
                const c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                c.getContext("2d").drawImage(img, 0, 0, w, h);
                const thumb = new Image();
                thumb.onload = () => resolve(thumb);
                thumb.onerror = () => resolve(img);
                thumb.src = c.toDataURL("image/jpeg", THUMB_JPEG_Q);
            };
            img.onerror = () => resolve(null);
            img.src = dataUrl.startsWith("data:") ? dataUrl : `data:image/jpeg;base64,${dataUrl}`;
        });
    },
    pickVideoFile() {
        if (this.isFl2vMode()) {
            openFl2vUpload(this);
            return;
        }
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.loadVideoFile(input.files[0]); };
        input.click();
    },
    async pickExistingVideoFile() {
        if (this.isFl2vMode()) return;
        try {
            const picked = await this.chooseVideoInput({
                title: t("mediaPicker.pickVideo"),
                currentValue: this.timeline.video?.videoFile || "",
            });
            if (!picked?.relPath) return;
            const btn = this.root.querySelector('[data-a="video-existing"]');
            if (btn) { btn.disabled = true; btn.textContent = t("common.analyzing"); }
            this.videoNameEl.textContent = t("upload.inProgress", { name: picked.fileName || picked.relPath });
            try {
                await this._applyLoadedVideo({
                    fileName: picked.fileName || picked.relPath,
                    relPath: picked.relPath,
                    subfolder: picked.subfolder || "",
                    type: picked.type || "input",
                    statusPrefix: t("parse.prefix"),
                });
            } catch (err) {
                console.error("[MiniMax H3Director] video load failed:", err);
                this.videoNameEl.textContent = t("upload.loadFailed", { err: formatUploadError(err) });
                this.updateVideoNameLabel();
                this._flushPendingThumbDrops();
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = t("mediaPicker.pickExistingVideo");
                }
            }
        } catch (err) {
            console.error("[MiniMax H3Director] video pick failed:", err);
        }
    },
    pickAppendVideoFile() {
        if (!this.hasVideo()) {
            this.showBdMessage(
                t("dialog.appendVideoTitle"),
                t("dialog.appendVideoNeedFirst")
            );
            return;
        }
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.appendVideoFile(input.files[0]); };
        input.click();
    },
    async appendVideoFile(file) {
        const btn = this.root.querySelector('[data-a="video-append"]');
        if (btn) { btn.disabled = true; btn.textContent = t("common.uploading"); }
        this.videoNameEl.textContent = t("upload.appendProgress", { name: file.name });
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > UPLOAD_SOFT_LIMIT ? t("upload.chunkMode") : t("upload.mode");
                this.videoNameEl.textContent = t("upload.appendChunk", {
                    mode, name: file.name, cur, total, pct,
                });
            });
            const relPath = videoRelativePath(uploaded);
            await this._applyAppendedVideo({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: t("parse.prefix"),
            });
        } catch (err) {
            console.error("[MiniMax H3Director] append video failed:", err);
            this.videoNameEl.textContent = t("upload.appendFailed", { err: formatUploadError(err) });
            this.updateVideoNameLabel();
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = t("toolbar.appendVideo");
            }
        }
    },
    async loadVideoFile(file) {
        const btn = this.root.querySelector('[data-a="video"]');
        if (btn) { btn.disabled = true; btn.textContent = t("common.uploading"); }
        this.videoNameEl.textContent = t("upload.inProgress", { name: file.name });
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > UPLOAD_SOFT_LIMIT ? t("upload.chunkMode") : t("upload.mode");
                this.videoNameEl.textContent = t("upload.loadChunk", {
                    mode, name: file.name, cur, total, pct,
                });
            });
            const relPath = videoRelativePath(uploaded);
            await this._applyLoadedVideo({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: t("parse.prefix"),
            });
        } catch (err) {
            console.error("[MiniMax H3Director] video load failed:", err);
            this.videoNameEl.textContent = t("upload.loadFailed", { err: formatUploadError(err) });
            this.updateVideoNameLabel();
            this._flushPendingThumbDrops();
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = t("toolbar.uploadVideo");
            }
        }
    }
};
