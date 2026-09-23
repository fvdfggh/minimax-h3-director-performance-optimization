


import { api } from "../../../scripts/api.js";
import { coerceTimelineFps } from "../core/dims.js";
import { healOversizedDirectorNode, hideWidget, parseTimeline, syncDirectorNodeSize } from "../core/editor_lifecycle.js";
import { buildIdentityFrameMap, deletedSourceRanges, normalizeFrameMapEntry, sourceToLogicalFrame } from "../core/frame_map.js";

import { HIDDEN_WIDGETS, MIN_SEG, RULER_H, SEG_LABEL_H, THUMB_JPEG_Q, THUMB_MAX_W, THUMB_PREFETCH_BATCH, TRACK_H } from "../core/layout_spec.js";


import { UPLOAD_SOFT_LIMIT, formatUploadError, uploadToInput, uploadToInputSmart } from "../core/upload.js";
import { clamp, relPath, uid } from "../core/utils.js";


import { inputViewUrl, refViewUrl, videoRelativePath } from "./urls.js";
import { openFl2vUpload, removeFl2vShot, updateFl2vDetailUI, updateFl2vToolbarBtns } from "../minimax_fl2v.js";
import { MAX_REFERENCE_AUDIOS, MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, defaultFrameCount, getDirectorMode, isVideoBatchTask, refAudioLabel, refImageLabel, refVideoLabel, resolveTaskKey, taskUsesReferenceAudios, taskUsesReferenceImages, taskUsesReferenceVideo } from "../minimax_gen_timeline.js";
import { onLocaleChange, t } from "../minimax_i18n.js";
import { bindDomWidgetContentComputeSize, bindR2vMediaPlayback, deleteImageBatchGroup, ensureImageBatchTimeline, formatMediaDuration, isBatchDetailSolo, rebaseR2vGroupSlotsForCommon, syncBatchPanelFillHeight, updateR2vToolbarBtns, wireMediaDuration } from "../minimax_image_batch.js";

import { refreshPromptTokenEditors } from "../minimax_prompt_mentions.js";
import { extractReferenceAudioFromExistingVideo, hasDuplicateReferenceAudio, prepareLocalReferenceAudio } from "../minimax_ref_audio.js";
import { canvasMixin } from "./mixins/canvas.js";
import { interactionMixin } from "./mixins/interaction.js";
import { dom_shellMixin } from "./mixins/dom_shell.js";
import { eventsMixin } from "./mixins/events.js";
import { timeline_payloadMixin } from "./mixins/timeline_payload.js";
import { run_selectionMixin } from "./mixins/run_selection.js";
import { export_pickersMixin } from "./mixins/export_pickers.js";
import { external_groupsMixin } from "./mixins/external_groups.js";
import { layout_scheduleMixin } from "./mixins/layout_schedule.js";
import { fieldsMixin } from "./mixins/fields.js";
import { modesMixin } from "./mixins/modes.js";
import { workspacesMixin } from "./mixins/workspaces.js";
import { task_layoutMixin } from "./mixins/task_layout.js";
import { ref_video_slotMixin } from "./mixins/ref_video_slot.js";
import { media_thumbsMixin } from "./mixins/media_thumbs.js";
import { video_loadMixin } from "./mixins/video_load.js";
import { segments_modelMixin } from "./mixins/segments_model.js";
import { frame_map_ioMixin } from "./mixins/frame_map_io.js";
import { task_uiMixin } from "./mixins/task_ui.js";
import { output_uiMixin } from "./mixins/output_ui.js";
import { widget_syncMixin } from "./mixins/widget_sync.js";

export class MiniMaxH3DirectorOptEditor {
    constructor(node, container, domWidget) {
        this.node = node;
        this.container = container;
        this.domWidget = domWidget;
        this.zoom = 1;
        this.zoomEnabled = false;
        this.selectedIndex = 0;
        /** @type {number|null} Selected editable split-point frame (logical). */
        this.selectedSplitFrame = null;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.isLooping = false;
        this._playRaf = null;
        this._drag = null;
        this._previewSegments = null;
        this._edgeSnapshot = null;
        this._isHovering = false;
        this._thumbCache = new Map();
        this._thumbPending = new Set();
        this._seekChain = Promise.resolve();
        this._legacyFrames = [];
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._previewVideo = null;
        this._previewVideos = new Map();
        this._thumbCanvas = null;
        this._syncTimer = null;
        this._resizeRaf = null;
        this._renderPending = false;
        this._settleRenderTimer = null;
        this._settleRenderLateTimer = null;
        this._promptRenderTimer = null;
        this._lastSeekUiMs = 0;
        this._playCanvasWidth = 0;
        this._pauseSettling = false;
        this._runHighlightSeg = -1;
        this._modalEl = null;
        this._modalKeyHandler = null;
        this._drawWidth = 0;
        this._reorderDropRank = -1;
        this._reorderFromRank = -1;
        this.canvasHeight = RULER_H + SEG_LABEL_H + TRACK_H;
        this._stageClipIndex = -1;
        this._stageSyncMs = 0;
        this._playHandoff = false;

        for (const w of node.widgets || []) {
            if (HIDDEN_WIDGETS.includes(w.name)) hideWidget(w);
        }

        this.timelineWidget = this.widget("timeline_data");
        this.totalFramesWidget = this.widget("total_frames");
        this.frameRateWidget = this.widget("frame_rate");
        this.taskTypeWidget = this.widget("task_type");
        this.globalPromptWidget = this.widget("global_prompt");
        this.negativePromptWidget = null;
        this.widthWidget = this.widget("width");
        this.heightWidget = this.widget("height");
        this.refMaxWidget = this.widget("ref_max_size");

        const initTotal = Math.max(0, parseInt(this.totalFramesWidget?.value || 124, 10));
        const initFps = coerceTimelineFps(this.frameRateWidget?.value || 24);
        this.timeline = parseTimeline(this.timelineWidget?.value, initTotal, initFps);
        this.buildDOM();
        this.bindEvents();
        this._unsubLocale = onLocaleChange(() => this.applyLocale());
        this.applyLocale();
        this._directorMode = getDirectorMode(this.taskTypeWidget?.value);
        this._taskKey = resolveTaskKey(this.taskTypeWidget?.value);
        if (this._directorMode === "video") {
            this.restoreVideoFromTimeline();
        } else if (this._directorMode === "prompt_batch" || this._directorMode === "image_batch") {
            ensureImageBatchTimeline(this);
        } else {
            this.ensureGenTimeline();
        }
        this.applyTaskLayout(this._directorMode);

        this.updateDomWidgetHeight();
        this.applyZoomWidth();
        this.syncFromWidgets();
        this.updateModeUI();
        this.updateSelectionUI();
        this.commit(true, { syncTimeline: false });
        this._observeViewportResize();
        this.syncExternalGroupsTimeline();
        this.scheduleSettleRender();
    }





    /**
     * Push a Director-card prompt edit into the matching external Group node
     * widget so execution (and the next sync) don't revive stale graph text.
     */



    /**
     * CSS layout width for the timeline bitmap.
     * Must NOT use getBoundingClientRect — ComfyUI graph zoom transforms inflate/deflate
     * that value while width:100% still follows clientWidth, and object-fit:fill then
     * stretches segment thumbnails.
     */

















































    /**
     * Re-base「选择运行」after the group at ``removedIndex`` is removed.
     *
     * ``runSelection`` is a plain list of array positions, so removing a group in
     * the middle used to hand its tick to the next group and silently drop the
     * last one. Everything above the removed position shifts down by one; the
     * removed position itself is dropped.
     */

    /**
     * Re-base「选择运行」after a group moves from ``fromRank`` to ``toRank``
     * (splice-out then splice-in). Without this the ticks stay on the old
     * positions and end up selecting whichever group landed there.
     */

    /**
     * Ask the backend to drop the cache files of the group at ``index``.
     *
     * Fire-and-forget: the cache is content-addressed now, so a failed call only
     * leaves orphaned files behind — the next run's slot sync removes them.
     */



    // ---------------------------------------------------------------------
    // 对齐下段 (align-to-next) availability
    // ---------------------------------------------------------------------


    /**
     * Refresh which segments may enable「对齐下段」.
     *
     * Asked of the backend because under「选择运行」the plan index is the
     * compact run order, so the frontend cannot infer "the next segment" from
     * the card list. Best-effort: on failure every segment stays disabled.
     */

    /**
     * Whether segment ``index`` may tick「对齐下段」: the next segment must hold
     * a cached AV latent. Unticked by default — this is an opt-in, cache-driven
     * middle-out mode.
     */






    // ---------------------------------------------------------------------
    // 分段导出 (segment export)
    // ---------------------------------------------------------------------






    // ---------------------------------------------------------------------
    // 二次采样 (second sample) — 复用分段导出弹窗结构，但写入 secondSample，
    // 执行时走节点 execute 的二采分支（需已加载 model / upscale_model / VAE）。
    // ---------------------------------------------------------------------













    /**
     * Turn the persisted「分段导出」flag back off.
     *
     * Export is a one-shot action, but ``resolveSegmentExport`` stores
     * ``enabled`` on ``timeline.output`` where it survives re-renders and page
     * reloads. Left set, every later run exports instead of generating — the run
     * silently becomes an export-only pass (no sampling, just a few seconds).
     * Mode/indices are kept so the picker reopens with the last selection.
     */












    /**
     * 方案B: 让「分段导出 / 二次采样」相关 UI 与「选择运行」状态彻底解耦。
     * 卡片进入 run-skipped(未选运行)时, 不应把卡片内/工具栏里的导出·二采元素
     * 一并灰化或禁用 —— 它们的可用与否只由各自功能的缓存/启用状态决定。
     * 这里作为 CSS 豁免之外的双保险(强制 opacity/disabled 不受 run-skipped 影响)。
     */







































































    /**
     * Shared params are always on in r2v — there is no「启用公共参数」toggle any
     * more: the shared asset page is a permanent page of the paginator, so the
     * runtime merge (concat prompt + merge refs) must always see it.
     */


    /**
     * Stable workflow id used by every cache path (``<root>/<slug>/node_<id>``).
     * Read-only routes（分段导出状态 / 二采状态 / segment_clip）must use this —
     * the hidden ``workflow_name`` widget is only synced at queue time, so
     * reading it here can still yield "" and point at the bare ``node_<id>`` dir.
     */



























    getVideoViewUrl() {
        return this.getClipViewUrl(0);
    }

    getSourceFrameIndex(logicalFrame) {
        return this.getFrameMapEntry(logicalFrame).frame;
    }

    _previewUrlEquals(videoEl, url) {
        if (!videoEl || !url) return false;
        const src = videoEl.currentSrc || videoEl.getAttribute("src") || videoEl.src || "";
        if (!src) return false;
        try {
            return new URL(src, location.href).href === new URL(url, location.href).href;
        } catch {
            return src === url;
        }
    }

    _assignPreviewSrc(videoEl, url) {
        if (!videoEl || !url) return;
        if (this._previewUrlEquals(videoEl, url)) return;
        videoEl.pause();
        videoEl.src = url;
        videoEl.load();
    }

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
    }

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
    }

    _restorePreviewVideos() {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        for (let i = 0; i < clips.length; i++) this._getPreviewVideoForClip(i);
        this._previewVideo = this._previewVideos.get(0) || this._previewVideo;
    }

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
    }

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
    }

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
    }

    _updateStageBadge(logicalFrame) {
        if (!this.stageBadge) return;
        const total = this.getTotalFrames();
        const frame = clamp(logicalFrame | 0, 0, Math.max(0, total - 1));
        const clips = this.getVideoClips();
        const entry = this.getFrameMapEntry(frame);
        const clipHint = clips.length > 1 ? t("canvas.clipHint", { n: entry.clip + 1 }) : "";
        this.stageBadge.textContent = t("player.frameOf", { cur: frame + 1, total, clip: clipHint });
        this.stageBadge.classList.remove("hidden");
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

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
    }

    _prefetchSegmentThumbs(from, to) {
        if (!this._usesSourceVideoThumbs()) return;
        for (let f = from; f < to; f++) this._queueThumbPrefetch(f);
    }

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
    }

    pickVideoFile() {
        if (this.isFl2vMode()) {
            openFl2vUpload(this);
            return;
        }
        const input = document.createElement("input");
        input.type = "file"; input.accept = "video/*";
        input.onchange = () => { if (input.files?.[0]) this.loadVideoFile(input.files[0]); };
        input.click();
    }

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
    }

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
    }

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
    }

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

    _closeBdModal() {
        if (this._modalKeyHandler) {
            window.removeEventListener("keydown", this._modalKeyHandler, true);
            this._modalKeyHandler = null;
        }
        if (this._modalEl) {
            this._modalEl.remove();
            this._modalEl = null;
        }
    }

    showBdMessage(title, message) {
        return this.showBdDialog({ title, message, confirmText: t("dialog.confirm"), cancelText: null });
    }

    showBdDialog(opts = {}) {
        const { title, message, items } = opts;
        const confirmText = opts.confirmText ?? t("dialog.confirm");
        const cancelText = Object.prototype.hasOwnProperty.call(opts, "cancelText")
            ? opts.cancelText
            : t("dialog.cancel");
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal";
            panel.innerHTML = `
                <div class="bd-modal-title"></div>
                <div class="bd-modal-body hidden"></div>
                <div class="bd-modal-list hidden"></div>
                <div class="bd-modal-actions"></div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";

            const bodyEl = panel.querySelector(".bd-modal-body");
            const listEl = panel.querySelector(".bd-modal-list");
            const actionsEl = panel.querySelector(".bd-modal-actions");

            let selectedValue = items?.length ? items[0].value : null;

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            if (message) {
                bodyEl.textContent = message;
                bodyEl.classList.remove("hidden");
            }

            if (items?.length) {
                listEl.classList.remove("hidden");
                for (const item of items) {
                    const row = document.createElement("div");
                    row.className = "bd-modal-item";
                    row.textContent = item.label ?? item.value;
                    row.title = item.label ?? item.value;
                    row.dataset.value = item.value;
                    if (item.value === selectedValue) row.classList.add("selected");
                    row.onclick = () => {
                        selectedValue = item.value;
                        for (const el of listEl.querySelectorAll(".bd-modal-item")) {
                            el.classList.toggle("selected", el === row);
                        }
                    };
                    row.ondblclick = () => finish(item.value);
                    listEl.appendChild(row);
                }
            }

            if (cancelText) {
                const cancelBtn = document.createElement("button");
                cancelBtn.type = "button";
                cancelBtn.className = "bd-btn";
                cancelBtn.textContent = cancelText;
                cancelBtn.onclick = () => finish(null);
                actionsEl.appendChild(cancelBtn);
            }

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = confirmText;
            okBtn.onclick = () => finish(items?.length ? selectedValue : true);
            actionsEl.appendChild(okBtn);

            overlay.onclick = (e) => {
                if (e.target === overlay && cancelText) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(cancelText ? null : true);
                } else if (e.key === "Enter" && items?.length) {
                    e.preventDefault();
                    finish(selectedValue);
                }
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            okBtn.focus();
        });
    }

    pickLocalFile(accept = "") {
        return new Promise((resolve) => {
            const input = document.createElement("input");
            input.type = "file";
            if (accept) input.accept = accept;
            input.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;pointer-events:none";
            const cleanup = () => input.remove();
            input.onchange = () => {
                const file = input.files?.[0] || null;
                cleanup();
                resolve(file);
            };
            input.addEventListener("cancel", () => {
                cleanup();
                resolve(null);
            }, { once: true });
            document.body.appendChild(input);
            input.click();
        });
    }

    async listInputMedia(kind, { includeCache = false } = {}) {
        // ``includeCache`` pulls in Director's own rendered clips from output/.
        // Opt-in per call so image/audio pickers keep their current contents —
        // this is only ever useful for video, and only when the user is hunting
        // for something they generated earlier.
        const params = new URLSearchParams({ kind });
        if (includeCache) params.set("includeCache", "1");
        const resp = await api.fetchApi(`/minimax/director_opt/list_input_media?${params.toString()}`);
        if (!resp.ok) {
            const text = (await resp.text()).trim();
            if (resp.status === 404) throw new Error(t("mediaPicker.needRestart"));
            throw new Error(text || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        return Array.isArray(data?.items) ? data.items : [];
    }

    probeInputImageDimensions(relPath, type = "input") {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve({
                width: img.naturalWidth || img.width || 0,
                height: img.naturalHeight || img.height || 0,
            });
            img.onerror = () => resolve({ width: 0, height: 0 });
            img.src = inputViewUrl(relPath, type || "input");
        });
    }

    showInputMediaPicker({ kind, title, accept, currentValue = "", multi = false, includeCache = false } = {}) {
        return new Promise((resolve) => {
            this._closeBdModal();

            const overlay = document.createElement("div");
            overlay.className = "bd-modal-overlay";
            const panel = document.createElement("div");
            panel.className = "bd-modal bd-media-modal";
            panel.innerHTML = `
                <div class="bd-media-head">
                    <div class="bd-modal-title"></div>
                    <div class="bd-modal-actions bd-media-head-actions"></div>
                </div>
                <div class="bd-media-status"></div>
                <div class="bd-media-body">
                    <div class="bd-media-left">
                        <div class="bd-media-table" tabindex="0">
                            <div class="bd-media-thead">
                                <button type="button" class="bd-media-th" data-sort="name">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                                <button type="button" class="bd-media-th" data-sort="dims">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                                <button type="button" class="bd-media-th" data-sort="time">
                                    <span></span><i class="bd-media-sort"></i>
                                </button>
                            </div>
                            <div class="bd-media-tbody"></div>
                        </div>
                    </div>
                    <div class="bd-media-right">
                        <div class="bd-media-preview">
                            <div class="bd-media-preview-empty"></div>
                        </div>
                        <div class="bd-media-meta"></div>
                    </div>
                </div>`;

            panel.querySelector(".bd-modal-title").textContent = title || "";
            const statusEl = panel.querySelector(".bd-media-status");
            const actionsTop = panel.querySelector(".bd-media-head-actions");
            const tableEl = panel.querySelector(".bd-media-table");
            const tbodyEl = panel.querySelector(".bd-media-tbody");
            const previewEl = panel.querySelector(".bd-media-preview");
            const previewEmptyEl = panel.querySelector(".bd-media-preview-empty");
            const metaEl = panel.querySelector(".bd-media-meta");
            const showDims = kind !== "audio" && kind !== "reference_audio";
            if (!showDims) {
                tableEl.classList.add("bd-media-nodims");
                tableEl.querySelector('.bd-media-th[data-sort="dims"]')?.remove();
            }
            if (multi) {
                // 批量模式：表格多一列复选框，给 thead 补一个占位 cell 保持列对齐，
                // 否则复选框会把 name 列挤掉，time 列还会换到下一行。
                tableEl.classList.add("bd-media-multi");
                // 注意：不要用 .bd-media-th —— 它会被列头文案/排序逻辑选中，
                // 而这里没有可填充的 span，会抛 "Cannot set properties of null"。
                const cbHead = document.createElement("span");
                cbHead.className = "bd-media-th-cb";
                cbHead.appendChild(document.createElement("span"));
                cbHead.setAttribute("aria-hidden", "true");
                const theadEl = tableEl.querySelector(".bd-media-thead");
                theadEl?.insertBefore(
                    cbHead,
                    theadEl?.querySelector(".bd-media-th[data-sort='name']"),
                );
            }
            const thEls = [...panel.querySelectorAll(".bd-media-th")];
            thEls.forEach((th) => {
                const key = th.dataset.sort;
                const label = key === "dims" ? t("mediaPicker.dims")
                    : key === "time" ? t("mediaPicker.time")
                    : t("mediaPicker.file");
                // 缺 span 只跳过自己，不要让整个弹窗初始化失败。
                const labelEl = th.querySelector("span");
                if (labelEl) labelEl.textContent = label;
            });

            let selectedValue = currentValue || "";
            let itemsByPath = new Map();
            let listedItems = [];
            let sortKey = "time";
            let sortDir = "desc";
            // 批量选择：选中集合（multi=true 时生效，返回数组）
            const multiSel = new Set();

            const finish = (val) => {
                this._closeBdModal();
                resolve(val);
            };

            const choiceFor = (relPath) => {
                const item = itemsByPath.get(relPath || "");
                if (!item) return null;
                return {
                    source: "existing",
                    relPath: item.relPath,
                    fileName: item.fileName || item.name || item.relPath,
                    subfolder: item.subfolder || "",
                    type: item.type || "input",
                    mediaKind: item.mediaKind || kind,
                };
            };

            const selectedChoice = () => choiceFor(selectedValue);

            const selectedChoices = () => {
                // 保持列表顺序，用户勾选顺序无关紧要
                return sortedItems()
                    .map((it) => it.relPath)
                    .filter((p) => multiSel.has(p))
                    .map((p) => choiceFor(p))
                    .filter(Boolean);
            };

            const formatMediaTime = (unixSec) => {
                const n = Number(unixSec);
                if (!Number.isFinite(n) || n <= 0) return "—";
                const d = new Date(n * 1000);
                if (Number.isNaN(d.getTime())) return "—";
                const pad = (v) => String(v).padStart(2, "0");
                return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
            };

            const dimsText = (item) => {
                const w = Number(item?.width) || 0;
                const h = Number(item?.height) || 0;
                return w > 0 && h > 0 ? `${w}x${h}` : "—";
            };

            const dimScore = (item) => {
                const w = Number(item?.width) || 0;
                const h = Number(item?.height) || 0;
                return w > 0 && h > 0 ? w * 100000 + h : -1;
            };

            const sortedItems = () => {
                const copy = listedItems.slice();
                copy.sort((a, b) => {
                    if (sortKey === "name") {
                        const av = (a.fileName || a.name || a.relPath || "").toLowerCase();
                        const bv = (b.fileName || b.name || b.relPath || "").toLowerCase();
                        const c = av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" });
                        if (c) return sortDir === "asc" ? c : -c;
                    } else if (sortKey === "dims") {
                        const as = dimScore(a);
                        const bs = dimScore(b);
                        const aMiss = as < 0;
                        const bMiss = bs < 0;
                        if (aMiss !== bMiss) return aMiss ? 1 : -1;
                        if (as !== bs) return sortDir === "asc" ? as - bs : bs - as;
                    } else {
                        const av = Number(a.modified) || 0;
                        const bv = Number(b.modified) || 0;
                        if (av !== bv) return sortDir === "asc" ? av - bv : bv - av;
                    }
                    return (a.relPath || "").localeCompare(b.relPath || "");
                });
                return copy;
            };

            const syncHeaderState = () => {
                thEls.forEach((th) => {
                    const active = th.dataset.sort === sortKey;
                    th.classList.toggle("is-active", active);
                    th.classList.toggle("is-asc", active && sortDir === "asc");
                });
            };

            const renderPreview = (item) => {
                previewEl.innerHTML = "";
                metaEl.innerHTML = "";
                if (!item?.relPath) {
                    previewEmptyEl.textContent = t("mediaPicker.previewEmpty");
                    previewEl.appendChild(previewEmptyEl);
                    return;
                }
                const relPath = item.relPath;
                const type = item.type || "input";
                const previewKind = kind === "reference_audio" ? item.mediaKind : kind;
                if (previewKind === "image") {
                    const img = document.createElement("img");
                    img.src = inputViewUrl(relPath, type);
                    img.alt = item.fileName || item.name || relPath;
                    previewEl.appendChild(img);
                } else if (previewKind === "audio") {
                    const audio = document.createElement("audio");
                    audio.src = inputViewUrl(relPath, type);
                    audio.controls = true;
                    audio.preload = "metadata";
                    previewEl.appendChild(audio);
                } else {
                    const video = document.createElement("video");
                    video.src = inputViewUrl(relPath, type);
                    video.controls = true;
                    video.preload = "metadata";
                    video.muted = true;
                    video.playsInline = true;
                    previewEl.appendChild(video);
                }
                const fileEl = document.createElement("div");
                fileEl.textContent = `${t("mediaPicker.file")}: ${item.fileName || item.name || relPath}`;
                metaEl.appendChild(fileEl);
                const pathEl = document.createElement("div");
                pathEl.textContent = `${t("mediaPicker.path")}: ${relPath}`;
                metaEl.appendChild(pathEl);
            };

            const selectRow = (relPath, { scroll = false, toggleCheck = false } = {}) => {
                if (multi) {
                    if (!relPath) return;
                    // 行点击 = 预览；只有点复选框才切换勾选。
                    if (toggleCheck) {
                        if (multiSel.has(relPath)) multiSel.delete(relPath);
                        else multiSel.add(relPath);
                    }
                    selectedValue = relPath;
                    tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                        const on = row.dataset.path === selectedValue;
                        row.classList.toggle("selected", on);
                        if (on && scroll) row.scrollIntoView({ block: "nearest" });
                    });
                    renderPreview(itemsByPath.get(relPath));
                    syncMultiRows();
                    return;
                }
                selectedValue = relPath || "";
                tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                    const on = row.dataset.path === selectedValue;
                    row.classList.toggle("selected", on);
                    if (on && scroll) row.scrollIntoView({ block: "nearest" });
                });
                renderPreview(itemsByPath.get(selectedValue));
            };

            const syncMultiRows = () => {
                tbodyEl.querySelectorAll(".bd-media-tr").forEach((row) => {
                    const on = multiSel.has(row.dataset.path);
                    // checked = 已勾选（左侧蓝条）；selected = 当前预览项（底色）。
                    row.classList.toggle("checked", on);
                    row.classList.toggle("selected", row.dataset.path === selectedValue);
                    const cb = row.querySelector(".bd-media-cb");
                    if (cb) cb.checked = on;
                });
                const n = multiSel.size;
                statusEl.textContent = n
                    ? t("mediaPicker.multiSelected", { n })
                    : t("mediaPicker.count", { n: listedItems.length });
                okBtn.disabled = n === 0;
            };

            const renderRows = () => {
                tbodyEl.innerHTML = "";
                const rows = sortedItems();
                if (!rows.length) {
                    const empty = document.createElement("div");
                    empty.className = "bd-media-empty-row";
                    empty.textContent = t("mediaPicker.empty");
                    tbodyEl.appendChild(empty);
                    selectedValue = "";
                    renderPreview(null);
                    syncHeaderState();
                    return;
                }
                if (!multi && (!selectedValue || !itemsByPath.has(selectedValue))) {
                    selectedValue = rows[0].relPath || "";
                }
                for (const item of rows) {
                    const row = document.createElement("div");
                    row.className = "bd-media-tr";
                    row.dataset.path = item.relPath;
                    if (multi) {
                        if (multiSel.has(item.relPath)) row.classList.add("checked");
                        if (item.relPath === selectedValue) row.classList.add("selected");
                    } else if (item.relPath === selectedValue) {
                        row.classList.add("selected");
                    }
                    if (multi) {
                        const cbTd = document.createElement("div");
                        cbTd.className = "bd-media-td bd-media-td-cb";
                        const cb = document.createElement("input");
                        cb.type = "checkbox";
                        cb.className = "bd-media-cb";
                        cb.checked = multiSel.has(item.relPath);
                        cb.onclick = (e) => e.stopPropagation();
                        cb.onchange = () => selectRow(item.relPath, { toggleCheck: true });
                        cbTd.appendChild(cb);
                        row.appendChild(cbTd);
                    }
                    const nameTd = document.createElement("div");
                    nameTd.className = "bd-media-td bd-media-td-name";
                    const mediaPrefix = kind === "reference_audio"
                        ? (item.mediaKind === "video" ? "🎞 " : "♪ ")
                        : "";
                    nameTd.textContent = `${mediaPrefix}${item.relPath || item.fileName || item.name || ""}`;
                    nameTd.title = nameTd.textContent;
                    const timeTd = document.createElement("div");
                    timeTd.className = "bd-media-td bd-media-td-time";
                    timeTd.textContent = formatMediaTime(item.modified);
                    if (showDims) {
                        const dimsTd = document.createElement("div");
                        dimsTd.className = "bd-media-td bd-media-td-dims";
                        dimsTd.textContent = dimsText(item);
                        row.append(nameTd, dimsTd, timeTd);
                    } else {
                        row.append(nameTd, timeTd);
                    }
                    row.addEventListener("click", () => selectRow(item.relPath));
                    row.addEventListener("dblclick", () => {
                        if (multi) return; // 批量模式下双击不提交，避免误关
                        const choice = selectedChoice();
                        if (choice) finish(choice);
                    });
                    tbodyEl.appendChild(row);
                }
                syncHeaderState();
                if (multi) {
                    // 默认预览第一项，避免刚打开弹窗时右侧预览区一片空白。
                    if (!selectedValue && rows.length) {
                        selectedValue = rows[0].relPath || "";
                        renderPreview(itemsByPath.get(selectedValue));
                    }
                    syncMultiRows();
                } else {
                    renderPreview(itemsByPath.get(selectedValue));
                    const selectedRow = tbodyEl.querySelector(".bd-media-tr.selected");
                    selectedRow?.scrollIntoView({ block: "nearest" });
                }
            };

            const moveSelection = (delta) => {
                if (multi) return; // 批量模式下方向键不应改动勾选集合
                const rows = sortedItems();
                if (!rows.length) return;
                const idx = rows.findIndex((item) => item.relPath === selectedValue);
                const next = rows[Math.max(0, Math.min(rows.length - 1, (idx < 0 ? 0 : idx) + delta))];
                if (next) selectRow(next.relPath, { scroll: true });
            };

            const loadItems = async () => {
                statusEl.textContent = t("mediaPicker.loading");
                tbodyEl.innerHTML = "";
                renderPreview(null);
                try {
                    listedItems = await this.listInputMedia(kind, { includeCache });
                    itemsByPath = new Map(listedItems.map((item) => [item.relPath, item]));
                    renderRows();
                    if (multi) {
                        syncMultiRows();
                    } else {
                        statusEl.textContent = listedItems.length
                            ? t("mediaPicker.count", { n: listedItems.length })
                            : t("mediaPicker.empty");
                    }
                } catch (err) {
                    listedItems = [];
                    itemsByPath = new Map();
                    tbodyEl.innerHTML = "";
                    statusEl.textContent = err?.message || String(err);
                }
            };

            thEls.forEach((th) => {
                th.addEventListener("click", () => {
                    const key = th.dataset.sort || "time";
                    if (sortKey === key) {
                        sortDir = sortDir === "asc" ? "desc" : "asc";
                    } else {
                        sortKey = key;
                        sortDir = key === "name" ? "asc" : "desc";
                    }
                    renderRows();
                });
            });

            const refreshBtn = document.createElement("button");
            refreshBtn.type = "button";
            refreshBtn.className = "bd-btn";
            refreshBtn.textContent = t("mediaPicker.refresh");
            refreshBtn.onclick = () => { void loadItems(); };
            actionsTop.appendChild(refreshBtn);

            const uploadBtn = document.createElement("button");
            uploadBtn.type = "button";
            uploadBtn.className = "bd-btn";
            uploadBtn.textContent = t("mediaPicker.upload");
            uploadBtn.onclick = async () => {
                const file = await this.pickLocalFile(accept || "");
                if (!file) return;
                finish(multi ? [{ source: "file", file }] : { source: "file", file });
            };
            actionsTop.appendChild(uploadBtn);

            if (multi) {
                const selAll = document.createElement("button");
                selAll.type = "button";
                selAll.className = "bd-btn";
                selAll.textContent = t("mediaPicker.selectAll");
                selAll.onclick = () => {
                    for (const it of listedItems) multiSel.add(it.relPath);
                    syncMultiRows();
                    renderRows();
                };
                actionsTop.appendChild(selAll);
                const clearSel = document.createElement("button");
                clearSel.type = "button";
                clearSel.className = "bd-btn";
                clearSel.textContent = t("mediaPicker.clearSelection");
                clearSel.onclick = () => {
                    multiSel.clear();
                    syncMultiRows();
                    renderRows();
                };
                actionsTop.appendChild(clearSel);
            }

            const actionsBottom = document.createElement("div");
            actionsBottom.className = "bd-modal-actions";
            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "bd-btn";
            cancelBtn.textContent = t("dialog.cancel");
            cancelBtn.onclick = () => finish(null);
            actionsBottom.appendChild(cancelBtn);

            const okBtn = document.createElement("button");
            okBtn.type = "button";
            okBtn.className = "bd-btn bd-btn-primary";
            okBtn.textContent = multi
                ? t("mediaPicker.useSelectedMulti")
                : t("mediaPicker.useSelected");
            okBtn.onclick = () => {
                if (multi) {
                    const choices = selectedChoices();
                    if (choices.length) finish(choices);
                    return;
                }
                const choice = selectedChoice();
                if (choice) finish(choice);
            };
            actionsBottom.appendChild(okBtn);
            panel.appendChild(actionsBottom);

            overlay.onclick = (e) => {
                if (e.target === overlay) finish(null);
            };
            panel.onclick = (e) => e.stopPropagation();

            this._modalKeyHandler = (e) => {
                if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    finish(null);
                    return;
                }
                if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                    if (!panel.contains(e.target) && e.target !== document.body) return;
                    e.preventDefault();
                    moveSelection(e.key === "ArrowDown" ? 1 : -1);
                    return;
                }
                if (e.key !== "Enter") return;
                if (e.target?.closest?.(".bd-media-th, .bd-media-head-actions, button.bd-btn:not(.bd-btn-primary)")) return;
                e.preventDefault();
                okBtn.click();
            };
            window.addEventListener("keydown", this._modalKeyHandler, true);

            overlay.appendChild(panel);
            this.root.appendChild(overlay);
            this._modalEl = overlay;
            void loadItems();
            tableEl.focus();
        });
    }

    async chooseImageInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "image",
            title: opts.title || t("mediaPicker.pickImage"),
            accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff",
            currentValue: opts.currentValue || "",
        });
        return this._resolveImageChoice(choice);
    }

    /** 批量版：返回数组（元素结构同 chooseImageInput）。 */
    async chooseImageInputs(opts = {}) {
        const choices = await this.showInputMediaPicker({
            kind: "image",
            title: opts.title || t("mediaPicker.pickReferenceImage"),
            accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp,.gif,.tif,.tiff",
            currentValue: opts.currentValue || "",
            multi: true,
        });
        if (!Array.isArray(choices) || !choices.length) return [];
        const out = [];
        for (const c of choices) {
            const r = await this._resolveImageChoice(c);
            if (r) out.push(r);
        }
        return out;
    }

    async _resolveImageChoice(choice) {
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            const uploaded = await uploadToInput(choice.file);
            const relPath = videoRelativePath(uploaded);
            const dims = await this.probeInputImageDimensions(relPath, uploaded.type || "input");
            return {
                imageFile: relPath,
                fileName: uploaded?.name || choice.file.name || relPath,
                subfolder: uploaded?.subfolder || "",
                type: uploaded?.type || "input",
                width: dims.width || 0,
                height: dims.height || 0,
            };
        }
        const dims = await this.probeInputImageDimensions(choice.relPath, choice.type || "input");
        return {
            imageFile: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
            width: dims.width || 0,
            height: dims.height || 0,
        };
    }

    async chooseVideoInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "video",
            title: opts.title || t("mediaPicker.pickVideo"),
            accept: "video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
            // Rendered clips count as pickable source material here — that is
            // the whole point of reusing an earlier render as a reference.
            // Entries render as ordinary rows; nothing about the picker's
            // look changes.
            includeCache: true,
        });
        return this._resolveVideoChoice(choice);
    }

    /** 批量版：返回数组（元素结构同 chooseVideoInput）。 */
    async chooseVideoInputs(opts = {}) {
        const choices = await this.showInputMediaPicker({
            kind: "video",
            title: opts.title || t("mediaPicker.pickReferenceVideo"),
            accept: "video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
            multi: true,
            includeCache: true,
        });
        if (!Array.isArray(choices) || !choices.length) return [];
        const out = [];
        for (const c of choices) {
            const r = await this._resolveVideoChoice(c);
            if (r) out.push(r);
        }
        return out;
    }

    async _resolveVideoChoice(choice) {
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            const uploaded = await uploadToInputSmart(choice.file);
            return {
                relPath: videoRelativePath(uploaded),
                fileName: uploaded?.name || choice.file.name || "",
                subfolder: uploaded?.subfolder || "",
                type: uploaded?.type || "input",
            };
        }
        return {
            relPath: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
        };
    }

    async chooseAudioInput(opts = {}) {
        const choice = await this.showInputMediaPicker({
            kind: "reference_audio",
            title: opts.title || t("mediaPicker.pickAudio"),
            accept: "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
        });
        return this._resolveAudioChoice(choice);
    }

    /** 批量版：返回数组（元素结构同 chooseAudioInput）。 */
    async chooseAudioInputs(opts = {}) {
        const choices = await this.showInputMediaPicker({
            kind: "reference_audio",
            title: opts.title || t("mediaPicker.pickReferenceAudio"),
            accept: "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts",
            currentValue: opts.currentValue || "",
            multi: true,
        });
        if (!Array.isArray(choices) || !choices.length) return [];
        const out = [];
        for (const c of choices) {
            const r = await this._resolveAudioChoice(c);
            if (r) out.push(r);
        }
        return out;
    }

    async _resolveAudioChoice(choice) {
        if (!choice) return null;
        if (choice.source === "file" && choice.file) {
            return prepareLocalReferenceAudio(choice.file);
        }
        if (choice.mediaKind === "video") {
            return extractReferenceAudioFromExistingVideo(choice);
        }
        return {
            relPath: choice.relPath,
            fileName: choice.fileName || choice.relPath,
            subfolder: choice.subfolder || "",
            type: choice.type || "input",
        };
    }








    onNodeResize() {
        if (this.isPlaying || this._pauseSettling) return;
        // Growable layout (no computeSize) → LiteGraph puts free space into computedHeight.
        bindDomWidgetContentComputeSize(this);
        this._resetLayoutStyles();
        this.applyZoomWidth();
        syncBatchPanelFillHeight(this);
        // Re-fill after LiteGraph finishes arranging widgets for the new node size.
        requestAnimationFrame(() => syncBatchPanelFillHeight(this));
        this.scheduleSettleRender();
    }











    /**
     * Touching clip pairs in visual (time) order. `rightIndex` is the array index
     * whose `continuityFromPrev` flag owns this joint.
     */







    /**
     * fl2v edge handles: top half → previous clip's right edge;
     * bottom half → next clip's left edge.
     */




    /**
     * While dragging a segment edge, keep group seconds / toolbar / output preview in sync
     * without rebuilding the whole batch DOM (that would break the drag).
     */




    addSplitAtMouse(e) {
        const { x } = this.getMousePos(e);
        this.splitAtFrame(this.xToFrame(x, this.getLayoutWidth()));
    }

    splitAtFrame(frame) {
        if (this.isGenMode()) {
            this.genSplitAtFrame(frame);
            return;
        }
        const total = this.getTotalFrames();
        if (frame <= MIN_SEG || frame >= total - MIN_SEG) return;
        const newSegs = [];
        for (const seg of [...this.timeline.segments].sort((a, b) => a.start - b.start)) {
            const end = seg.start + seg.length;
            if (frame > seg.start && frame < end) {
                newSegs.push({ ...seg, length: frame - seg.start });
                newSegs.push({ id: uid(), start: frame, length: end - frame, prompt: "", taskType: "", refs: [], referenceVideo: {} });
            } else newSegs.push({ ...seg });
        }
        this.timeline.segments = newSegs;
        this.selectedSplitFrame = null;
        this.commit();
        this.updateSplitPointUI();
    }

    equalSplit() {
        if (this.isGenMode()) {
            this.genEqualSplit();
            return;
        }
        const n = parseInt(this.equalCountInput?.value || "2", 10);
        if (!n || n < 2) return;
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) return;
        const maxSeg = Math.floor(total / MIN_SEG);
        const count = clamp(n, 2, Math.max(2, maxSeg || 2));
        if (this.equalCountInput) this.equalCountInput.value = String(count);

        const points = new Set([0, total]);
        const clipBounds = this.getClipBoundaries();
        for (const b of clipBounds) {
            if (b > 0 && b < total) points.add(b);
        }
        for (let i = 1; i < count; i++) {
            const p = Math.round((i * total) / count);
            if (p > 0 && p < total) points.add(p);
        }

        const forced = new Set([0, total, ...clipBounds]);
        const newSegs = this._buildSegmentsFromSplitPoints([...points], forced);
        if (!newSegs?.length) return;
        this.timeline.segments = newSegs;
        this.selectedSplitFrame = null;
        this.commit();
        this.updateSplitPointUI();
    }

    /** Logical ranges for each video clip on the timeline. */
    getClipLogicalRanges() {
        const clips = this.getVideoClips();
        const total = this.getTotalFrames();
        if (!clips.length) return [];
        const map = this.getFrameMap();
        if (map.length) {
            const ranges = clips.map((clip, clipIndex) => ({
                clip,
                clipIndex,
                start: total,
                end: 0,
            }));
            for (let i = 0; i < map.length; i++) {
                const entry = normalizeFrameMapEntry(map[i]);
                const r = ranges[entry.clip];
                if (!r) continue;
                if (i < r.start) r.start = i;
                if (i + 1 > r.end) r.end = i + 1;
            }
            return ranges.filter((r) => r.end > r.start);
        }
        if (clips.length === 1) {
            return [{ clip: clips[0], clipIndex: 0, start: 0, end: total }];
        }
        let cursor = 0;
        return clips.map((clip, clipIndex) => {
            const len = Math.max(0, parseInt(clip.sourceFrameCount, 10) || 0);
            const start = cursor;
            const end = Math.min(total, cursor + len);
            cursor = end;
            return { clip, clipIndex, start, end };
        }).filter((r) => r.end > r.start);
    }

    /** Interior segment boundaries that can be selected/deleted (not clip seams). */
    getEditableSplitFrames() {
        if (this.isFl2vMode() || this.isGenMode() || this.isImageBatch()) return [];
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) return [];
        const forced = new Set([0, total, ...this.getClipBoundaries()]);
        const segs = this._previewSegments || this.timeline.segments || [];
        const points = [];
        for (const seg of segs) {
            const start = Math.max(0, parseInt(seg.start, 10) || 0);
            if (start > 0 && start < total && !forced.has(start)) points.push(start);
        }
        return [...new Set(points)].sort((a, b) => a - b);
    }

    selectSplitFrame(frame) {
        const editable = this.getEditableSplitFrames();
        const n = Number(frame);
        if (!Number.isFinite(n) || !editable.includes(n)) {
            this.selectedSplitFrame = null;
        } else {
            // Toggle off if clicking the same selected split again.
            this.selectedSplitFrame = this.selectedSplitFrame === n ? null : n;
            if (this.selectedSplitFrame != null) {
                const segs = this.timeline.segments || [];
                const idx = segs.findIndex((s) => (parseInt(s.start, 10) || 0) === n);
                if (idx >= 0) this.selectedIndex = idx;
            }
        }
        this.updateSplitPointUI();
        this.updateSelectionUI();
        this.scheduleRender();
    }

    clearSplitSelection() {
        if (this.selectedSplitFrame == null) return;
        this.selectedSplitFrame = null;
        this.updateSplitPointUI();
        this.scheduleRender();
    }

    updateSplitPointUI() {
        const bar = this.splitEditBarEl || this.root?.querySelector('[data-r="split-edit-bar"]');
        const hint = this.splitEditHintEl || this.root?.querySelector('[data-r="split-edit-hint"]');
        const btn = this.root?.querySelector('[data-a="del-split"]');
        if (this.isImageBatch() || this.isGenMode()) {
            bar?.classList.add("hidden");
            return;
        }
        const has = this.selectedSplitFrame != null
            && this.getEditableSplitFrames().includes(this.selectedSplitFrame);
        if (bar) bar.classList.toggle("hidden", !has);
        if (hint && has) {
            hint.textContent = t("split.hintSelected", { f: this.selectedSplitFrame });
        }
        if (btn) {
            btn.disabled = !has;
            btn.title = has
                ? t("split.tooltipDelete", { f: this.selectedSplitFrame })
                : t("split.tooltipSelectFirst");
        }
        if (has && this.boundsEl) {
            this.boundsEl.textContent = t("split.boundsSelected", { f: this.selectedSplitFrame });
        }
    }

    deleteSelectedSplitPoint() {
        if (this.isGenMode() || this.isImageBatch()) return;
        const frame = this.selectedSplitFrame;
        if (frame == null) return;
        if (!this.getEditableSplitFrames().includes(frame)) {
            this.clearSplitSelection();
            return;
        }
        const segs = [...(this.timeline.segments || [])].sort((a, b) => a.start - b.start);
        const rightIdx = segs.findIndex((s) => (parseInt(s.start, 10) || 0) === frame);
        if (rightIdx <= 0) {
            this.clearSplitSelection();
            return;
        }
        const left = segs[rightIdx - 1];
        const right = segs[rightIdx];
        left.length = (parseInt(left.length, 10) || 0) + (parseInt(right.length, 10) || 0);
        segs.splice(rightIdx, 1);
        this.timeline.segments = segs;
        // The right-hand segment is gone: drop its cache and rebase the ticks.
        this.onSegmentRemoved(rightIdx);
        this.selectedSplitFrame = null;
        this.selectedIndex = Math.max(0, rightIdx - 1);
        this.commit();
        this.updateSelectionUI();
        this.updateSplitPointUI();
        this.setSmartSplitMessage("");
        this.scheduleRender();
    }

    setSmartSplitMessage(text, { ok = false } = {}) {
        const el = this.smartSplitMsgEl || this.root?.querySelector('[data-r="smart-split-msg"]');
        if (!el) return;
        const msg = String(text || "").trim();
        if (!msg) {
            el.textContent = "";
            el.classList.add("hidden");
            el.classList.remove("ok");
            return;
        }
        el.textContent = msg;
        el.classList.toggle("ok", !!ok);
        el.classList.remove("hidden");
    }

    async smartSplit() {
        if (this.isGenMode() || this.isImageBatch()) return;
        if (!this.hasVideo()) {
            this.setSmartSplitMessage(t("smartSplit.needVideo"));
            return;
        }
        const total = this.getTotalFrames();
        if (total < MIN_SEG * 2) {
            this.setSmartSplitMessage(t("smartSplit.tooShort"));
            return;
        }
        const ranges = this.getClipLogicalRanges();
        if (!ranges.length) {
            this.setSmartSplitMessage(t("smartSplit.noMaterial"));
            return;
        }
        const btn = this.root?.querySelector('[data-a="smart-split"]');
        const prevLabel = btn?.textContent;
        if (btn) {
            btn.disabled = true;
            btn.textContent = t("common.analyzing");
        }
        this.setSmartSplitMessage(t("smartSplit.analyzing"));
        try {
            const clips = ranges.map((r) => ({
                videoFile: r.clip.videoFile || r.clip.fileName,
                subfolder: r.clip.subfolder || "",
                type: r.clip.type || "input",
                logicalStart: r.start,
                logicalEnd: r.end,
                nativeFps: r.clip.nativeFps || r.clip.native_fps || null,
            }));
            const resp = await api.fetchApi("/minimax/director_opt/detect_shots", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    clips,
                    frameRate: this.getFrameRate(),
                    totalFrames: total,
                    sensitivity: "medium",
                    minShotFrames: Math.max(MIN_SEG, 12),
                }),
            });
            if (!resp.ok) {
                throw new Error((await resp.text()) || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const cutFrames = Array.isArray(data.cutFrames) ? data.cutFrames.map((n) => parseInt(n, 10) || 0) : [];
            const points = new Set([0, total, ...cutFrames.filter((f) => f > 0 && f < total)]);
            const clipBounds = this.getClipBoundaries();
            for (const b of clipBounds) {
                if (b > 0 && b < total) points.add(b);
            }
            const forced = new Set([0, total, ...clipBounds]);
            const newSegs = this._buildSegmentsFromSplitPoints([...points], forced);
            if (!newSegs?.length) {
                this.setSmartSplitMessage(t("smartSplit.noSegments"));
                return;
            }
            this.timeline.segments = newSegs;
            this.selectedIndex = 0;
            this.selectedSplitFrame = null;
            this.commit();
            this.updateSelectionUI();
            this.updateSplitPointUI();
            const shotCount = data.shotCount ?? Math.max(0, newSegs.length);
            const warn = Array.isArray(data.warnings) && data.warnings.length
                ? ` ${data.warnings[0]}`
                : "";
            this.setSmartSplitMessage(
                t("smartSplit.done", { shots: shotCount, segs: newSegs.length }) + (warn || ""),
                { ok: !warn },
            );
        } catch (err) {
            console.error("[MiniMax H3 Director Opt] smartSplit failed", err);
            this.setSmartSplitMessage(t("smartSplit.failed", { err: err?.message || err }));
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = prevLabel || t("toolbar.smartSplit");
            }
        }
    }

    deleteSelectedSegment() {
        if (this.isGenMode()) {
            this.genDeleteSelectedSegment();
            return;
        }
        if (this.usesBatchTimeline()) {
            deleteImageBatchGroup(this, this.selectedIndex);
            this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, this.getTotalFrames() - 1));
            if (this.seekBar) {
                this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
                this.seekBar.value = this.currentFrame;
            }
            this.updateVideoNameLabel();
            this.updateDomWidgetHeight();
            this.scheduleRender();
            return;
        }
        if (this.isImageBatch()) {
            this.genDeleteSelectedSegment();
            return;
        }
        if (this.isFl2vMode()) {
            const idx = this.selectedIndex;
            const shots = this.timeline.shots || [];
            if (!shots[idx] && !(this.timeline.segments || [])[idx]) return;
            // Only rebase ticks / drop the cache when a shot really went away.
            if (removeFl2vShot(this, idx)) this.onSegmentRemoved(idx);
            this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, this.getTotalFrames() - 1));
            if (this.seekBar) {
                this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
                this.seekBar.value = this.currentFrame;
            }
            this.commit(false, { syncTimeline: true });
            updateFl2vDetailUI(this);
            this.updateVideoNameLabel();
            this.updateDomWidgetHeight();
            return;
        }
        const idx = this.selectedIndex;
        const seg = this.timeline.segments[idx];
        if (!seg) return;

        const start = Math.max(0, parseInt(seg.start, 10) || 0);
        const len = Math.max(0, parseInt(seg.length, 10) || 0);
        this.selectedSplitFrame = null;

        // Remove segment UI entry first, then cut matching frames from the
        // logical timeline so preview / export no longer include that range.
        this.timeline.segments.splice(idx, 1);
        this.onSegmentRemoved(idx);

        let total = this.getTotalFrames();
        let map = [];
        if (len > 0 && total > 0) {
            // Sparse uploads start with an empty frameMap; materialize so we can
            // splice out the deleted range from the source-frame mapping.
            if (!this.getFrameMap().length) this.materializeFrameMap();
            map = [...this.getFrameMap()];
            if (map.length) {
                const from = clamp(start, 0, map.length);
                const count = clamp(len, 0, map.length - from);
                if (count > 0) map.splice(from, count);
                this.setFrameMap(map);
                this._syncPrimaryVideoFromClips(map);
                total = map.length;
            } else {
                // Fallback: record deleted source ranges (kept across sync).
                const video = this.timeline.video || {};
                video.deletedSourceRanges = video.deletedSourceRanges || [];
                const srcStart = this.logicalToSourceFrame(start);
                video.deletedSourceRanges.push([srcStart, srcStart + len]);
                video.deletedSourceRanges.sort((a, b) => a[0] - b[0]);
                this.timeline.video = video;
                total = this.getTotalFrames();
                this.timeline.totalFrames = total;
                this._syncPrimaryVideoFromClips([]);
            }
        }

        // Keep thumbs for remaining source frames; drop only if the clip is gone unused.
        this.timeline.videoWorkspace = null;

        if (this.totalFramesWidget) this.totalFramesWidget.value = total;

        this.compactSegmentsAfterDelete();

        this.selectedIndex = clamp(idx, 0, Math.max(0, this.timeline.segments.length - 1));
        this.currentFrame = clamp(this.currentFrame, 0, Math.max(0, total - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = this.currentFrame;
        }

        if (!total) {
            this.videoNameEl.textContent = t("toolbar.noVideo");
            this._clearVideoState({ dropUnusedThumbs: true });
        } else {
            this.updateVideoNameLabel();
            this._prefetchSegmentThumbs(0, Math.min(total, THUMB_PREFETCH_BATCH * 4));
            this._syncStagePreview(this.currentFrame, { force: true });
            this.updateStageVisibility();
        }

        this.commit(false, { syncTimeline: true });
    }

    compactSegmentsAfterDelete() {
        const total = this.getTotalFrames();
        if (total <= 0) {
            this.timeline.segments = [];
            return;
        }
        const segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        if (!segs.length) {
            this.timeline.segments = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
            return;
        }
        let cursor = 0;
        const fixed = [];
        for (const seg of segs) {
            let length = seg.length ?? MIN_SEG;
            if (cursor + length > total) length = total - cursor;
            if (length < MIN_SEG) {
                if (fixed.length) fixed[fixed.length - 1].length += length;
                cursor += length;
                continue;
            }
            fixed.push({ ...seg, start: cursor, length, refs: seg.refs || [] });
            cursor += length;
        }
        if (!fixed.length) {
            this.timeline.segments = [{ id: uid(), start: 0, length: total, prompt: "", taskType: "", refs: [], referenceVideo: {} }];
        } else if (cursor < total) {
            fixed[fixed.length - 1].length += total - cursor;
        }
        this.timeline.segments = fixed;
    }











    /** Jump to an exact 0-based logical frame; syncs seek bar, preview, playhead. */
    seekToFrame(frame, { fromUi = false } = {}) {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.isPlaying) this._stopPlay();
        const next = clamp(Math.round(Number(frame) || 0), 0, total - 1);
        this.currentFrame = next;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, total - 1);
            this.seekBar.value = next;
        }
        this._syncStagePreview(next, { force: true });
        this._updateTimelineDom({ skipSeek: true });
        // Select the segment that contains this frame for editing context.
        const segs = this.timeline.segments || [];
        for (let i = 0; i < segs.length; i++) {
            const s = segs[i];
            if (next >= s.start && next < s.start + s.length) {
                if (this.selectedIndex !== i) {
                    this.selectedIndex = i;
                    this.updateSelectionUI();
                }
                break;
            }
        }
        this.scheduleRender();
        if (fromUi) this._queueThumbPrefetch?.(next);
    }

    stepFrame(delta) {
        const total = this.getTotalFrames();
        if (total < 1) return;
        this.seekToFrame(this.currentFrame + (Number(delta) || 0), { fromUi: true });
    }

    formatTime(frames) { return (frames / this.getFrameRate()).toFixed(2); }

    updateSelectionUI() {
        this.timeline.global = this.timeline.global || { taskType: "", prompt: "", refs: [] };
        if (this.globalTask) this.globalTask.value = this.timeline.global.taskType || "";
        if (this.globalPrompt) this.globalPrompt.value = this.timeline.global.prompt || "";
        this.syncNegativeFromWidget();
        updateFl2vToolbarBtns(this);
        updateR2vToolbarBtns(this);
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
        if (this.isImageBatch()) {
            const shown = this.batchList?.querySelector(".bd-batch-card");
            const shownIdx = shown ? parseInt(shown.dataset.batchIndex, 10) : NaN;
            if (isBatchDetailSolo(this) && shownIdx !== this.selectedIndex) {
                this.renderImageBatchGroups();
            } else {
                this._syncR2vCardSelection();
            }
        }

        const r2vOn = this.isR2vCommonEnabled();
        const hideTimeline = (this.isImageBatch() && !r2vOn) || this.isGenMode();
        const seg = this.usesGlobalRefPanel() ? null : this.timeline.segments[this.selectedIndex];
        this.updateReferenceImageVisibility({ hideTimeline, seg: seg || null });

        if (this.usesGlobalRefPanel() && taskUsesReferenceImages(this.getTaskKey())) {
            this.timeline.global.refs = this.timeline.global.refs || [];
            this.renderRefSlots(this.timeline.global.refs, this.globalRefsBox, true);
        }
        if (this.usesGlobalRefPanel() && taskUsesReferenceAudios(this.getTaskKey())) {
            this.timeline.global.refAudios = this.timeline.global.refAudios || [];
            this.renderRefAudioSlots();
        }
        if (this.usesGlobalRefPanel() && this.usesR2vCommonPanel()) {
            this.timeline.global.refVideos = this.timeline.global.refVideos || [];
            this.renderR2vCommonVideoSlots();
        }
        const refVideoKey = this.usesGlobalRefPanel()
            ? this.getTaskKey()
            : resolveTaskKey(seg?.taskType || this.timeline.global?.taskType || this.getTaskKey());
        if (taskUsesReferenceVideo(refVideoKey)) {
            this.renderRefVideoSlot();
        }
        if (this.isGenImage() && this.isGlobalMode()) {
            this.renderGenSrcSlot(
                this.genGlobalImg,
                this.timeline.global?.genImage?.imageFile,
                t("panel.uploadSourceImage"),
            );
        }
        if (this.isGenMode() && this.isGlobalMode()) {
            const defFc = this.timeline.gen?.defaultFrameCount ?? defaultFrameCount(this.getTaskKey());
            if (this.genDefaultFc) this.genDefaultFc.value = defFc;
        }

        if (this.usesGlobalRefPanel()) {
            this.syncSegmentRefImageSizeUI();
            return;
        }

        if (!seg) return;
        const liveSeg = (this._previewSegments || this.timeline.segments)?.[this.selectedIndex] || seg;
        const segKey = resolveTaskKey(liveSeg.taskType || this.timeline.global?.taskType || this.getTaskKey());
        this.segLabel.textContent = t("panel.segmentN", { n: this.selectedIndex + 1 });
        this.syncSegmentContinuityFromPrevUI();
        this.syncSegmentRefImageSizeUI();
        this._updateSegInfoFromSegment(liveSeg);
        this.segPrompt.value = liveSeg.prompt || "";
        if (taskUsesReferenceImages(segKey)) {
            this.renderRefSlots(liveSeg.refs, this.segRefsBox, false);
        }
        if (taskUsesReferenceAudios(segKey)) {
            this.renderRefAudioSlots();
        }
        if (this.isGenImage() && !this.isGlobalMode()) {
            this.renderGenSrcSlot(this.genSegImg, liveSeg.genImage?.imageFile, t("panel.uploadSegmentSourceImage"));
        }
        if (this.isGenMode() && !this.isGlobalMode()) {
            const fc = liveSeg.frameCount ?? liveSeg.length ?? defaultFrameCount(this.getTaskKey());
            if (this.genSegFc) this.genSegFc.value = fc;
        }
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
    }

    renderRefSlots(refs, box, isGlobal) {
        if (!box) return;
        box.innerHTML = "";
        const target = isGlobal
            ? this.timeline.global
            : this.timeline.segments[this.selectedIndex];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const wrap = isGlobal ? this.globalRefsImagesWrap : this.segRefsImagesWrap;
        const countEl = isGlobal ? this.globalRefsCount : this.segRefsCount;
        const PIC_STEP = 3;
        const PIC_SLOTS = MAX_REFERENCE_IMAGES;

        let filled = 0;
        let highestFilled = -1;
        for (const r of refs || []) {
            const idx = Number(r.index ?? r.slot);
            const has = !!(r?.imageFile || r?.imageB64);
            if (!has || !Number.isFinite(idx) || idx < 0 || idx >= PIC_SLOTS) continue;
            filled += 1;
            highestFilled = Math.max(highestFilled, idx);
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${PIC_SLOTS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-refs-pick"]' : '[data-r="seg-refs-pick"]',
            filled >= PIC_SLOTS,
        );

        if (!this._rv2vPicsVisible) this._rv2vPicsVisible = {};
        const visKey = isGlobal ? "global" : `seg:${target?.id ?? this.selectedIndex}`;
        const minVisible = highestFilled >= 0
            ? Math.min(PIC_SLOTS, Math.ceil((highestFilled + 1) / PIC_STEP) * PIC_STEP)
            : PIC_STEP;
        let visible = polished
            ? (Number(this._rv2vPicsVisible[visKey]) || PIC_STEP)
            : PIC_SLOTS;
        if (polished) {
            visible = Math.max(PIC_STEP, Math.min(PIC_SLOTS, visible));
            if (visible < minVisible) visible = minVisible;
            this._rv2vPicsVisible[visKey] = visible;
        }

        for (let i = 0; i < PIC_SLOTS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref";
            if (polished && i >= visible) el.classList.add("bd-r2v-pic-hidden");
            el.dataset.refSlot = String(i);
            el.dataset.refKind = "image";
            el.dataset.refIndex = String(i);
            el.dataset.refScope = isGlobal ? "global" : "seg";
            const label = refImageLabel(i);
            el.title = t("ref.slotTitle", { label });
            const ref = (refs || []).find((r) => Number(r.index ?? r.slot) === i);
            const tag = document.createElement("span");
            tag.className = polished ? "cap" : "bd-ref-tag";
            tag.textContent = label;
            el.appendChild(tag);
            if (ref?.imageFile) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = refViewUrl(ref.imageFile);
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            } else if (ref?.imageB64) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            }
            this._bindRefSlotDnD(el, target, i, isGlobal);
            el.onclick = () => {
                if (this._refDragMoved) {
                    this._refDragMoved = false;
                    return;
                }
                this.pickRef(target, i, isGlobal);
            };
            box.appendChild(el);
        }

        wrap?.querySelectorAll(".bd-r2v-pics-toggle").forEach((btn) => btn.remove());
        if (polished && wrap) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "bd-r2v-pics-toggle";
            const syncToggleLabel = () => {
                if (visible < PIC_SLOTS) {
                    const next = Math.min(PIC_STEP, PIC_SLOTS - visible);
                    toggle.textContent = t("batch.r2v.expandPics", { n: next });
                } else {
                    toggle.textContent = t("batch.r2v.collapsePics");
                }
            };
            syncToggleLabel();
            toggle.onclick = (e) => {
                e.stopPropagation();
                if (visible < PIC_SLOTS) {
                    visible = Math.min(PIC_SLOTS, visible + PIC_STEP);
                } else {
                    visible = Math.max(PIC_STEP, minVisible);
                }
                this._rv2vPicsVisible[visKey] = visible;
                box.querySelectorAll(".bd-ref").forEach((el, i) => {
                    el.classList.toggle("bd-r2v-pic-hidden", i >= visible);
                });
                syncToggleLabel();
                this.updateDomWidgetHeight?.();
            };
            wrap.appendChild(toggle);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    _bindRefSlotDnD(el, target, slotIndex, isGlobal) {
        const hasImg = el.classList.contains("has-img");
        el.draggable = hasImg;
        el.addEventListener("dragstart", (e) => {
            if (!hasImg) {
                e.preventDefault();
                return;
            }
            this._refDragMoved = false;
            const payload = JSON.stringify({
                scope: isGlobal ? "global" : "seg",
                segIndex: isGlobal ? -1 : this.selectedIndex,
                from: slotIndex,
            });
            e.dataTransfer.setData("application/x-minimax-ref-slot", payload);
            e.dataTransfer.setData("text/plain", payload);
            e.dataTransfer.effectAllowed = "move";
        });
        el.addEventListener("dragend", () => {
            // click may fire after dragend; keep suppress for one tick
            setTimeout(() => { this._refDragMoved = false; }, 0);
        });
        el.addEventListener("dragover", (e) => {
            const types = e.dataTransfer?.types || [];
            if (![...types].includes("application/x-minimax-ref-slot") && ![...types].includes("Files")) {
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = [...types].includes("application/x-minimax-ref-slot")
                ? "move"
                : "copy";
        });
        el.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const raw = e.dataTransfer.getData("application/x-minimax-ref-slot")
                || e.dataTransfer.getData("text/plain");
            if (raw) {
                try {
                    const data = JSON.parse(raw);
                    const scope = isGlobal ? "global" : "seg";
                    if (data.scope !== scope) return;
                    if (!isGlobal && data.segIndex !== this.selectedIndex) return;
                    this._refDragMoved = true;
                    this.moveRefSlot(target, Number(data.from), slotIndex, isGlobal);
                    return;
                } catch (_) { /* fall through to file drop */ }
            }
            const f = e.dataTransfer.files?.[0];
            if (f?.type?.startsWith("image/")) {
                this.addRefFromFile(f, target, slotIndex, isGlobal);
            }
        });
    }

    moveRefSlot(target, fromIndex, toIndex, isGlobal) {
        if (!target || fromIndex === toIndex) return;
        const refs = [...(target.refs || [])];
        const fromRef = refs.find((r) => Number(r.index ?? r.slot) === fromIndex);
        if (!fromRef) return;
        const toRef = refs.find((r) => Number(r.index ?? r.slot) === toIndex);
        target.refs = refs.filter((r) => {
            const idx = Number(r.index ?? r.slot);
            return idx !== fromIndex && idx !== toIndex;
        });
        target.refs.push({ ...fromRef, index: toIndex, slot: undefined });
        if (toRef) {
            target.refs.push({ ...toRef, index: fromIndex, slot: undefined });
        }
        if (isGlobal) {
            this.timeline.global = target;
            if (this.isR2vCommonEnabled()) rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    }

    removeRef(target, index) {
        target.refs = (target.refs || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    }

    renderRefAudioSlots() {
        const isGlobal = this.usesGlobalRefPanel();
        const box = isGlobal ? this.globalRefAudiosBox : this.segRefAudiosBox;
        if (!box) return;
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const countEl = isGlobal ? this.globalAudiosCount : this.segAudiosCount;
        let filled = 0;
        for (const r of target.refAudios) {
            if (r?.audioFile || r?.fileName) filled += 1;
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${MAX_REFERENCE_AUDIOS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-audios-pick"]' : '[data-r="seg-audios-pick"]',
            filled >= MAX_REFERENCE_AUDIOS,
        );

        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_AUDIOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-audio";
            el.dataset.audioSlot = String(i);
            el.dataset.refKind = "audio";
            el.dataset.refIndex = String(i);
            const label = refAudioLabel(i);
            const ref = (target.refAudios || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.audioFile || ref?.fileName || "";
            el.title = file
                ? t("ref.audioTitleFilled", { label, file })
                : t("ref.audioTitleEmpty", { label });
            if (polished) {
                const thumb = document.createElement("div");
                thumb.className = "bd-r2v-thumb";
                const meta = document.createElement("div");
                meta.className = "bd-r2v-meta";
                const tag = document.createElement("span");
                tag.className = "tag";
                tag.textContent = label;
                meta.appendChild(tag);
                el.appendChild(thumb);
                el.appendChild(meta);
                if (file) {
                    el.classList.add("has-audio");
                    const playBtn = document.createElement("button");
                    playBtn.type = "button";
                    playBtn.className = "bd-r2v-play";
                    playBtn.title = t("batch.r2v.play");
                    playBtn.textContent = "▶";
                    thumb.appendChild(playBtn);
                    const dur = document.createElement("span");
                    dur.className = "bd-r2v-dur";
                    dur.textContent = ref?.durationSec != null
                        ? formatMediaDuration(ref.durationSec)
                        : "--:--";
                    meta.appendChild(dur);
                    const progress = document.createElement("div");
                    progress.className = "bd-r2v-progress";
                    progress.title = t("batch.r2v.seek");
                    progress.innerHTML = `<div class="bd-r2v-progress-fill"></div>`;
                    el.appendChild(progress);
                    const audio = document.createElement("audio");
                    audio.preload = "metadata";
                    audio.src = refViewUrl(file);
                    audio.className = "bd-r2v-media";
                    el.appendChild(audio);
                    bindR2vMediaPlayback(audio, playBtn, progress);
                    wireMediaDuration(audio, dur, (sec) => {
                        if (ref) ref.durationSec = sec;
                    });
                    const x = document.createElement("span");
                    x.className = "x";
                    x.textContent = "×";
                    x.onclick = (e) => {
                        e.stopPropagation();
                        this.removeRefAudio(target, i);
                    };
                    el.appendChild(x);
                } else {
                    thumb.textContent = "♪";
                    const hint = document.createElement("span");
                    hint.className = "name";
                    hint.textContent = t("batch.r2v.uploadHint");
                    meta.appendChild(hint);
                }
            } else if (file) {
                el.classList.add("has-audio");
                const tag = document.createElement("span");
                tag.textContent = label;
                el.appendChild(tag);
                const name = document.createElement("span");
                name.className = "bd-ref-audio-name";
                name.textContent = file.split("/").pop() || file;
                el.appendChild(name);
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRefAudio(target, i);
                };
                el.appendChild(x);
            } else {
                el.textContent = t("ref.audioUpload", { label });
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-progress, .x")) return;
                this.pickRefAudio(target, i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    removeRefAudio(target, index) {
        if (!target) return;
        target.refAudios = (target.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
        this.renderRefAudioSlots();
    }

    pickRefAudio(target, index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefAudioFromFile(file, target, index);
        };
        input.click();
    }

    async addRefAudioFromFile(file, target, slotIndex = null) {
        if (!target || !file) return;
        target.refAudios = target.refAudios || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_AUDIOS }, (_, i) => i)
                .find((i) => !target.refAudios.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const prepared = await prepareLocalReferenceAudio(file);
            const relPath = prepared.relPath;
            if (hasDuplicateReferenceAudio(target.refAudios, relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: relPath,
                fileName: prepared.fileName || file.name,
                type: prepared.type || "input",
                subfolder: prepared.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && target === this.timeline.global) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio upload failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    }

    /** r2v common panel: multi-slot global.refVideos (1–3), merged into groups at run time. */
    renderR2vCommonVideoSlots() {
        const box = this.globalRefVideosBox;
        if (!box || !this.usesR2vCommonPanel()) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let filled = 0;
        for (const r of target.refVideos) {
            if (r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked) {
                filled += 1;
            }
        }
        if (this.globalVideosCount) {
            this.globalVideosCount.textContent = `${filled}/${MAX_REFERENCE_VIDEOS}`;
        }
        this._syncPickExistingDisabled('[data-r="global-videos-pick"]', filled >= MAX_REFERENCE_VIDEOS);
        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_VIDEOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-video";
            el.dataset.videoSlot = String(i);
            el.dataset.refKind = "video";
            el.dataset.refIndex = String(i);
            const label = refVideoLabel(i);
            const ref = (target.refVideos || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.videoFile || "";
            const posterSrc = ref?.previewImageUrl
                || (ref?.previewImageFile ? refViewUrl(ref.previewImageFile) : "");
            const hasMedia = !!(file || posterSrc || ref?.linked);
            const titleFile = file || ref?.fileName || ref?.previewImageFile || "";
            el.title = hasMedia
                ? t("ref.videoTitleFilled", { label, file: titleFile || label })
                : t("ref.videoTitleEmpty", { label });
            const thumb = document.createElement("div");
            thumb.className = "bd-r2v-thumb bd-r2v-thumb-video";
            const meta = document.createElement("div");
            meta.className = "bd-r2v-meta";
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.textContent = label;
            meta.appendChild(tag);
            el.appendChild(thumb);
            el.appendChild(meta);
            if (file) {
                el.classList.add("has-video");
                const video = document.createElement("video");
                video.preload = "metadata";
                video.muted = true;
                video.playsInline = true;
                video.src = refViewUrl(file);
                video.className = "bd-r2v-media";
                thumb.appendChild(video);
                const playBtn = document.createElement("button");
                playBtn.type = "button";
                playBtn.className = "bd-r2v-play";
                playBtn.title = t("batch.r2v.play");
                playBtn.textContent = "▶";
                thumb.appendChild(playBtn);
                const dur = document.createElement("span");
                dur.className = "bd-r2v-dur";
                dur.textContent = ref?.durationSec != null
                    ? formatMediaDuration(ref.durationSec)
                    : "--:--";
                meta.appendChild(dur);
                bindR2vMediaPlayback(video, playBtn);
                playBtn.addEventListener("click", () => { video.muted = false; });
                wireMediaDuration(video, dur, (sec) => {
                    if (ref) ref.durationSec = sec;
                });
                video.addEventListener("loadeddata", () => {
                    if (video.readyState >= 2 && video.currentTime < 0.05) {
                        try {
                            video.currentTime = Math.min(0.1, (video.duration || 1) * 0.05);
                        } catch (_) { /* ignore */ }
                    }
                }, { once: true });
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeR2vCommonVideo(i);
                };
                el.appendChild(x);
            } else if (posterSrc) {
                el.classList.add("has-video");
                const img = document.createElement("img");
                img.className = "bd-r2v-media";
                img.src = posterSrc;
                img.alt = label;
                thumb.appendChild(img);
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.externalPoster");
                meta.appendChild(hint);
            } else {
                thumb.textContent = "▶";
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.uploadHint");
                meta.appendChild(hint);
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-dur, .x, video")) return;
                if (file && e.target?.closest?.(".bd-r2v-thumb")) {
                    el.querySelector(".bd-r2v-play")?.click();
                    return;
                }
                this.pickR2vCommonVideo(i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    removeR2vCommonVideo(index) {
        const target = this.timeline.global;
        if (!target) return;
        target.refVideos = (target.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled()) {
            rebaseR2vGroupSlotsForCommon(this);
            this.renderImageBatchGroups?.();
        }
        this.commit();
        this.renderR2vCommonVideoSlots();
    }

    pickR2vCommonVideo(index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*,.mp4,.mov,.webm,.mkv";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addR2vCommonVideoFromFile(file, index);
        };
        input.click();
    }

    async pickExistingR2vCommonVideo() {
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        const index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
            .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i && (r.videoFile || r.fileName)));
        if (index == null) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseVideoInput({
                title: t("mediaPicker.pickReferenceVideo"),
            });
            if (!picked?.relPath) return;
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video pick failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    }

    async addR2vCommonVideoFromFile(file, slotIndex = null) {
        if (!file) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
                .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInputSmart(file);
            const relPath = videoRelativePath(uploaded);
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: relPath,
                fileName: uploaded?.name || file.name,
                type: "input",
                subfolder: uploaded?.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video upload failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    }

    pickRef(target, index, isGlobal) {
        const input = document.createElement("input");
        input.type = "file"; input.accept = "image/*";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefFromFile(file, target, index, isGlobal);
        };
        input.click();
    }

    _nextEmptyMediaSlot(items, max, hasFn) {
        for (let i = 0; i < max; i++) {
            const hit = (items || []).find((r) => Number(r.index ?? r.slot) === i);
            if (!hasFn(hit)) return i;
        }
        return -1;
    }

    _syncPickExistingDisabled(selector, disabled) {
        const btn = this.root?.querySelector(selector);
        if (!btn) return;
        btn.disabled = !!disabled;
        btn.title = disabled ? t("mediaPicker.slotsFull") : t("mediaPicker.pickExistingHint");
    }

    async pickExistingRef(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refs = target.refs || [];
        const index = this._nextEmptyMediaSlot(
            target.refs,
            MAX_REFERENCE_IMAGES,
            (r) => !!(r?.imageFile || r?.imageB64),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseImageInput({
                title: t("mediaPicker.pickReferenceImage"),
            });
            if (!picked?.imageFile) return;
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: picked.imageFile, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
            this.renderRefSlots(
                target.refs,
                isGlobal ? this.globalRefsBox : this.segRefsBox,
                isGlobal,
            );
        } catch (err) {
            console.error("[MiniMax H3Director] ref pick failed:", err);
        }
    }

    async pickExistingRefAudio(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const index = this._nextEmptyMediaSlot(
            target.refAudios,
            MAX_REFERENCE_AUDIOS,
            (r) => !!(r?.audioFile || r?.fileName),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseAudioInput({
                title: t("mediaPicker.pickReferenceAudio"),
            });
            if (!picked?.relPath) return;
            if (hasDuplicateReferenceAudio(target.refAudios, picked.relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && isGlobal) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio pick failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    }

    async addRefFromFile(file, target, slotIndex = null, isGlobal = null) {
        target.refs = target.refs || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_IMAGES }, (_, i) => i)
                .find((i) => !target.refs.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInput(file);
            const relPath = videoRelativePath(uploaded);
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: relPath, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
        } catch (err) {
            console.error("[MiniMax H3Director] ref upload failed:", err);
        }
    }







    isLiveTaePreviewEnabled() {
        return this.timeline?.liveTaePreview !== false;
    }

    /** fl2v / v2v / rv2v (and aliases): show dedicated live-sample panel when toggle is on. */
    needsLiveSamplePanel() {
        if (!this.isLiveTaePreviewEnabled()) return false;
        if (this.isImageBatch?.()) return false;
        if (this.isFl2vMode?.()) return true;
        const key = this.getTaskKey?.() || "";
        return key === "v2v" || key === "mv2v" || key === "ads2v"
            || key === "rv2v" || key === "vrc2v" || key === "vi2v";
    }

    toggleLiveTaePreview() {
        this.timeline.liveTaePreview = !this.isLiveTaePreviewEnabled();
        this.refreshLiveTaePreviewButton();
        this.updateLiveSamplePanel();
        this.scheduleTimelineSync();
        this.updateDomWidgetHeight?.();
        syncDirectorNodeSize(this.node, this);
    }

    refreshLiveTaePreviewButton() {
        const btn = this.root?.querySelector('[data-a="live-tae-preview"]');
        if (!btn) return;
        const on = this.isLiveTaePreviewEnabled();
        btn.classList.toggle("active", on);
        btn.textContent = t("toolbar.liveTaePreview");
        btn.title = on ? t("tooltip.liveTaePreviewOn") : t("tooltip.liveTaePreviewOff");
        btn.setAttribute("data-i18n", "toolbar.liveTaePreview");
        btn.removeAttribute("data-i18n-title");
    }

    _clearEmbeddedLiveLayoutClasses() {
        this.globalPromptLayout?.classList.remove("bd-v2v-with-live", "bd-rv2v-with-live");
        this.segPromptLayout?.classList.remove("bd-v2v-with-live", "bd-rv2v-with-live");
    }

    _activePromptLayout() {
        return this.isGlobalMode?.() ? this.globalPromptLayout : this.segPromptLayout;
    }

    _placeLiveSamplePanel() {
        const panel = this.liveSampleEl;
        if (!panel) return;

        if (this.isFl2vMode?.() && this.fl2vUi?.workbench && this.fl2vUi?.shotsEl) {
            this._clearEmbeddedLiveLayoutClasses();
            if (this._liveSampleHost !== "fl2v" || panel.parentElement !== this.fl2vUi.workbench) {
                this.fl2vUi.workbench.insertBefore(panel, this.fl2vUi.shotsEl);
                this._liveSampleHost = "fl2v";
            }
            return;
        }

        const layout = this._activePromptLayout();

        // v2v: preview sits to the right of the prompt column.
        if (this.usesV2vPromptStyle?.() && this.isLiveTaePreviewEnabled() && layout) {
            if (panel.parentElement !== layout) layout.appendChild(panel);
            this.globalPromptLayout?.classList.toggle("bd-v2v-with-live", layout === this.globalPromptLayout);
            this.segPromptLayout?.classList.toggle("bd-v2v-with-live", layout === this.segPromptLayout);
            this.globalPromptLayout?.classList.remove("bd-rv2v-with-live");
            this.segPromptLayout?.classList.remove("bd-rv2v-with-live");
            this._liveSampleHost = "v2v";
            return;
        }

        // rv2v: preview under the prompt (same stack as r2v right column).
        if (this.usesRv2vRefStyle?.() && this.isLiveTaePreviewEnabled() && layout) {
            const promptCol = layout.querySelector(".bd-prompt-col");
            if (promptCol) {
                if (panel.parentElement !== promptCol) promptCol.appendChild(panel);
                this.globalPromptLayout?.classList.toggle("bd-rv2v-with-live", layout === this.globalPromptLayout);
                this.segPromptLayout?.classList.toggle("bd-rv2v-with-live", layout === this.segPromptLayout);
                this.globalPromptLayout?.classList.remove("bd-v2v-with-live");
                this.segPromptLayout?.classList.remove("bd-v2v-with-live");
                this._liveSampleHost = "rv2v";
                return;
            }
        }

        this._clearEmbeddedLiveLayoutClasses();
        if (this.outputBarEl && (this._liveSampleHost !== "main" || panel.parentElement !== this.mainBody)) {
            this.outputBarEl.insertAdjacentElement("afterend", panel);
            this._liveSampleHost = "main";
        }
    }

    updateLiveSamplePanel() {
        const panel = this.liveSampleEl;
        if (!panel) return;
        const show = this.needsLiveSamplePanel();
        this._placeLiveSamplePanel();
        panel.classList.toggle("hidden", !show);
        if (!show) {
            panel.classList.remove("receiving");
            return;
        }
        if (!this._liveSampleB64) {
            this.liveSampleImg?.classList.add("hidden");
            this.liveSampleEmpty?.classList.remove("hidden");
            this.liveSampleBadge?.classList.add("hidden");
            if (this.liveSampleMeta) this.liveSampleMeta.textContent = t("liveSample.idleHint");
        }
    }

    clearLiveSamplePreview() {
        this._liveSampleB64 = "";
        this._liveSampleStep = null;
        this._liveSampleTotal = null;
        this._liveSampleSeg = null;
        this.liveSampleEl?.classList.remove("receiving");
        if (this.liveSampleImg) {
            this.liveSampleImg.removeAttribute("src");
            this.liveSampleImg.classList.add("hidden");
        }
        this.liveSampleEmpty?.classList.remove("hidden");
        this.liveSampleBadge?.classList.add("hidden");
        if (this.liveSampleMeta) this.liveSampleMeta.textContent = t("liveSample.idleHint");
    }

    setLiveSamplePreview(detail = {}) {
        if (!this.needsLiveSamplePanel()) return;
        const b64 = detail.image_b64 || detail.imageB64 || "";
        if (!b64) return;
        this._placeLiveSamplePanel();
        this.liveSampleEl?.classList.remove("hidden");
        this._liveSampleB64 = b64;
        this._liveSampleStep = detail.step ?? null;
        this._liveSampleTotal = detail.total_steps ?? detail.totalSteps ?? null;
        this._liveSampleSeg = detail.segment_index ?? detail.segmentIndex ?? null;

        const src = b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
        if (this.liveSampleImg) {
            this.liveSampleImg.src = src;
            this.liveSampleImg.classList.remove("hidden");
        }
        this.liveSampleEmpty?.classList.add("hidden");
        this.liveSampleEl?.classList.toggle("receiving", !!detail.live);

        const step = this._liveSampleStep;
        const total = this._liveSampleTotal;
        const seg = this._liveSampleSeg;
        let badge = "";
        if (step && total) badge = t("batch.generatingStep", { step, total });
        else if (detail.live) badge = t("batch.generating");
        if (this.liveSampleBadge) {
            this.liveSampleBadge.textContent = badge;
            this.liveSampleBadge.classList.toggle("hidden", !badge);
        }
        if (this.liveSampleMeta) {
            const unit = this.isFl2vMode?.() ? t("unit.shot") : t("unit.segment");
            const segLabel = (seg != null && seg !== "")
                ? t("liveSample.segmentHint", { unit, n: Number(seg) + 1 })
                : "";
            this.liveSampleMeta.textContent = detail.live
                ? (segLabel || t("liveSample.sampling"))
                : (segLabel || t("liveSample.done"));
        }
    }

    /** Keep timeline / 素材组 selection on the segment the run is currently on. */
    _followRunSelection(timelineSeg1Based) {
        const segs = this.timeline?.segments || [];
        if (!segs.length) return;
        const idx = clamp(Math.round(Number(timelineSeg1Based) || 1) - 1, 0, segs.length - 1);
        if (this.selectedIndex === idx) return;
        this.selectedIndex = idx;
        this.updateSelectionUI();
        const seg = segs[idx];
        if (seg && Number.isFinite(seg.start) && !this.isPlaying) {
            const f = Math.max(0, Math.round(Number(seg.start) || 0));
            this.currentFrame = f;
            if (this.seekBar) this.seekBar.value = f;
            this._syncStagePreview?.(f, { force: true });
        }
        this.scheduleRender();
    }

    setRunProgress(detail) {
        if (!this.runStatusEl) return;
        const timelineTotal = this.timeline?.segments?.length || 0;
        const runTotal = Math.max(detail.segment_total || this.getRunProgressSegmentTotal(), 1);
        const runSeg = Math.max(1, detail.segment || 1);
        const timelineSeg = detail.timeline_segment ?? runSeg;
        const partialRun = !!detail.partial_run
            || (this.isRunSelectEnabled?.() && runTotal < timelineTotal);
        const phaseLabel = detail.phase_label || detail.phase || t("run.phase.default");
        const overallPct = detail.overall_max > 0
            ? Math.round((100 * detail.overall_value) / detail.overall_max)
            : 0;
        const phasePct = detail.phase_max > 0
            ? Math.round((100 * detail.phase_value) / detail.phase_max)
            : 0;
        const remain = Math.max(0, runTotal - runSeg);

        if (detail.phase === "finish") {
            this.runStatusEl.className = "bd-run-status done";
            this.runTitleEl.textContent = t("run.titleDone");
            this.runDetailEl.textContent = runTotal
                ? (this.isImageBatch()
                    ? (isVideoBatchTask(this.getTaskKey())
                        ? t("run.detailDoneVideos", { n: runTotal })
                        : t("run.detailDoneImages", { n: runTotal }))
                    : (partialRun
                        ? t("run.detailDoneSegmentsPartial", { n: runTotal })
                        : t("run.detailDoneSegments", { n: runTotal })))
                : t("run.detailDoneGeneric");
            this.runOverallEl.style.width = "100%";
            this.runPhaseEl.style.width = "100%";
            this._runHighlightSeg = -1;
            this._runProgressSegKey = null;
            this._followRunSelection(timelineSeg);
            this.updateRunSelectUI();
            if (this.isImageBatch()) this.renderImageBatchGroups();
            else this.scheduleRender();
            return;
        }

        this.runStatusEl.className = "bd-run-status active";
        // Hide the pre-run "将运行 N 段" chip while progress is live — it sits
        // under the title in the same green accent and reads as a layout glitch.
        this.runSelectBar?.classList.add("hidden");
        this._runHighlightSeg = timelineSeg - 1;
        this._followRunSelection(timelineSeg);
        let title;
        if (detail.phase === "plan") {
            title = runTotal > 1 ? t("run.titlePlanning", { n: runTotal, phase: phaseLabel }) : phaseLabel;
        } else if (this.isImageBatch()) {
            // Partial run: show timeline card number (e.g. group 4), not compact run order.
            title = partialRun
                ? t("run.titleBatchGroupPartial", {
                    timeline: timelineSeg, i: runSeg, n: runTotal, phase: phaseLabel,
                })
                : t("run.titleBatchGroup", { i: runSeg, n: runTotal, phase: phaseLabel });
        } else if (partialRun) {
            title = t("run.titleSegmentPartial", { timeline: timelineSeg, i: runSeg, n: runTotal, phase: phaseLabel });
        } else {
            title = t("run.titleSegment", { i: runSeg, n: runTotal, phase: phaseLabel });
        }
        if (phasePct > 0 && detail.phase !== "plan") {
            title += ` · ${phasePct}%`;
        }
        this.runTitleEl.textContent = title;
        const parts = [];
        if (detail.frames_label) parts.push(detail.frames_label);
        if (detail.task_key) parts.push(detail.task_key);
        parts.push(t("run.detailOverall", { pct: overallPct }));
        if (runTotal > 1) {
            parts.push(this.isImageBatch()
                ? t("run.detailRemainingGroups", { n: remain })
                : t("run.detailRemainingSegments", { n: remain }));
        }
        if (partialRun && timelineTotal > runTotal) {
            parts.push(t("run.detailTimelineTotal", { n: timelineTotal }));
        }
        this.runDetailEl.textContent = parts.join(" · ");
        this.runOverallEl.style.width = `${overallPct}%`;
        this.runPhaseEl.style.width = `${phasePct}%`;
        // Do NOT syncDirectorNodeSize / full batch rebuild every tick — that was the
        // cross-mode (t2v/i2v/r2v/…) infinite-height feedback loop. Status has a fixed
        // min-height; live previews patch in place via minimax_director_opt_preview.
        const segKey = `${timelineSeg}|${detail.phase}|${runSeg}`;
        const segChanged = this._runProgressSegKey !== segKey;
        this._runProgressSegKey = segKey;
        if (this.isImageBatch()) {
            this._syncBatchRunHighlight();
            if (segChanged) healOversizedDirectorNode(this.node, this);
        } else if (segChanged) {
            this.scheduleRender();
            healOversizedDirectorNode(this.node, this);
        }
    }

    clearRunProgress(title, detail) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status idle";
        this.runTitleEl.textContent = title || t("run.titleIdle");
        this.runDetailEl.textContent = detail || t("run.detailIdle");
        this.runOverallEl.style.width = "0%";
        this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this._runProgressSegKey = null;
        this.updateRunSelectUI();
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    }

    setRunError(message) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status error";
        this.runTitleEl.textContent = t("run.titleError");
        this.runDetailEl.textContent = message || t("run.detailError");
        if (this.runOverallEl) this.runOverallEl.style.width = "0%";
        if (this.runPhaseEl) this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this._runProgressSegKey = null;
        this.updateRunSelectUI();
        this.scheduleRender();
    }

    _stopPlay() {
        this.isPlaying = false;
        this._playHandoff = false;
        this._nativePlayFailed = false;
        this._pauseSettling = true;
        cancelAnimationFrame(this._playRaf);
        this._playRaf = null;
        this.stageVideo?.pause();
        this.root.querySelector('[data-a="play"]').textContent = "▶";
        this._resizeObserver?.disconnect();

        const w = this._playCanvasWidth;
        this._releasePlayLayoutLock();

        if (w) this._drawTimelineCanvas(w);
        this._updateTimelineDom({ skipSeek: true });
        this._syncStagePreview(this.currentFrame, { force: true });

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (+this.seekBar.value !== this.currentFrame) {
                    this.seekBar.value = this.currentFrame;
                }
                this._observeViewportResize();
                const drawW = this._measureDrawWidth() || this.viewport?.clientWidth || w;
                if (drawW) this._drawTimelineCanvas(drawW);
                this._syncStagePreview(this.currentFrame, { force: true });
                this._pauseSettling = false;
            });
        });
    }

    async _beginNativePlay() {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.currentFrame >= total) this.currentFrame = 0;
        await this._ensureStageReadyForFrame(this.currentFrame);
        if (!this.isPlaying) return;
        const v = this.stageVideo;
        if (!v) return;
        try {
            await v.play();
        } catch {
            // Native play blocked/failed — keep isPlaying but drive via frame clock.
            this._nativePlayFailed = true;
        }
    }

    async _advanceNativePlayToNextClipOrEnd() {
        if (this._playHandoff || !this.isPlaying) return;
        this._playHandoff = true;
        try {
            const total = this.getTotalFrames();
            const range = this._logicalRangeForClip(this._stageClipIndex);
            const next = range.end < total ? range.end : -1;
            if (next >= 0) {
                this.currentFrame = next;
                await this._beginNativePlay();
                return;
            }
            if (this.isLooping) {
                this.currentFrame = 0;
                await this._beginNativePlay();
                return;
            }
            this.currentFrame = Math.max(0, total - 1);
            this._stopPlay();
        } finally {
            this._playHandoff = false;
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this._stopPlay();
            return;
        }
        const total = this.getTotalFrames();
        if (total < 1) return;

        this.isPlaying = true;
        this._nativePlayFailed = false;
        this.root.querySelector('[data-a="play"]').textContent = "⏸";
        this._lockPlayLayout();
        this._resizeObserver?.disconnect();

        if (this.currentFrame >= total) this.currentFrame = 0;
        this.renderTimelineOnly();
        this._updateTimelineDom();

        const useNative = !this._legacyFrames.length && !!this.stageVideo;
        if (useNative) {
            this._beginNativePlay();
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }

        const tick = () => {
            if (!this.isPlaying) return;
            const fps = Math.max(0.001, this.getFrameRate());

            if (useNative && this.stageVideo && !this._nativePlayFailed) {
                const v = this.stageVideo;
                const clipIndex = this._stageClipIndex >= 0 ? this._stageClipIndex : 0;
                const range = this._logicalRangeForClip(clipIndex);
                const lastLogical = Math.max(range.start, range.end - 1);
                const lastTime = this.getFrameMapEntry(lastLogical).frame / fps;
                const atMappedEnd = v.currentTime >= Math.max(0, lastTime - 0.04);
                const hasTimelineEdits = !!(
                    this.getFrameMap().length
                    || deletedSourceRanges(this.timeline.video || {}).length
                );
                // With deletes, file duration still includes removed tails — trust mapped end.
                const atMediaEnd = !hasTimelineEdits && (
                    v.ended || (v.duration > 0 && v.currentTime >= v.duration - 0.04)
                );

                if ((atMappedEnd || atMediaEnd) && !v.seeking && !this._playHandoff) {
                    this.currentFrame = lastLogical;
                    this.renderTimelineOnly();
                    this._updateTimelineDom();
                    this._advanceNativePlayToNextClipOrEnd();
                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                    return;
                }

                if (!v.paused) {
                    const srcFrame = Math.max(0, Math.round(v.currentTime * fps));
                    let logical = this._logicalFromStageTime(clipIndex, v.currentTime);
                    const jumpToKept = () => {
                        const nextLogical = this._nextLogicalAfterSourceFrame(clipIndex, srcFrame);
                        if (nextLogical >= 0) {
                            const nextSrc = this.getFrameMapEntry(nextLogical).frame;
                            try { v.currentTime = nextSrc / fps; } catch { /* seek race */ }
                            return nextLogical;
                        }
                        return -1;
                    };
                    // Sparse deleted gap, or mid/leading gap vs mapped source.
                    if (logical < 0) {
                        const next = jumpToKept();
                        if (next < 0) {
                            this.currentFrame = lastLogical;
                            this.renderTimelineOnly();
                            this._updateTimelineDom();
                            this._advanceNativePlayToNextClipOrEnd();
                            if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                            return;
                        }
                        logical = next;
                    } else {
                        const mapped = this.getFrameMapEntry(logical);
                        if (mapped.clip === clipIndex && mapped.frame !== srcFrame) {
                            // leading gap (mapped > src) or mid gap (mapped < src)
                            if (mapped.frame > srcFrame || mapped.frame < srcFrame) {
                                const next = mapped.frame > srcFrame ? logical : jumpToKept();
                                if (next < 0) {
                                    this.currentFrame = clamp(logical, 0, total - 1);
                                    this.renderTimelineOnly();
                                    this._updateTimelineDom();
                                    this._advanceNativePlayToNextClipOrEnd();
                                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                                    return;
                                }
                                if (mapped.frame > srcFrame) {
                                    try { v.currentTime = mapped.frame / fps; } catch { /* seek race */ }
                                }
                                logical = next;
                            }
                        }
                    }
                    this.currentFrame = clamp(logical, 0, total - 1);
                    this.renderTimelineOnly();
                    const now = performance.now();
                    if (now - this._lastSeekUiMs > 66) {
                        this._updateTimelineDom();
                        this._lastSeekUiMs = now;
                    }
                }
            } else {
                // Legacy embedded frames (or native play unavailable): step by logical frame.
                this.currentFrame += 1;
                if (this.currentFrame >= total) {
                    if (this.isLooping) this.currentFrame = 0;
                    else {
                        this.currentFrame = total - 1;
                        this._stopPlay();
                        return;
                    }
                }
                this.renderTimelineOnly();
                this._syncStagePreview(this.currentFrame, { force: true });
                const now = performance.now();
                if (now - this._lastSeekUiMs > 80) {
                    this._updateTimelineDom();
                    this._lastSeekUiMs = now;
                }
            }
            this._playRaf = requestAnimationFrame(tick);
        };
        this._playRaf = requestAnimationFrame(tick);
    }
}

Object.assign(MiniMaxH3DirectorOptEditor.prototype, widget_syncMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, output_uiMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, task_uiMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, frame_map_ioMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, segments_modelMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, video_loadMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, media_thumbsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, ref_video_slotMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, task_layoutMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, workspacesMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, modesMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, fieldsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, layout_scheduleMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, external_groupsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, export_pickersMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, run_selectionMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, timeline_payloadMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, eventsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, dom_shellMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, interactionMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, canvasMixin);
