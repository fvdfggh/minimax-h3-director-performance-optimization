/** video_load mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { resolveOutputDimensions } from "../../core/dims.js";
import { buildClipFrameMap, deletedSourceRanges } from "../../core/frame_map.js";
import { THUMB_PREFETCH_BATCH } from "../../core/layout_spec.js";
import { relPath, uid, viewUrl } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { inputViewUrl } from "../urls.js";
import { t } from "../../minimax_i18n.js";
export const video_loadMixin = {
    async _prepareVideoFrames({ fileName, relPath, subfolder, type, statusPrefix, syncNativeFps = true }) {
        this.videoNameEl.textContent = `${statusPrefix}: ${fileName}…`;
        const viewUrl = inputViewUrl(relPath, type || "input");

        let serverProbe = null;
        try {
            serverProbe = await this.probeVideoFile(relPath, subfolder, type);
        } catch (err) {
            console.warn("[MiniMax H3Director] video probe failed, using browser estimate:", err);
        }
        const browserMeta = await this.probeVideoMetadata(viewUrl);
        const nativeFps = Number(serverProbe?.native_fps || 0);
        const nativeFrameCount = Number(serverProbe?.frame_count || 0);
        const meta = {
            width: Number(serverProbe?.width || browserMeta.width || 0),
            height: Number(serverProbe?.height || browserMeta.height || 0),
            duration: Number(serverProbe?.duration ?? browserMeta.duration ?? 0),
            nativeFps,
            nativeFrameCount,
            probeMethod: serverProbe?.probe_method || "browser_estimate",
        };

        if (syncNativeFps && nativeFps > 0) {
            this.syncFrameRateUI(nativeFps);
        }

        const fps = this.getFrameRate();
        const totalFrames = Math.max(
            1,
            Math.round(meta.duration * fps) || nativeFrameCount,
        );

        const store = resolveOutputDimensions(meta.width, meta.height, this.timeline.output || { mode: "long_edge", longEdge: 864 }, {
            refMaxSize: this.refMaxWidget?.value,
        });

        return { fileName, relPath, subfolder, type, meta, totalFrames, store, viewUrl };
    },
    async probeVideoFile(relPath, subfolder = "", type = "input") {
        const resp = await api.fetchApi("/minimax/director_opt/probe_video", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ videoFile: relPath, subfolder, type: type || "input" }),
        });
        if (!resp.ok) {
            throw new Error(await resp.text());
        }
        return resp.json();
    },
    _buildClipRecord({ fileName, relPath, subfolder, type, meta, totalFrames, store }) {
        return {
            id: uid(),
            fileName,
            videoFile: relPath,
            subfolder: subfolder || "",
            type: type || "input",
            width: meta.width,
            height: meta.height,
            duration: meta.duration,
            nativeFps: meta.nativeFps || null,
            nativeFrameCount: meta.nativeFrameCount || null,
            sourceFrameCount: totalFrames,
            storageWidth: store.width,
            storageHeight: store.height,
        };
    },
    _syncPrimaryVideoFromClips(frameMap) {
        const clips = this.getVideoClips();
        const primary = clips[0] || {};
        const prev = this.timeline.video || {};
        const map = Array.isArray(frameMap) ? frameMap : (prev.frameMap || []);
        this.timeline.video = {
            ...prev,
            ...primary,
            // Keep path/type from the clip record, but never drop timeline edits.
            fileName: primary.fileName || prev.fileName || "",
            videoFile: primary.videoFile || prev.videoFile || "",
            subfolder: primary.subfolder ?? prev.subfolder ?? "",
            type: primary.type || prev.type || "input",
            frames: prev.frames || [],
            frameMap: map,
            // Explicit map already encodes deletes; sparse mode keeps ranges.
            deletedSourceRanges: map.length ? [] : (prev.deletedSourceRanges || []),
            sourceFrameCount: prev.sourceFrameCount || primary.sourceFrameCount || map.length || 0,
        };
        if (map.length) this.timeline.totalFrames = map.length;
    },
    async _applyLoadedVideo({ fileName, relPath, subfolder, type, statusPrefix }) {
        const prep = await this._prepareVideoFrames({ fileName, relPath, subfolder, type, statusPrefix });
        const { totalFrames, store, viewUrl } = prep;

        this._storageWidth = store.width;
        this._storageHeight = store.height;
        const clip = this._buildClipRecord(prep);

        this.timeline.videoClips = [clip];
        this.setSparseVideoFrames(totalFrames);
        this._syncPrimaryVideoFromClips([]);
        this._setSingleSegment(totalFrames);

        this._clearPreviewVideos(true);
        this._previewVideo = this._getPreviewVideoForClip(0);
        if (this._previewVideo && viewUrl) this._previewVideo.src = viewUrl;

        // Force stage to drop any previous media before binding the new clip.
        this._stageClipIndex = -1;
        if (this.stageVideo) {
            this.stageVideo.pause();
            this.stageVideo.removeAttribute("src");
            this.stageVideo.load();
        }
        this.currentFrame = 0;

        if (this.totalFramesWidget) this.totalFramesWidget.value = totalFrames;
        this.syncOutputUIFromTimeline();
        this.updateVideoNameLabel();
        this._flushPendingThumbDrops();
        this._prefetchSegmentThumbs(0, Math.min(totalFrames, THUMB_PREFETCH_BATCH * 4));
        this.updateStageVisibility();
        this._syncStagePreview(this.currentFrame, { force: true });
        this.commit(false, { syncTimeline: true });
    },
    async _applyAppendedVideo({ fileName, relPath, subfolder, type, statusPrefix }) {
        const prep = await this._prepareVideoFrames({
            fileName, relPath, subfolder, type, statusPrefix,
            syncNativeFps: false,
        });
        const { totalFrames, store } = prep;

        this._ensureVideoClipsArray();
        const clipIndex = this.timeline.videoClips.length;
        const clip = this._buildClipRecord(prep);
        this.timeline.videoClips.push(clip);

        const prevTotal = this.getTotalFrames();
        if (!this.getFrameMap().length && prevTotal > 0) {
            this.materializeFrameMap();
        }
        const newEntries = buildClipFrameMap(clipIndex, totalFrames);
        const map = [...this.getFrameMap(), ...newEntries];
        this.setFrameMap(map);
        this.timeline.totalFrames = map.length;
        this._syncPrimaryVideoFromClips(map);

        this._getPreviewVideoForClip(clipIndex);

        this.timeline.segments.push({
            id: uid(),
            start: prevTotal,
            length: totalFrames,
            prompt: "",
            taskType: "",
            refs: [],
            referenceVideo: {},
            videoClipId: clip.id,
        });

        if (this.totalFramesWidget) this.totalFramesWidget.value = map.length;
        this.selectedIndex = this.timeline.segments.length - 1;
        this.currentFrame = prevTotal;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, map.length - 1);
            this.seekBar.value = this.currentFrame;
        }

        this.normalizeSegments();
        this.syncOutputUIFromTimeline();
        this.updateVideoNameLabel();
        this._prefetchSegmentThumbs(prevTotal, Math.min(prevTotal + totalFrames, prevTotal + THUMB_PREFETCH_BATCH * 4));
        this.updateStageVisibility();
        this.commit(false, { syncTimeline: true });
    },
    async probeVideoMetadata(url) {
        const video = document.createElement("video");
        video.src = url;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        await new Promise((res, rej) => {
            video.onloadedmetadata = () => res();
            video.onerror = () => rej(new Error(t("upload.metaReadFailed")));
        });
        return {
            width: video.videoWidth || 0,
            height: video.videoHeight || 0,
            duration: video.duration || 0,
        };
    }
};
