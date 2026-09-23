/** frame_map_io mixin for the Director editor (frame_map_io).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { coerceTimelineFps } from "../../core/dims.js";
import { buildIdentityFrameMap, deletedSourceRanges, logicalToSourceFrame, normalizeFrameMapEntry } from "../../core/frame_map.js";
import { THUMB_PREFETCH_BATCH } from "../../core/layout_spec.js";
import { clamp } from "../../core/utils.js";
import { getFl2vTotalDurationSec, getFl2vVisualFrames } from "../../minimax_fl2v.js";
import { preferredDurationSecFromFrames, roundDurationSec, sumFrameCounts } from "../../minimax_gen_timeline.js";
export const frame_map_ioMixin = {
    materializeFrameMap() {
        const total = this.getTotalFrames();
        const video = this.timeline.video || {};
        if (video.frameMap?.length === total) return;
        const map = [];
        for (let i = 0; i < total; i++) map.push(this.getFrameMapEntry(i));
        video.frameMap = map;
        video.deletedSourceRanges = [];
        this.timeline.video = video;
        this.timeline.totalFrames = total;
    },
    getFrameMap() {
        const v = this.timeline?.video || {};
        if (v.frameMap?.length) return v.frameMap;
        if (this._legacyFrames.length) return buildIdentityFrameMap(this._legacyFrames.length);
        if (v.frames?.length) return buildIdentityFrameMap(v.frames.length);
        return [];
    },
    setFrameMap(map) {
        this.timeline.video = this.timeline.video || {};
        this.timeline.video.frameMap = map;
        if (map.length) {
            this.timeline.totalFrames = map.length;
            this.timeline.video.deletedSourceRanges = [];
        }
    },
    setSparseVideoFrames(totalFrames) {
        this.timeline.video = this.timeline.video || {};
        this.timeline.video.frameMap = [];
        this.timeline.video.sourceFrameCount = totalFrames;
        this.timeline.video.deletedSourceRanges = [];
        this.timeline.totalFrames = totalFrames;
    },
    logicalToSourceFrame(logical) {
        return logicalToSourceFrame(logical, this.timeline.video || {});
    },
    getTotalFrames() {
        // fl2v: visual canvas may be longer than the sampling window (overflow = dashed).
        if (this.isFl2vMode()) return getFl2vVisualFrames(this);
        if (this.isImageBatch() || this.isGenMode()) {
            // t2v/i2v: never use drag preview for totals (inputs are the source of truth).
            // r2v may temporarily use _previewSegments while resizing on the timeline.
            if (this.usesBatchTimeline() && this._previewSegments) {
                return sumFrameCounts(this._previewSegments);
            }
            return sumFrameCounts(this.timeline.segments);
        }
        const mapLen = this.timeline?.video?.frameMap?.length || 0;
        if (mapLen > 0) return mapLen;
        // Sparse deletes: sourceFrameCount − ranges beats a stale totalFrames.
        const src = parseInt(this.timeline?.video?.sourceFrameCount || 0, 10);
        if (src > 0) {
            const removed = deletedSourceRanges(this.timeline.video).reduce((s, [a, b]) => s + (b - a), 0);
            return Math.max(0, src - removed);
        }
        const total = Math.max(0, parseInt(this.timeline?.totalFrames || this.totalFramesWidget?.value || 0, 10));
        if (total > 0) return total;
        if (!this.hasVideo()) return 0;
        return 0;
    },
    getMaxExportFrames() {
        const n = parseInt(this.timeline.output?.maxExportFrames ?? 0, 10);
        return Number.isFinite(n) && n > 0 ? n : 0;
    },
    getExportFrameTotal() {
        const total = this.getTotalFrames();
        const cap = this.getMaxExportFrames();
        return cap > 0 ? Math.min(total, cap) : total;
    },
    getFrameRate() {
        return coerceTimelineFps(this.fpsInput?.value ?? this.frameRateWidget?.value ?? this.timeline.frameRate ?? 24);
    },
    syncFrameRateUI(value = null) {
        const fps = coerceTimelineFps(value ?? this.fpsInput?.value ?? this.frameRateWidget?.value ?? this.timeline.frameRate ?? 24);
        this.timeline.frameRate = fps;
        if (this.frameRateWidget) this.frameRateWidget.value = fps;
        if (this.fpsInput) this.fpsInput.value = fps;
        return fps;
    },
    _clipFrameCountAtFps(clip, fps, fallback = 0) {
        const nativeFps = Number(clip?.nativeFps || 0);
        const nativeCount = Number(clip?.nativeFrameCount || 0);
        if (nativeFps > 0 && nativeCount > 0) {
            return Math.max(1, Math.round((nativeCount / nativeFps) * fps));
        }
        const duration = Number(clip?.duration || 0);
        if (duration > 0) return Math.max(1, Math.round(duration * fps));
        return Math.max(1, Math.round(fallback || Number(clip?.sourceFrameCount || 0) || 1));
    },
    _timelineFrameCountAtFps(fps, oldFps = null, oldTotal = null) {
        const nextFps = coerceTimelineFps(fps);
        const prevTotal = Number(oldTotal ?? this.getTotalFrames() ?? 0);
        const prevFps = coerceTimelineFps(oldFps ?? this.timeline.frameRate ?? this.frameRateWidget?.value ?? 24);
        // When user changes timeline FPS, preserve wall-clock duration: T = N/fps → N' = T * fps'.
        if (prevTotal > 0 && oldFps != null && Math.abs(prevFps - nextFps) >= 0.001) {
            return Math.max(1, Math.round(prevTotal * nextFps / prevFps));
        }
        const clips = this.getVideoClips();
        if (clips.length && clips.some((c) => Number(c.duration || 0) > 0 || Number(c.nativeFrameCount || 0) > 0)) {
            return clips.reduce((sum, clip) => sum + this._clipFrameCountAtFps(clip, nextFps), 0);
        }
        if (prevTotal > 0) {
            return Math.max(1, Math.round(prevTotal * nextFps / Math.max(prevFps, 0.001)));
        }
        return 1;
    },
    _rescaleSegmentsForTotal(oldTotal, newTotal) {
        if (!oldTotal || !newTotal || !this.timeline.segments?.length) {
            this._setSingleSegment(newTotal);
            return;
        }
        const ordered = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        let cursor = 0;
        this.timeline.segments = ordered.map((seg, idx) => {
            const rawStart = idx === 0 ? 0 : Math.round((seg.start / oldTotal) * newTotal);
            const rawEnd = idx === ordered.length - 1
                ? newTotal
                : Math.round(((seg.start + seg.length) / oldTotal) * newTotal);
            const start = clamp(rawStart, cursor, newTotal);
            const end = clamp(rawEnd, start + 1, newTotal);
            cursor = end;
            return {
                ...seg,
                start,
                length: Math.max(1, end - start),
                frameCount: Math.max(1, end - start),
            };
        });
    },
    _syncClipFrameCountsForFps(fps, oldFps = null) {
        const clips = this.getVideoClips();
        if (!clips.length) return;
        const prevFps = coerceTimelineFps(oldFps ?? this.timeline.frameRate ?? 24);
        this.timeline.videoClips = clips.map((clip) => {
            const fallback = Number(clip.sourceFrameCount || 0) * fps / Math.max(prevFps, 0.001);
            return { ...clip, sourceFrameCount: this._clipFrameCountAtFps(clip, fps, fallback) };
        });
    },
    _resampleFrameMapForFps(oldFps, newFps, newTotal) {
        const oldTotal = this.getTotalFrames();
        if (!oldTotal || !newTotal) return [];
        const oldEntries = Array.from({ length: oldTotal }, (_, i) => this.getFrameMapEntry(i));
        const clips = this.getVideoClips();
        const map = [];
        for (let i = 0; i < newTotal; i++) {
            const oldLogical = clamp(Math.round((i / newFps) * oldFps), 0, oldTotal - 1);
            const entry = normalizeFrameMapEntry(oldEntries[oldLogical]);
            const clip = clips[entry.clip] || clips[0] || {};
            const maxFrame = this._clipFrameCountAtFps(clip, newFps) - 1;
            const sourceTime = Number(entry.frame || 0) / Math.max(oldFps, 0.001);
            map.push({
                clip: entry.clip,
                frame: clamp(Math.round(sourceTime * newFps), 0, Math.max(0, maxFrame)),
            });
        }
        return map;
    },
    _resampleTimelineForFrameRate(oldFps, newFps) {
        if (this.isImageBatch() || this.isGenMode() || !this.hasVideo()) return;
        const oldTotal = this.getTotalFrames();
        const newTotal = this._timelineFrameCountAtFps(newFps, oldFps, oldTotal);
        const hasExplicitMap = this.getFrameMap().length > 0;
        const hasSparseDeletes = deletedSourceRanges(this.timeline.video || {}).length > 0;

        if (hasExplicitMap || hasSparseDeletes || this.getVideoClips().length > 1) {
            const newMap = this._resampleFrameMapForFps(oldFps, newFps, newTotal);
            this.setFrameMap(newMap);
            this._syncClipFrameCountsForFps(newFps, oldFps);
            this._syncPrimaryVideoFromClips(newMap);
        } else {
            this._syncClipFrameCountsForFps(newFps, oldFps);
            this.setSparseVideoFrames(newTotal);
            this._syncPrimaryVideoFromClips([]);
        }

        this._rescaleSegmentsForTotal(oldTotal, newTotal);
        this.currentFrame = clamp(Math.round((this.currentFrame / Math.max(oldTotal, 1)) * newTotal), 0, Math.max(0, newTotal - 1));
        if (this.totalFramesWidget) this.totalFramesWidget.value = newTotal;
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, newTotal - 1);
            this.seekBar.value = this.currentFrame;
        }
        this._prefetchSegmentThumbs(0, Math.min(newTotal, THUMB_PREFETCH_BATCH * 4));
    },
    onFrameRateChanged(value) {
        const oldFps = coerceTimelineFps(this.timeline.frameRate ?? this.frameRateWidget?.value ?? 24);
        const newFps = this.syncFrameRateUI(value);
        if (Math.abs(oldFps - newFps) < 0.001) {
            this.commit(false, { syncTimeline: true });
            return;
        }
        this._resampleTimelineForFrameRate(oldFps, newFps);
        this.updateVideoNameLabel();
        this.updateOutputPreview();
        this.scheduleRender();
        this.commit(false, { syncTimeline: true });
    },
    getTimelineDurationSec() {
        if (this.isFl2vMode()) return getFl2vTotalDurationSec(this);
        const total = this.getTotalFrames();
        const fps = this.getFrameRate();
        return total / Math.max(fps, 0.001);
    },
    /** User-facing seconds for ruler ticks (batch 秒数, not MiniMax-aligned play length). */
    getRulerDurationSec() {
        if (this.isFl2vMode()) return Math.max(0.001, getFl2vTotalDurationSec(this));
        if (this.usesBatchTimeline()) {
            const segs = this._previewSegments || this.timeline.segments || [];
            let sec = 0;
            const dragging = !!this._previewSegments;
            for (const seg of segs) {
                const fc = Math.max(0, parseInt(seg.frameCount ?? seg.length, 10) || 0);
                const raw = Number(seg.durationSec);
                if (dragging) {
                    sec += preferredDurationSecFromFrames(fc, 24);
                } else if (Number.isFinite(raw) && raw > 0) {
                    sec += raw;
                } else if (fc > 0) {
                    sec += preferredDurationSecFromFrames(fc, 24);
                }
            }
            return Math.max(0.001, roundDurationSec(sec));
        }
        return Math.max(0.001, this.getTimelineDurationSec());
    }
};
