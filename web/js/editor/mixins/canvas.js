/** canvas mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { CLIP_SEGMENT_COLORS, MAX_THUMBS_PER_SEGMENT, RULER_H, RUN_CHECK_SIZE, SEG_LABEL_H, TRACK_H, TRACK_Y } from "../../core/layout_spec.js";
import { formatRulerTime, pickRulerMajorStepSec, pickRulerMinorStepSec } from "../../core/ruler.js";
import { clamp } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { refViewUrl } from "../urls.js";
import { drawFl2vSegmentThumbnails, getFl2vSampleFrames } from "../../minimax_fl2v.js";
import { t } from "../../minimax_i18n.js";
import { listCommonImageRefs } from "../../minimax_image_batch.js";
export const canvasMixin = {
    getFrameImage(frameIndex) {
        return this._thumbCache.get(this._frameThumbKey(frameIndex)) || null;
    },
    drawSegmentThumbnails(ctx, seg, startX, pxWidth, y0, h, index = -1) {
        if (this.isFl2vMode()) {
            drawFl2vSegmentThumbnails(this, ctx, seg, startX, pxWidth, y0, h);
            return;
        }
        ctx.save();
        ctx.beginPath();
        ctx.rect(startX, y0 + 1, pxWidth, h - 2);
        ctx.clip();

        if (this.isR2vBatch()) {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            const refs = [...(seg.refs || [])].sort(
                (a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0),
            );
            const commonRefs = listCommonImageRefs(this);
            const imgFile = refs.find((r) => r?.imageFile)?.imageFile
                || commonRefs.find((r) => r?.imageFile)?.imageFile
                || "";
            const previewB64 = seg.previewB64 || (Array.isArray(seg.previewFrames) ? seg.previewFrames[0] : "");
            const vidRef = [...(seg.refVideos || [])]
                .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0))
                .find((r) => r?.videoFile || r?.previewImageFile || r?.previewImageUrl || r?.linked);
            const vidPath = vidRef?.videoFile || "";
            const vidType = vidRef?.type || "input";
            const posterFile = vidRef?.previewImageFile || "";
            const posterUrl = vidRef?.previewImageUrl || "";
            let cacheKey = "";
            let srcKind = "";
            if (imgFile) {
                cacheKey = `r2v:${imgFile}`;
                srcKind = "image";
            } else if (previewB64) {
                cacheKey = `r2v-prev:${seg.id || startX}`;
                srcKind = "preview";
            } else if (vidPath) {
                cacheKey = `r2v-vid:${vidType}:${vidPath}`;
                srcKind = "video";
            } else if (posterFile || posterUrl) {
                cacheKey = `r2v-vid-poster:${posterFile || posterUrl}`;
                srcKind = "poster";
            }
            const drawCached = (img) => {
                if (!img?.naturalWidth && !img?.width) return false;
                const natW = img.naturalWidth || img.width;
                const natH = Math.max(1, img.naturalHeight || img.height);
                const ratio = natW / natH;
                let dw = pxWidth - 4;
                let dh = dw / ratio;
                if (dh > h - 4) {
                    dh = h - 4;
                    dw = dh * ratio;
                }
                ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                return true;
            };
            if (cacheKey) {
                const img = this._thumbCache.get(cacheKey);
                if (!drawCached(img)) {
                    if (srcKind === "video") {
                        this._queueR2vVideoThumb(cacheKey, vidPath, vidType);
                    } else if (!this._thumbPending.has(cacheKey)) {
                        this._thumbPending.add(cacheKey);
                        const el = new Image();
                        el.crossOrigin = "anonymous";
                        el.onload = () => {
                            this._thumbCache.set(cacheKey, el);
                            this._thumbPending.delete(cacheKey);
                            this.scheduleRender();
                        };
                        el.onerror = () => this._thumbPending.delete(cacheKey);
                        if (srcKind === "image") el.src = refViewUrl(imgFile);
                        else if (srcKind === "poster") {
                            el.src = posterUrl || refViewUrl(posterFile);
                        } else {
                            el.src = String(previewB64).startsWith("data:")
                                ? previewB64
                                : `data:image/png;base64,${previewB64}`;
                        }
                    }
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(t("canvas.uploadR2vMedia"), startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        if (this.getTaskKey() === "i2v") {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            const imgFile = seg.genImage?.imageFile || seg.imageFile || "";
            const previewB64 = seg.previewB64 || (Array.isArray(seg.previewFrames) ? seg.previewFrames[0] : "");
            const cacheKey = imgFile
                ? `i2v:${imgFile}`
                : (previewB64 ? `i2v-prev:${seg.id || startX}` : "");
            const drawCached = (img) => {
                if (!img?.naturalWidth && !img?.width) return false;
                const natW = img.naturalWidth || img.width;
                const natH = Math.max(1, img.naturalHeight || img.height);
                const ratio = natW / natH;
                let dw = pxWidth - 4;
                let dh = dw / ratio;
                if (dh > h - 4) {
                    dh = h - 4;
                    dw = dh * ratio;
                }
                ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                return true;
            };
            if (cacheKey) {
                const img = this._thumbCache.get(cacheKey);
                if (!drawCached(img) && !this._thumbPending.has(cacheKey)) {
                    this._thumbPending.add(cacheKey);
                    const el = new Image();
                    el.crossOrigin = "anonymous";
                    el.onload = () => {
                        this._thumbCache.set(cacheKey, el);
                        this._thumbPending.delete(cacheKey);
                        this.scheduleRender();
                    };
                    el.onerror = () => this._thumbPending.delete(cacheKey);
                    if (imgFile) el.src = refViewUrl(imgFile);
                    else {
                        el.src = String(previewB64).startsWith("data:")
                            ? previewB64
                            : `data:image/png;base64,${previewB64}`;
                    }
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(t("batch.uploadSource"), startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        if (this.getTaskKey() === "t2v") {
            const segs = this._previewSegments || this.timeline.segments || [];
            const idx = (Number.isFinite(index) && index >= 0)
                ? index
                : Math.max(0, segs.indexOf(seg));
            const fills = ["#2a2618", "#182028", "#182818", "#281820"];
            const accents = ["#d4a017", "#66aaff", "#4fff8f", "#ff66aa"];
            ctx.fillStyle = fills[idx % fills.length];
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            ctx.fillStyle = accents[idx % accents.length];
            ctx.fillRect(startX, y0 + 1, Math.min(4, Math.max(2, pxWidth * 0.04)), h - 2);
            if (pxWidth > 40) {
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                const title = t("batch.groupTitle.prompt", { n: idx + 1 });
                const sec = Number(seg.durationSec);
                const sub = Number.isFinite(sec) && sec > 0
                    ? `${sec.toFixed(1)}s`
                    : `${seg.length || 0}f`;
                ctx.fillStyle = "#e8e8e8";
                ctx.font = "bold 12px sans-serif";
                ctx.fillText(title, startX + pxWidth / 2, y0 + h * 0.38);
                ctx.fillStyle = "#9aa";
                ctx.font = "11px sans-serif";
                ctx.fillText(sub, startX + pxWidth / 2, y0 + h * 0.52);
            }
            ctx.restore();
            return;
        }

        if (this.isGenBlank()) {
            ctx.fillStyle = "#0d0d0d";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(startX + 2, y0 + 4, pxWidth - 4, h - 8);
            ctx.setLineDash([]);
            ctx.fillStyle = "#888";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const fc = seg.frameCount ?? seg.length;
            ctx.fillText(`${fc}f`, startX + pxWidth / 2, y0 + h / 2 - 6);
            ctx.fillStyle = "#555";
            ctx.font = "10px sans-serif";
            ctx.fillText(t("canvas.blankCanvas"), startX + pxWidth / 2, y0 + h / 2 + 8);
            ctx.restore();
            return;
        }

        if (this.isGenImage()) {
            const imgFile = this.isGlobalMode()
                ? this.timeline.global?.genImage?.imageFile
                : (seg.genImage?.imageFile || "");
            ctx.fillStyle = "#111";
            ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
            if (imgFile) {
                const cacheKey = `gen:${imgFile}`;
                let img = this._thumbCache.get(cacheKey);
                if (img?.naturalWidth) {
                    const ratio = img.naturalWidth / img.naturalHeight;
                    let dw = pxWidth - 4, dh = dw / ratio;
                    if (dh > h - 4) { dh = h - 4; dw = dh * ratio; }
                    ctx.drawImage(img, startX + (pxWidth - dw) / 2, y0 + (h - dh) / 2, dw, dh);
                } else if (!this._thumbPending.has(cacheKey)) {
                    this._thumbPending.add(cacheKey);
                    const el = new Image();
                    el.crossOrigin = "anonymous";
                    el.onload = () => {
                        this._thumbCache.set(cacheKey, el);
                        this._thumbPending.delete(cacheKey);
                        this.scheduleRender();
                    };
                    el.onerror = () => this._thumbPending.delete(cacheKey);
                    el.src = refViewUrl(imgFile);
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(t("canvas.uploadSourceImage"), startX + pxWidth / 2, y0 + h / 2);
            }
            ctx.restore();
            return;
        }

        ctx.fillStyle = "#000";
        ctx.fillRect(startX, y0 + 1, pxWidth, h - 2);
        if (!this.hasVideo()) {
            ctx.fillStyle = "#666";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
                this.isFl2vMode() ? t("canvas.clickAddShot") : t("canvas.clickUploadVideo"),
                startX + pxWidth / 2,
                y0 + h / 2,
            );
            ctx.restore();
            return;
        }
        const thumbW = Math.max(32, pxWidth / Math.max(1, Math.min(MAX_THUMBS_PER_SEGMENT, Math.ceil(seg.length / 4))));
        const step = Math.max(1, Math.floor(seg.length / Math.max(1, Math.ceil(pxWidth / thumbW))));
        let drawn = 0;
        for (let f = seg.start; f < seg.start + seg.length && drawn < MAX_THUMBS_PER_SEGMENT; f += step, drawn++) {
            this._queueThumbPrefetch(f);
            const img = this.getFrameImage(f);
            const tx = startX + ((f - seg.start) / seg.length) * pxWidth;
            if (img?.naturalWidth) {
                const ratio = img.naturalWidth / img.naturalHeight;
                let dw = thumbW, dh = thumbW / ratio;
                if (dh > h - 2) { dh = h - 2; dw = dh * ratio; }
                ctx.drawImage(img, tx, y0 + (h - dh) / 2, dw, dh);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(tx, y0 + 2, Math.max(8, thumbW * 0.6), h - 4);
            }
        }
        ctx.restore();
    },
    _drawSegmentRunCheck(x, y, enabled) {
        const ctx = this.ctx;
        const s = RUN_CHECK_SIZE;
        ctx.save();
        // Opaque plate so the control never blends into timeline chrome.
        ctx.fillStyle = "#0e0e0e";
        ctx.fillRect(x - 1, y - 1, s + 2, s + 2);
        ctx.fillStyle = enabled ? "#1a3a2a" : "#1c1c1c";
        ctx.strokeStyle = enabled ? "#4fff8f" : "#888";
        ctx.lineWidth = 1;
        ctx.fillRect(x, y, s, s);
        ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
        if (enabled) {
            ctx.fillStyle = "#4fff8f";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "alphabetic";
            ctx.fillText("✓", x + 2, y + 11);
        }
        ctx.restore();
    },
    _drawReorderInsertMarker(ix) {
        const ctx = this.ctx;
        const y0 = TRACK_Y;
        const y1 = TRACK_Y + TRACK_H;
        ctx.save();
        ctx.strokeStyle = "#4fff8f";
        ctx.fillStyle = "#4fff8f";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(ix, y0);
        ctx.lineTo(ix, y1);
        ctx.stroke();
        // Triangles at top/bottom
        const t = 7;
        ctx.beginPath();
        ctx.moveTo(ix, y0);
        ctx.lineTo(ix - t, y0 - t);
        ctx.lineTo(ix + t, y0 - t);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(ix, y1);
        ctx.lineTo(ix - t, y1 + t);
        ctx.lineTo(ix + t, y1 + t);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    },
    /** Floating ghost card that follows the pointer while reordering clips. */
    _drawReorderGhost(width, segs, fromRank) {
        if (fromRank < 0 || this._drag?.pointerX == null) return;
        const ordered = this._orderedSegmentsWithRank();
        const item = ordered.find((o) => o.visualRank === fromRank);
        if (!item?.seg) return;
        const seg = item.seg;
        const srcW = Math.max(48, this.frameToX(seg.start + seg.length, width) - this.frameToX(seg.start, width));
        const gw = Math.min(140, Math.max(72, srcW * 0.55));
        const gh = TRACK_H * 0.78;
        const gx = this._drag.pointerX - gw / 2;
        const gy = clamp(this._drag.pointerY - gh / 2, TRACK_Y - 8, TRACK_Y + TRACK_H - gh + 8);
        const ctx = this.ctx;
        ctx.save();
        // Drop shadow
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        ctx.fillRect(gx + 4, gy + 5, gw, gh);
        ctx.globalAlpha = 0.95;
        this.drawSegmentThumbnails(ctx, seg, gx, gw, gy, gh, item.arrayIndex);
        ctx.strokeStyle = "#4fff8f";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1);
        ctx.fillStyle = "rgba(20,40,28,0.9)";
        ctx.fillRect(gx + 4, gy + 4, 44, 16);
        ctx.fillStyle = "#4fff8f";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(t("canvas.dragging"), gx + 8, gy + 12);
        ctx.restore();
    },
    drawPromptOverlay(ctx, seg, startX, pxWidth, y0, h) {
        const prompt = this.getDisplayPrompt(seg);
        if (!prompt || pxWidth < 24) return;
        const overlayH = Math.round(h * 0.22);
        const overlayY = y0 + h - overlayH;
        ctx.save();
        ctx.beginPath();
        ctx.rect(startX, overlayY, pxWidth, overlayH);
        ctx.clip();
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.fillRect(startX, overlayY, pxWidth, overlayH);
        ctx.font = `${Math.min(11, overlayH * 0.55)}px sans-serif`;
        ctx.fillStyle = "#e0e3ed";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        let label = prompt;
        const maxW = pxWidth - 10;
        if (ctx.measureText(label).width > maxW) {
            // Estimate truncation first — avoids O(n) measureText on long prompts.
            const fullW = ctx.measureText(label).width;
            const approx = Math.floor(maxW * label.length / Math.max(1, fullW));
            label = label.slice(0, Math.max(0, approx - 2));
            while (label.length > 0 && ctx.measureText(label + "…").width > maxW) label = label.slice(0, -1);
            while (label.length < prompt.length && ctx.measureText(label + prompt[label.length] + "…").width <= maxW) {
                label += prompt[label.length];
            }
            label += "…";
        }
        ctx.fillText(label, startX + pxWidth / 2, overlayY + overlayH / 2);
        ctx.restore();
    },
    render() {
        if (this.isPlaying) {
            this.renderTimelineOnly();
            return;
        }
        const width = this._measureDrawWidth();
        if (!width) {
            // Host not laid out yet — retry after the next layout pass.
            this.scheduleSettleRender();
            return;
        }
        this._drawWidth = width;
        this._drawTimelineCanvas(width);
        this._updateTimelineDom();
        this._syncStagePreview(this.currentFrame);
    },
    renderTimelineOnly() {
        const width = this._measureDrawWidth()
            || this.node?.size?.[0]
            || 0;
        if (!width) return;
        this._drawWidth = width;
        this._drawTimelineCanvas(width);
        this._syncStagePreview(this.currentFrame);
    },
    _drawTimelineCanvas(width) {
        const height = this.canvasHeight;
        const dpr = window.devicePixelRatio || 1;
        const bw = Math.round(width * dpr);
        const bh = Math.round(height * dpr);
        // Keep bitmap ↔ CSS aspect in lockstep. Mixing getBoundingClientRect (graph-zoom
        // transformed) with width:100% clientWidth used to squash/stretch thumbs.
        if (this.canvas.width !== bw || this.canvas.height !== bh) {
            this.canvas.width = bw;
            this.canvas.height = bh;
        }
        if (this.getTimelineZoom() > 1) {
            this.canvas.style.width = `${Math.round(width)}px`;
        } else if (this.canvas.style.width !== "100%") {
            this.canvas.style.width = "100%";
        }
        this.canvas.style.height = `${height}px`;
        this.canvas.style.maxHeight = `${height}px`;
        this.canvas.style.minHeight = `${height}px`;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.ctx.clearRect(0, 0, width, height);

        const total = this.getTotalFrames();
        const segs = this._previewSegments || this.timeline.segments;

        this.ctx.fillStyle = "#252525";
        this.ctx.fillRect(0, 0, width, RULER_H);
        this.ctx.font = "10px sans-serif";
        this.ctx.textAlign = "left";
        this.ctx.textBaseline = "alphabetic";
        const fl2vSampleN = this.isFl2vMode() ? getFl2vSampleFrames(this) : total;
        // Batch/fl2v: ticks follow user 秒数 so 5.0s clips land on "5".
        // v2v: ticks follow play length (frames / fps).
        const durationSec = this.getRulerDurationSec();
        const spanX = this.isFl2vMode() ? this.frameToX(fl2vSampleN, width) : width;
        const pxPerSec = spanX / Math.max(durationSec, 0.001);
        const majorSec = pickRulerMajorStepSec(pxPerSec);
        const minorSec = pickRulerMinorStepSec(majorSec, pxPerSec);
        const secToX = (s) => (s / Math.max(durationSec, 0.001)) * spanX;
        if (minorSec < majorSec) {
            this.ctx.fillStyle = "#5a5a5a";
            const nMinor = Math.floor(durationSec / minorSec + 1e-9);
            for (let i = 0; i <= nMinor; i++) {
                const s = i * minorSec;
                if (s % majorSec === 0) continue;
                this.ctx.fillRect(secToX(s), RULER_H - 4, 1, 4);
            }
        }
        this.ctx.fillStyle = "#aaa";
        const nMajor = Math.floor(durationSec / majorSec + 1e-9);
        for (let i = 0; i <= nMajor; i++) {
            const s = i * majorSec;
            const x = secToX(s);
            this.ctx.fillRect(x, RULER_H - 7, 1, 7);
            const label = formatRulerTime(s);
            const tw = this.ctx.measureText(label).width;
            if (x + 3 + tw > width - 2 && s > 0) {
                if (x - tw - 2 >= 2) this.ctx.fillText(label, x - tw - 2, 11);
            } else {
                this.ctx.fillText(label, x + 3, 11);
            }
        }
        // Sample-window end marker on ruler (overflow hatch drawn after segments).
        if (this.isFl2vMode() && total > fl2vSampleN) {
            const ox = this.frameToX(fl2vSampleN, width);
            this.ctx.save();
            this.ctx.strokeStyle = "rgba(180,180,180,0.75)";
            this.ctx.lineWidth = 1.5;
            this.ctx.setLineDash([6, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(ox + 0.5, 0);
            this.ctx.lineTo(ox + 0.5, RULER_H + SEG_LABEL_H);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            this.ctx.fillStyle = "#999";
            this.ctx.font = "10px sans-serif";
            this.ctx.textAlign = "left";
            if (width - ox > 64) {
                this.ctx.fillText(t("canvas.beyondSampling"), ox + 6, RULER_H - 3);
            }
            this.ctx.restore();
        }

        // Frame-range labels above each segment (1-based inclusive, e.g. 1-10).
        this.ctx.fillStyle = "#1a1a1a";
        this.ctx.fillRect(0, RULER_H, width, SEG_LABEL_H);
        this.ctx.font = "10px sans-serif";
        this.ctx.textBaseline = "middle";
        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const pxW = Math.max(0, x1 - x0);
            if (pxW < 8 || seg.length <= 0) continue;
            const a = seg.start + 1;
            const b = seg.start + seg.length;
            const rangeText = `${a}-${b}`;
            // v2v / r2v / fl2v: emphasize selected segment label (matches card selection).
            const showSegSel = this.usesBatchTimeline() || this.isFl2vMode()
                || !(this.isImageBatch() || this.isGenMode());
            this.ctx.fillStyle = (showSegSel && i === this.selectedIndex) ? "#eee" : "#9a9a9a";
            let draw = rangeText;
            if (this.ctx.measureText(draw).width > pxW - 6) {
                while (draw.length > 1 && this.ctx.measureText(`${draw}…`).width > pxW - 6) {
                    draw = draw.slice(0, -1);
                }
                draw = draw.length < rangeText.length ? `${draw}…` : draw;
            }
            this.ctx.fillText(draw, x0 + 4, RULER_H + SEG_LABEL_H / 2);
        }

        this.ctx.fillStyle = "#111";
        this.ctx.fillRect(0, TRACK_Y, width, TRACK_H);

        if (!segs.length && (this.isFl2vMode() || this.usesBatchTimeline())) {
            this.ctx.fillStyle = "#666";
            this.ctx.font = "12px sans-serif";
            this.ctx.textAlign = "center";
            this.ctx.textBaseline = "middle";
            this.ctx.fillText(
                this.isR2vBatch()
                    ? t("canvas.clickAddRefGroup")
                    : (this.usesBatchTimeline() ? t("canvas.clickAddPromptGroup") : t("canvas.clickAddShot")),
                width / 2,
                TRACK_Y + TRACK_H / 2,
            );
        }

        const clipBounds = this.usesBatchTimeline() ? [] : this.getClipBoundaries();
        if (clipBounds.length) {
            this.ctx.strokeStyle = "rgba(102,170,255,0.55)";
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 4]);
            for (const b of clipBounds) {
                const bx = this.frameToX(b, width);
                this.ctx.beginPath();
                this.ctx.moveTo(bx, TRACK_Y);
                this.ctx.lineTo(bx, TRACK_Y + TRACK_H);
                this.ctx.stroke();
            }
            this.ctx.setLineDash([]);
        }

        const reordering = this._drag?.kind === "reorder";
        const dragFromRank = reordering ? this._drag.fromRank : -1;
        const dropRank = reordering ? this._reorderDropRank : -1;
        // v2v / r2v / fl2v: selection chrome matches card selected border.
        const showSegSel = this.usesBatchTimeline() || this.isFl2vMode()
            || !(this.isImageBatch() || this.isGenMode());

        for (let i = 0; i < segs.length; i++) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const pxW = x1 - x0;
            const sel = showSegSel && i === this.selectedIndex;
            const running = i === this._runHighlightSeg;
            const runOn = this.isSegmentRunEnabled(i);
            const fl2vStart = !this.isFl2vMode() || !!seg.isStartFrame;
            const visualRank = this._visualRankFromArrayIndex(i);
            const isDragSource = reordering && visualRank === dragFromRank;
            const isDropTarget = reordering && dropRank >= 0 && visualRank === dropRank && dropRank !== dragFromRank;
            if (this.isRunSelectEnabled() && this.getRunnableSegmentCount() >= 2 && fl2vStart && !runOn) {
                this.ctx.globalAlpha = 0.32;
            } else if (isDragSource) {
                this.ctx.globalAlpha = 0.28;
            } else if (reordering && !isDropTarget) {
                this.ctx.globalAlpha = 0.55;
            } else if (this.isFl2vMode() && !seg.isStartFrame) {
                this.ctx.globalAlpha = 0.72;
            }
            this.drawSegmentThumbnails(this.ctx, seg, x0, pxW, TRACK_Y, TRACK_H, i);
            if (!this.isFl2vMode() || seg.isStartFrame) {
                this.drawPromptOverlay(this.ctx, seg, x0, pxW, TRACK_Y, TRACK_H);
            }
            const clipIdx = this.usesBatchTimeline() ? i : this.getSegmentClipIndex(seg);
            const clipColor = CLIP_SEGMENT_COLORS[clipIdx % CLIP_SEGMENT_COLORS.length];
            if (isDropTarget) {
                this.ctx.fillStyle = "rgba(79,255,143,0.14)";
                this.ctx.fillRect(x0, TRACK_Y, pxW, TRACK_H);
                this.ctx.strokeStyle = "#4fff8f";
                this.ctx.lineWidth = 3;
                this.ctx.setLineDash([7, 4]);
                this.ctx.strokeRect(x0 + 1, TRACK_Y + 1, pxW - 2, TRACK_H - 2);
                this.ctx.setLineDash([]);
                this.ctx.fillStyle = "rgba(20,40,28,0.92)";
                const label = this.isFl2vMode() ? t("canvas.swapHere") : t("canvas.insertHere");
                this.ctx.font = "bold 11px sans-serif";
                const tw = this.ctx.measureText(label).width + 12;
                this.ctx.fillRect(x0 + (pxW - tw) / 2, TRACK_Y + 8, tw, 18);
                this.ctx.fillStyle = "#4fff8f";
                this.ctx.textAlign = "center";
                this.ctx.textBaseline = "middle";
                this.ctx.fillText(label, x0 + pxW / 2, TRACK_Y + 17);
            } else {
                this.ctx.strokeStyle = running || sel ? "#4fff8f" : clipColor;
                this.ctx.lineWidth = running ? 3 : sel ? 2.5 : 1.5;
                this.ctx.strokeRect(x0 + 0.5, TRACK_Y + 0.5, pxW - 1, TRACK_H - 1);
                if (sel && !running) {
                    this.ctx.fillStyle = "rgba(79,255,143,0.08)";
                    this.ctx.fillRect(x0 + 1, TRACK_Y + 1, Math.max(0, pxW - 2), TRACK_H - 2);
                }
            }
            if (this.isFl2vMode()) {
                // Hatch the portion past the sampling window (不计入采样).
                const sampleN = getFl2vSampleFrames(this);
                const segEnd = seg.start + seg.length;
                if (segEnd > sampleN && seg.start < segEnd) {
                    const ox0 = this.frameToX(Math.max(seg.start, sampleN), width);
                    const ox1 = this.frameToX(segEnd, width);
                    if (ox1 > ox0 + 1) {
                        this.ctx.save();
                        this.ctx.beginPath();
                        this.ctx.rect(ox0, TRACK_Y + 1, ox1 - ox0, TRACK_H - 2);
                        this.ctx.clip();
                        this.ctx.fillStyle = "rgba(0,0,0,0.45)";
                        this.ctx.fillRect(ox0, TRACK_Y + 1, ox1 - ox0, TRACK_H - 2);
                        this.ctx.strokeStyle = "rgba(200,200,200,0.55)";
                        this.ctx.lineWidth = 1;
                        this.ctx.setLineDash([5, 4]);
                        this.ctx.strokeRect(ox0 + 0.5, TRACK_Y + 1.5, Math.max(0, ox1 - ox0 - 1), TRACK_H - 3);
                        this.ctx.setLineDash([]);
                        this.ctx.restore();
                    }
                }
                this._drawFl2vEdgeHandles(segs, i, x0, x1, width);
            } else {
                this.ctx.fillStyle = "#ffcc00";
                this.ctx.fillRect(x0 - 2, TRACK_Y + TRACK_H / 2 - 12, 4, 24);
                this.ctx.fillRect(x1 - 2, TRACK_Y + TRACK_H / 2 - 12, 4, 24);
            }
            this.ctx.globalAlpha = 1;
            // Checkbox on top-left; drawn last so it stays clear on dimmed segments.
            if (
                this.isRunSelectEnabled()
                && this.getRunnableSegmentCount() >= 2
                && pxW >= RUN_CHECK_SIZE + 8
                && (!this.isFl2vMode() || seg.isStartFrame)
            ) {
                const g = this._runCheckGeometry(seg, width);
                this._drawSegmentRunCheck(g.boxX, g.boxY, runOn);
            }
        }

        // fl2v: dashed overlay for the region past the sampling window.
        if (this.isFl2vMode()) {
            const sampleN = getFl2vSampleFrames(this);
            if (total > sampleN) {
                const ox = this.frameToX(sampleN, width);
                this.ctx.save();
                this.ctx.strokeStyle = "rgba(180,180,180,0.7)";
                this.ctx.lineWidth = 1.5;
                this.ctx.setLineDash([6, 5]);
                this.ctx.beginPath();
                this.ctx.moveTo(ox + 0.5, TRACK_Y);
                this.ctx.lineTo(ox + 0.5, TRACK_Y + TRACK_H);
                this.ctx.stroke();
                this.ctx.strokeRect(ox + 1, TRACK_Y + 1, Math.max(0, width - ox - 2), TRACK_H - 2);
                this.ctx.setLineDash([]);
                this.ctx.restore();
            }
        }

        if (reordering) {
            this._drawReorderGhost(width, segs, dragFromRank);
            if (dropRank >= 0 && dropRank !== dragFromRank && !this.isFl2vMode()) {
                const insertFrame = this._getReorderInsertFrame(dropRank, dragFromRank);
                const ix = this.frameToX(insertFrame, width);
                this._drawReorderInsertMarker(ix);
            }
        }

        // Editable split-point markers: click = select only; delete via toolbar button.
        const splitFrames = this.getEditableSplitFrames();
        if (splitFrames.length) {
            for (const frame of splitFrames) {
                const sx = this.frameToX(frame, width);
                const selected = this.selectedSplitFrame === frame;
                this.ctx.strokeStyle = selected ? "#ffe066" : "rgba(80, 220, 255, 0.95)";
                this.ctx.fillStyle = selected ? "#ffe066" : "rgba(80, 220, 255, 0.9)";
                this.ctx.lineWidth = selected ? 3.5 : 2;
                this.ctx.beginPath();
                this.ctx.moveTo(sx, RULER_H + 2);
                this.ctx.lineTo(sx, TRACK_Y + TRACK_H - 2);
                this.ctx.stroke();
                const cy = RULER_H + SEG_LABEL_H / 2;
                const r = selected ? 8 : 6;
                this.ctx.beginPath();
                this.ctx.moveTo(sx, cy - r);
                this.ctx.lineTo(sx + r, cy);
                this.ctx.lineTo(sx, cy + r);
                this.ctx.lineTo(sx - r, cy);
                this.ctx.closePath();
                this.ctx.fill();
                if (selected) {
                    this.ctx.strokeStyle = "#fff";
                    this.ctx.lineWidth = 1.5;
                    this.ctx.stroke();
                    // Halo so selection is obvious on dense timelines.
                    this.ctx.strokeStyle = "rgba(255, 224, 102, 0.55)";
                    this.ctx.lineWidth = 6;
                    this.ctx.beginPath();
                    this.ctx.moveTo(sx, TRACK_Y);
                    this.ctx.lineTo(sx, TRACK_Y + TRACK_H);
                    this.ctx.stroke();
                }
            }
        }

        this._drawContinuityJoints(width, segs);

        const phx = this.frameToX(this.currentFrame, width);
        this.ctx.strokeStyle = "#ff4444";
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(phx, 0);
        this.ctx.lineTo(phx, height);
        this.ctx.stroke();

        const exportCap = this.getMaxExportFrames();
        const exportTotal = this.getExportFrameTotal();
        if (exportCap > 0 && exportTotal < total) {
            const capX = this.frameToX(exportTotal, width);
            this.ctx.fillStyle = "rgba(0,0,0,0.35)";
            this.ctx.fillRect(capX, TRACK_Y, width - capX, TRACK_H);
            this.ctx.strokeStyle = "#66aaff";
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([4, 3]);
            this.ctx.beginPath();
            this.ctx.moveTo(capX, 0);
            this.ctx.lineTo(capX, height);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            this.ctx.fillStyle = "#66aaff";
            this.ctx.font = "10px sans-serif";
            this.ctx.fillText(t("canvas.exportCap", { n: exportTotal }), capX + 4, TRACK_Y + 12);
        }
    },
    _updateTimelineDom({ skipSeek = false } = {}) {
        const segs = this._previewSegments || this.timeline.segments;
        const totalFrames = Math.max(0, this.getTotalFrames());
        const cur = this.formatTime(this.currentFrame);
        const total = this.formatTime(totalFrames);
        if (this.timecodeEl) this.timecodeEl.textContent = `${cur}s`;
        if (this.playerTimecodeEl) this.playerTimecodeEl.textContent = `${cur} / ${total}`;
        if (this.frameTotalEl) this.frameTotalEl.textContent = String(totalFrames);
        if (this.frameInputEl) {
            this.frameInputEl.max = String(Math.max(1, totalFrames));
            // Don't overwrite while the user is typing a target frame.
            if (document.activeElement !== this.frameInputEl) {
                this.frameInputEl.value = String(totalFrames > 0 ? this.currentFrame + 1 : 1);
            }
        }
        if (!skipSeek && this.seekBar && +this.seekBar.value !== this.currentFrame) {
            this.seekBar.value = this.currentFrame;
        }
        if (this.seekBar) this.seekBar.max = Math.max(0, totalFrames - 1);
        if (this.selectedSplitFrame != null && this.getEditableSplitFrames().includes(this.selectedSplitFrame)) {
            if (this.boundsEl) {
                this.boundsEl.textContent = t("split.boundsEditable", { f: this.selectedSplitFrame });
            }
        } else {
            const seg = segs[this.selectedIndex];
            if (seg && this.boundsEl) {
                this.boundsEl.textContent = t("bounds.range", {
                    start: this.formatTime(seg.start),
                    end: this.formatTime(seg.start + seg.length),
                });
            }
        }
        this.updateSplitPointUI();
    }
};
