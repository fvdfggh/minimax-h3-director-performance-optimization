/** ref_video_slot mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { formatProbeFps } from "../../core/ruler.js";
import { UPLOAD_SOFT_LIMIT, formatUploadError, uploadToInput, uploadToInputSmart } from "../../core/upload.js";
import { clamp, relPath, uid, viewUrl } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { refViewUrl, videoRelativePath } from "../urls.js";
import { getFl2vTotalDurationSec } from "../../minimax_fl2v.js";
import { MAX_GEN_FRAMES, defaultFrameCount, durationToClampedMiniMaxFrames, framesToDurationSec, isVideoBatchTask, minFrameCount, preferredDurationSecFromFrames, resolveTaskKey, roundDurationSec, taskUsesReferenceVideo } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const ref_video_slotMixin = {
    renderGenSrcSlot(el, imageFile, label) {
        if (!el) return;
        el.classList.toggle("has-img", !!imageFile);
        if (imageFile) {
            el.innerHTML = `<img src="${refViewUrl(imageFile)}" alt="">`;
        } else {
            el.textContent = label;
        }
    },
    _paintRefVideoSlot(el, nameEl, refBlock) {
        if (!el) return;
        const ref = refBlock || {};
        const has = !!(ref.videoFile || ref.fileName);
        el.classList.toggle("has-img", false);
        el.classList.toggle("has-video", has);
        if (nameEl) {
            if (has) {
                const dur = ref.duration > 0 ? ` · ${ref.duration.toFixed(2)}s` : "";
                const fps = ref.nativeFps > 0 ? ` · ${Math.round(ref.nativeFps)}fps` : "";
                const dim = ref.width && ref.height ? ` · ${ref.width}×${ref.height}` : "";
                nameEl.textContent = `${ref.fileName || ref.videoFile || ""}${dim}${dur}${fps}`;
            } else {
                nameEl.textContent = "";
            }
        }
        if (!has) {
            el.innerHTML = "";
            el.textContent = t("panel.uploadRefVideo");
            el.onclick = () => this.pickReferenceVideoFile();
            return;
        }
        const viewUrl = this.getReferenceVideoViewUrl(ref);
        el.innerHTML = `
            <video class="bd-ref-video-preview" muted playsinline preload="metadata" controls></video>
            <button type="button" class="bd-ref-replace" title="${t("ref.replace")}">${t("ref.replace")}</button>
            <span class="x" title="${t("ref.removeVideo")}">×</span>`;
        el.onclick = null;
        const video = el.querySelector("video");
        if (video && viewUrl) {
            video.src = viewUrl;
            video.addEventListener("click", (e) => e.stopPropagation());
            video.addEventListener("dblclick", (e) => {
                e.stopPropagation();
                if (video.paused) video.play().catch(() => {});
                else video.pause();
            });
        }
        const replaceBtn = el.querySelector(".bd-ref-replace");
        if (replaceBtn) {
            replaceBtn.onclick = (e) => {
                e.stopPropagation();
                this.pickReferenceVideoFile();
            };
        }
        const removeBtn = el.querySelector(".x");
        if (removeBtn) {
            removeBtn.onclick = (e) => {
                e.stopPropagation();
                this.clearReferenceVideo();
            };
        }
    },
    renderRefVideoSlot() {
        if (this.isGlobalMode()) {
            this._stopRefVideoPreviews([this.segRefVideo]);
            this._paintRefVideoSlot(
                this.globalRefVideo,
                this.globalRefVideoNameEl,
                this.timeline.global?.referenceVideo || {},
            );
        } else {
            this._stopRefVideoPreviews([this.globalRefVideo]);
            const seg = this.timeline.segments[this.selectedIndex];
            this._paintRefVideoSlot(this.segRefVideo, this.segRefVideoNameEl, seg?.referenceVideo || {});
        }
    },
    _activeRefVideoTaskKey() {
        if (this.isGlobalMode()) return this.getTaskKey();
        const seg = this.timeline.segments[this.selectedIndex];
        return resolveTaskKey(seg?.taskType || this.timeline.global?.taskType || this.getTaskKey());
    },
    pickReferenceVideoFile() {
        if (!taskUsesReferenceVideo(this._activeRefVideoTaskKey())) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*";
        input.onchange = () => {
            if (input.files?.[0]) this.loadReferenceVideoFile(input.files[0]);
        };
        input.click();
    },
    async pickExistingReferenceVideo() {
        if (!taskUsesReferenceVideo(this._activeRefVideoTaskKey())) return;
        const currentValue = this.getRefVideoTarget()?.referenceVideo?.videoFile || "";
        const picked = await this.chooseVideoInput({
            title: t("mediaPicker.pickReferenceVideo"),
            currentValue,
        });
        if (!picked?.relPath) return;
        const slotEl = this.isGlobalMode() ? this.globalRefVideo : this.segRefVideo;
        const nameEl = this.isGlobalMode() ? this.globalRefVideoNameEl : this.segRefVideoNameEl;
        const status = t("upload.inProgress", { name: picked.fileName || picked.relPath });
        if (slotEl) {
            slotEl.classList.remove("has-img", "has-video");
            slotEl.textContent = status;
        }
        if (nameEl) nameEl.textContent = status;
        try {
            const prep = await this._prepareVideoFrames({
                fileName: picked.fileName || picked.relPath,
                relPath: picked.relPath,
                subfolder: picked.subfolder || "",
                type: picked.type || "input",
                statusPrefix: t("parse.refVideo"),
                syncNativeFps: false,
            });
            this.getRefVideoTarget().referenceVideo = this._buildClipRecord(prep);
            this.renderRefVideoSlot();
            this.commit(false, { syncTimeline: true });
        } catch (err) {
            console.error("[MiniMax H3Director] reference video load failed:", err);
            if (nameEl) nameEl.textContent = t("upload.refVideoFailed", { err: formatUploadError(err) });
            this.renderRefVideoSlot();
        }
    },
    clearReferenceVideo() {
        const target = this.getRefVideoTarget();
        this._stopRefVideoPreviews();
        target.referenceVideo = {};
        this.renderRefVideoSlot();
        this.commit();
    },
    async loadReferenceVideoFile(file) {
        const slotEl = this.isGlobalMode() ? this.globalRefVideo : this.segRefVideo;
        const nameEl = this.isGlobalMode() ? this.globalRefVideoNameEl : this.segRefVideoNameEl;
        const status = t("upload.inProgress", { name: file.name });
        if (slotEl) {
            slotEl.classList.remove("has-img", "has-video");
            slotEl.textContent = status;
        }
        if (nameEl) nameEl.textContent = status;
        try {
            const uploaded = await uploadToInputSmart(file, (frac, cur, total) => {
                const pct = Math.round(frac * 100);
                const mode = file.size > UPLOAD_SOFT_LIMIT ? t("upload.chunkMode") : t("upload.mode");
                if (nameEl) {
                    nameEl.textContent = t("upload.refVideoProgress", {
                        mode, name: file.name, cur, total, pct,
                    });
                }
            });
            const relPath = videoRelativePath(uploaded);
            const prep = await this._prepareVideoFrames({
                fileName: file.name,
                relPath,
                subfolder: uploaded.subfolder || "",
                type: uploaded.type || "input",
                statusPrefix: t("parse.refVideo"),
                syncNativeFps: false,
            });
            this.getRefVideoTarget().referenceVideo = this._buildClipRecord(prep);
            this.renderRefVideoSlot();
            this.commit(false, { syncTimeline: true });
        } catch (err) {
            console.error("[MiniMax H3Director] reference video load failed:", err);
            if (nameEl) nameEl.textContent = t("upload.refVideoFailed", { err: formatUploadError(err) });
            this.renderRefVideoSlot();
        }
    },
    pickGenSrcImage(isGlobal) {
        if (!this.isGenImage()) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = async () => {
            const file = input.files?.[0];
            if (!file) return;
            try {
                const uploaded = await uploadToInput(file);
                const relPath = videoRelativePath(uploaded);
                if (isGlobal) {
                    this.timeline.global = this.timeline.global || { refs: [] };
                    this.timeline.global.genImage = { imageFile: relPath };
                } else {
                    const seg = this.timeline.segments[this.selectedIndex];
                    if (seg) {
                        seg.genImage = { imageFile: relPath };
                        seg.imageFile = relPath;
                    }
                }
                this.commit();
            } catch (err) {
                console.error("[MiniMax H3Director] gen image upload failed:", err);
            }
        };
        input.click();
    },
    onGenDefaultFcChange() {
        const fc = clamp(parseInt(this.genDefaultFc?.value, 10) || 1, minFrameCount(this.getTaskKey()), MAX_GEN_FRAMES);
        if (this.genDefaultFc) this.genDefaultFc.value = fc;
        this.timeline.gen = this.timeline.gen || {};
        this.timeline.gen.defaultFrameCount = fc;
        if (this.timeline.segments.length === 1) {
            this.timeline.segments[0].frameCount = fc;
            this.timeline.segments[0].length = fc;
        }
        this.commit();
    },
    onGenSegFcChange() {
        const seg = this.timeline.segments[this.selectedIndex];
        if (!seg) return;
        const minFc = minFrameCount(this.getTaskKey());
        seg.frameCount = clamp(parseInt(this.genSegFc?.value, 10) || minFc, minFc, MAX_GEN_FRAMES);
        if (this.genSegFc) this.genSegFc.value = seg.frameCount;
        this.commit();
    },
    genSplitAtFrame(frame) {
        const total = this.getTotalFrames();
        const minFc = minFrameCount(this.getTaskKey());
        if (frame <= minFc || frame >= total - minFc) return;
        const newSegs = [];
        let cursor = 0;
        for (const seg of this.timeline.segments) {
            const fc = seg.frameCount ?? seg.length;
            const end = cursor + fc;
            if (frame > cursor && frame < end) {
                const left = frame - cursor;
                const right = end - frame;
                newSegs.push({ ...seg, frameCount: left, length: left });
                newSegs.push({
                    id: uid(), start: frame, frameCount: right, length: right,
                    prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
                });
            } else {
                newSegs.push({ ...seg });
            }
            cursor = end;
        }
        this.timeline.segments = newSegs;
        this.commit();
    },
    genEqualSplit() {
        const n = parseInt(this.equalCountInput?.value || "2", 10);
        if (!n || n < 2) return;
        const total = this.getTotalFrames();
        const minFc = minFrameCount(this.getTaskKey());
        const count = clamp(n, 2, Math.max(2, Math.floor(total / minFc)));
        const base = Math.floor(total / count);
        let rem = total - base * count;
        this.timeline.segments = Array.from({ length: count }, () => {
            const fc = base + (rem > 0 ? 1 : 0);
            if (rem > 0) rem -= 1;
            return {
                id: uid(), frameCount: fc, length: fc, prompt: "", taskType: "", refs: [],
                genImage: { imageFile: "" },
            };
        });
        this.commit();
    },
    genDeleteSelectedSegment() {
        if (this.timeline.segments.length <= 1) return;
        const removed = this.selectedIndex;
        this.timeline.segments.splice(removed, 1);
        this.selectedIndex = clamp(this.selectedIndex, 0, this.timeline.segments.length - 1);
        this.onSegmentRemoved(removed);
        this.commit();
    },
    updateVideoNameLabel() {
        if (this.isFl2vMode()) {
            const shots = this.timeline.shots || [];
            const n = shots.length;
            const total = this.getTotalFrames();
            const withEnd = shots.filter((s) => s.endImage?.imageFile).length;
            const withStart = shots.filter((s) => s.startImage?.imageFile).length;
            const sec = getFl2vTotalDurationSec(this);
            if (!n) {
                this.videoNameEl.textContent = t("videoName.fl2vEmpty", { sec, frames: total });
            } else {
                this.videoNameEl.textContent = t("videoName.fl2vSummary", {
                    n, start: withStart, end: withEnd, sec, frames: total,
                });
            }
            return;
        }
        if (this.isImageBatch()) {
            // Prefer live drag preview so toolbar totals track the divider.
            const segs = this._previewSegments || this.timeline.segments || [];
            const n = segs.length || 0;
            const key = this.getTaskKey();
            if (isVideoBatchTask(key)) {
                let sec = 0;
                let total = 0;
                for (const seg of segs) {
                    const fc = Math.max(1, parseInt(seg.frameCount ?? seg.length, 10) || 1);
                    const raw = Number(seg.durationSec);
                    // During edge drag, frames are authoritative; durationSec may be stale.
                    const resolved = this._previewSegments
                        ? {
                            frames: fc,
                            durationSec: preferredDurationSecFromFrames(fc, 24),
                        }
                        : durationToClampedMiniMaxFrames(
                            Number.isFinite(raw)
                                ? raw
                                : preferredDurationSecFromFrames(fc || defaultFrameCount(key), 24),
                            24,
                        );
                    sec += resolved.durationSec;
                    total += resolved.frames;
                }
                sec = roundDurationSec(sec);
                const play = framesToDurationSec(total, 24);
                this.videoNameEl.textContent = total
                    ? t("videoName.batchVideo", {
                        key,
                        n,
                        sec: sec || play,
                        frames: total,
                        play,
                    })
                    : t("videoName.batchVideoEmpty", { key, n });
            } else {
                this.videoNameEl.textContent = t("videoName.batchImage", { key, n });
            }
            return;
        }
        if (this.isGenMode()) {
            const total = this.getTotalFrames();
            const key = this.getTaskKey();
            if (this.isGenBlank()) {
                this.videoNameEl.textContent = total
                    ? t("videoName.blankCanvas", { frames: total })
                    : t("videoName.blankCanvasNeedFrames");
            } else {
                this.videoNameEl.textContent = total
                    ? `${key} · ${total}f`
                    : t("videoName.genNeedSource", { key });
            }
            return;
        }
        const clips = this.getVideoClips();
        const total = this.getTotalFrames();
        if (!clips.length || !total) {
            this.videoNameEl.textContent = t("toolbar.noVideo");
            return;
        }
        if (clips.length === 1) {
            const c = clips[0];
            const nativeWh = c.width && c.height ? `${c.width}×${c.height}` : "";
            const storeW = c.storageWidth || this._storageWidth;
            const storeH = c.storageHeight || this._storageHeight;
            const storeWh = storeW && storeH ? `${storeW}×${storeH}` : "";
            let dim = "";
            if (nativeWh && storeWh && nativeWh !== storeWh) dim = ` · ${nativeWh} → ${storeWh}`;
            else if (nativeWh) dim = ` · ${nativeWh}`;
            else if (storeWh) dim = ` · ${storeWh}`;
            const nativeHint = c.nativeFps > 0 ? t("canvas.nativeFps", { fps: formatProbeFps(c.nativeFps) }) : "";
            const tlFps = this.getFrameRate();
            const dur = this.getTimelineDurationSec().toFixed(2);
            const name = c.fileName || c.videoFile;
            this.videoNameEl.textContent = t("videoName.singleClip", {
                name, total, fps: formatProbeFps(tlFps), dur, native: nativeHint, dim,
            });
            return;
        }
        const tlFps = this.getFrameRate();
        const dur = this.getTimelineDurationSec().toFixed(2);
        this.videoNameEl.textContent = t("videoName.multiClip", {
            n: clips.length, total, fps: formatProbeFps(tlFps), dur,
        });
    }
};
