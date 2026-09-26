/** interaction mixin for the Director editor (interaction).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { stopDomEvent } from "../../core/editor_lifecycle.js";
import { CONT_JOINT_H, CONT_JOINT_HIT_PAD, CONT_JOINT_W, CONT_JOINT_Y, HANDLE_PX, MIN_SEG, RULER_H, RUN_CHECK_HIT_PAD_X, RUN_CHECK_HIT_PAD_Y, RUN_CHECK_SIZE, SEG_LABEL_H, THUMB_PREFETCH_BATCH, TRACK_H, TRACK_Y } from "../../core/layout_spec.js";
import { isContinuityEligible } from "../../core/timeline_sanitize.js";
import { clamp } from "../../core/utils.js";
import { flushFl2vPromptDraft, getFl2vTotalDurationSec, openFl2vUpload, rippleFl2vRightEdge, syncFl2vDurationSecAfterDrag, updateFl2vDetailUI } from "../../minimax_fl2v.js";
import { MAX_GEN_FRAMES, MAX_REFERENCE_AUDIOS, MAX_REFERENCE_IMAGES, framesToDurationSec, isContinuityMasterEnabled, isSegmentContinuityFromPrev, isVideoBatchTask, minFrameCount, preferredDurationSecFromFrames, resolveTaskKey, roundDurationSec, taskUsesReferenceAudios, taskUsesReferenceImages, taskUsesReferenceVideo } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
import { addImageBatchGroup, normalizeImageBatchSegments } from "../../minimax_image_batch.js";
export const interactionMixin = {
    getTimelineZoom() {
        return this.zoomEnabled ? Math.max(1, Number(this.zoom) || 1) : 1;
    },
    syncTimelineZoomUI() {
        this.zoomToggleBtn?.classList.toggle("active", !!this.zoomEnabled);
        this.zoomSlider?.classList.toggle("hidden", !this.zoomEnabled);
        if (this.zoomSlider && this.zoomEnabled) this.zoomSlider.value = String(this.zoom);
    },
    toggleTimelineZoom() {
        this.zoomEnabled = !this.zoomEnabled;
        this.syncTimelineZoomUI();
        this.applyZoomWidth();
        this.scheduleRender();
    },
    applyZoomWidth() {
        if (!this.canvas) return;
        const z = this.getTimelineZoom();
        const vp = this.viewport;
        if (z <= 1) {
            this.canvas.style.width = "100%";
            vp?.classList.remove("bd-zoomed");
            if (vp) vp.scrollLeft = 0;
            return;
        }
        const base = vp?.clientWidth || 960;
        const nextW = Math.max(base, base * z);
        const prevW = this.canvas.clientWidth || base;
        const midRatio = prevW > 0 ? ((vp?.scrollLeft || 0) + (vp?.clientWidth || base) / 2) / prevW : 0.5;
        this.canvas.style.width = `${nextW}px`;
        vp?.classList.add("bd-zoomed");
        if (vp) {
            requestAnimationFrame(() => {
                vp.scrollLeft = Math.max(0, midRatio * nextW - vp.clientWidth / 2);
            });
        }
    },
    adjustZoom(delta) {
        if (!this.zoomEnabled) return;
        this.zoom = clamp(this.zoom + delta, 1, 10);
        this.syncTimelineZoomUI();
        this.applyZoomWidth();
        this.scheduleRender();
    },
    frameToX(frame, width) { return (frame / Math.max(1, this.getTotalFrames())) * width; },
    xToFrame(x, width) { return clamp(Math.round((x / width) * this.getTotalFrames()), 0, this.getTotalFrames()); },
    getLayoutWidth() {
        return this._drawWidth || this._measureDrawWidth();
    },
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const layoutW = this.getLayoutWidth();
        const layoutH = this.canvasHeight || (RULER_H + SEG_LABEL_H + TRACK_H);
        const scaleX = rect.width > 0 ? layoutW / rect.width : 1;
        const scaleY = rect.height > 0 ? layoutH / rect.height : 1;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    },
    /** Shared draw + hit geometry for per-segment run checkboxes (segment top-left). */
    _runCheckGeometry(seg, width) {
        const x0 = this.frameToX(seg.start, width);
        const size = RUN_CHECK_SIZE;
        const boxX = x0 + 5;
        const boxY = TRACK_Y + 5;
        return {
            boxX,
            boxY,
            size,
            hitX0: boxX - RUN_CHECK_HIT_PAD_X,
            hitY0: boxY - RUN_CHECK_HIT_PAD_Y,
            hitX1: boxX + size + RUN_CHECK_HIT_PAD_X,
            hitY1: boxY + size + RUN_CHECK_HIT_PAD_Y,
        };
    },
    /** Master「段间引导」on + eligible task with ≥2 clips. */
    _showsContinuityJoints() {
        return isContinuityEligible(this) && isContinuityMasterEnabled(this.timeline?.output);
    },
    _continuityJointList(segs) {
        const ordered = (segs || [])
            .map((seg, arrayIndex) => ({ seg, arrayIndex }))
            .sort((a, b) => a.seg.start - b.seg.start || a.arrayIndex - b.arrayIndex);
        const joints = [];
        for (let r = 1; r < ordered.length; r++) {
            const left = ordered[r - 1];
            const right = ordered[r];
            const leftEnd = (left.seg.start || 0) + (left.seg.length || 0);
            if (Math.abs(leftEnd - (right.seg.start || 0)) > 2) continue;
            joints.push({
                leftIndex: left.arrayIndex,
                rightIndex: right.arrayIndex,
                a: r,
                b: r + 1,
                frame: right.seg.start,
                on: isSegmentContinuityFromPrev(right.seg, right.arrayIndex),
            });
        }
        return joints;
    },
    _continuityJointGeometry(frame, width) {
        const x = this.frameToX(frame, width);
        const w = CONT_JOINT_W;
        const h = CONT_JOINT_H;
        const y = CONT_JOINT_Y;
        // fl2v 首帧/尾帧 badges sit on the seam; give them a wider click target.
        const padX = this.isFl2vMode() ? 36 : CONT_JOINT_HIT_PAD;
        const padY = this.isFl2vMode() ? 12 : CONT_JOINT_HIT_PAD;
        return {
            x,
            y,
            w,
            h,
            hitX0: x - Math.max(w / 2, 16) - padX,
            hitX1: x + Math.max(w / 2, 16) + padX,
            // Include the S{n}→S{n+1} chip in the label band.
            hitY0: RULER_H + 1,
            hitY1: y + h + padY,
        };
    },
    _roundRectPath(ctx, x, y, w, h, r) {
        const rr = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
            ctx.roundRect(x, y, w, h, rr);
            return;
        }
        ctx.moveTo(x + rr, y);
        ctx.arcTo(x + w, y, x + w, y + h, rr);
        ctx.arcTo(x + w, y + h, x, y + h, rr);
        ctx.arcTo(x, y + h, x, y, rr);
        ctx.arcTo(x, y, x + w, y, rr);
        ctx.closePath();
    },
    /** Simple ↔ link glyph centered in the joint pill. */
    _drawContinuityLinkArrow(ctx, cx, cy, color) {
        const half = 6.5;
        const head = 3.2;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.6;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(cx - half, cy);
        ctx.lineTo(cx + half, cy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - half + head, cy - 3);
        ctx.lineTo(cx - half, cy);
        ctx.lineTo(cx - half + head, cy + 3);
        ctx.moveTo(cx + half - head, cy - 3);
        ctx.lineTo(cx + half, cy);
        ctx.lineTo(cx + half - head, cy + 3);
        ctx.stroke();
        ctx.restore();
    },
    _drawContinuityJoints(width, segs) {
        if (!this._showsContinuityJoints() || this._drag?.kind === "reorder") return;
        const ctx = this.ctx;
        for (const joint of this._continuityJointList(segs)) {
            const g = this._continuityJointGeometry(joint.frame, width);
            const on = joint.on;
            const accent = on ? "#4fff8f" : "#7a7a7a";
            const fill = on ? "rgba(18, 48, 32, 0.96)" : "rgba(38, 38, 38, 0.94)";
            const rx = g.x - g.w / 2;
            ctx.save();
            this._roundRectPath(ctx, rx, g.y, g.w, g.h, 8);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = accent;
            ctx.lineWidth = on ? 1.5 : 1.15;
            ctx.stroke();
            this._drawContinuityLinkArrow(ctx, g.x, g.y + g.h / 2, accent);
            const label = `S${joint.a}→S${joint.b}`;
            ctx.font = "800 8px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const tw = Math.max(g.w + 4, ctx.measureText(label).width + 8);
            const ly = RULER_H + 3;
            const lh = SEG_LABEL_H - 6;
            this._roundRectPath(ctx, g.x - tw / 2, ly, tw, lh, 4);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = on ? "#b8ffd0" : "#9a9a9a";
            ctx.fillText(label, g.x, ly + lh / 2);
            ctx.restore();
        }
    },
    toggleContinuityJoint(rightIndex) {
        if (!(rightIndex > 0)) return;
        const seg = this.timeline.segments?.[rightIndex];
        if (!seg) return;
        const next = !isSegmentContinuityFromPrev(seg, rightIndex);
        seg.continuityFromPrev = next;
        if (this.isFl2vMode()) {
            const shot = this.timeline.shots?.[rightIndex];
            if (shot) shot.continuityFromPrev = next;
        }
        this.commit(false, { syncTimeline: true });
        this.flushTimelineSync?.();
        if (this.isFl2vMode()) updateFl2vDetailUI(this);
    },
    /** Draw fl2v edge grips; joints are split (top=prev yellow, bottom=next cyan). */
    _drawFl2vEdgeHandles(segs, index, x0, x1, width) {
        const ordered = (segs || [])
            .map((seg, i) => ({ seg, i }))
            .sort((a, b) => a.seg.start - b.seg.start || a.i - b.i);
        const rank = ordered.findIndex((o) => o.i === index);
        if (rank < 0) return;
        const prev = rank > 0 ? ordered[rank - 1] : null;
        const next = rank < ordered.length - 1 ? ordered[rank + 1] : null;
        const prevX1 = prev
            ? this.frameToX(prev.seg.start + prev.seg.length, width)
            : null;
        const nextX0 = next ? this.frameToX(next.seg.start, width) : null;
        const jointLeft = prev != null && Math.abs(prevX1 - x0) <= 2;
        const jointRight = next != null && Math.abs(nextX0 - x1) <= 2;
        const mid = TRACK_Y + TRACK_H / 2;
        const half = Math.max(10, TRACK_H / 2 - 6);

        if (!jointLeft) {
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x0 - 2, mid - 12, 4, 24);
        }
        if (jointRight) {
            // Draw once on the left segment of the joint.
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x1 - 2, TRACK_Y + 4, 4, half);
            this.ctx.fillStyle = "#5ec8ff";
            this.ctx.fillRect(x1 - 2, mid + 2, 4, half);
            this.ctx.fillStyle = "rgba(255,255,255,0.85)";
            this.ctx.fillRect(x1 - 3, mid - 1, 6, 2);
        } else {
            this.ctx.fillStyle = "#ffcc00";
            this.ctx.fillRect(x1 - 2, mid - 12, 4, 24);
        }
    },
    _hitTestFl2vEdge(x, y, width, segs) {
        const ordered = (segs || [])
            .map((seg, index) => ({ seg, index }))
            .sort((a, b) => a.seg.start - b.seg.start || a.index - b.index);
        if (!ordered.length) return null;
        const trackMid = TRACK_Y + TRACK_H / 2;
        const preferNext = y >= trackMid;
        let best = null;
        let bestDist = HANDLE_PX + 1;

        for (let r = 0; r < ordered.length; r++) {
            const { seg, index } = ordered[r];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const prev = r > 0 ? ordered[r - 1] : null;
            const next = r < ordered.length - 1 ? ordered[r + 1] : null;
            const prevX1 = prev
                ? this.frameToX(prev.seg.start + prev.seg.length, width)
                : null;
            const nextX0 = next ? this.frameToX(next.seg.start, width) : null;
            const jointLeft = prev != null && Math.abs(prevX1 - x0) <= 2;
            const jointRight = next != null && Math.abs(nextX0 - x1) <= 2;

            const d0 = Math.abs(x - x0);
            if (d0 <= HANDLE_PX && d0 < bestDist) {
                if (jointLeft) {
                    best = preferNext
                        ? { type: "edge", index, edge: "left" }
                        : { type: "edge", index: prev.index, edge: "right" };
                } else {
                    best = { type: "edge", index, edge: "left" };
                }
                bestDist = d0;
            }
            const d1 = Math.abs(x - x1);
            if (d1 <= HANDLE_PX && d1 < bestDist) {
                if (jointRight) {
                    best = preferNext
                        ? { type: "edge", index: next.index, edge: "left" }
                        : { type: "edge", index, edge: "right" };
                } else {
                    best = { type: "edge", index, edge: "right" };
                }
                bestDist = d1;
            }
        }
        return best;
    },
    hitTest(x, y) {
        const width = this.getLayoutWidth();
        if (!width) return null;
        const segs = this._previewSegments || this.timeline.segments;
        const phx = this.frameToX(this.currentFrame, width);
        const trackBottom = TRACK_Y + TRACK_H;

        if (y <= RULER_H) {
            if (Math.abs(x - phx) <= HANDLE_PX) return { type: "playhead" };
            return { type: "ruler" };
        }

        // Checkbox corner wins over generic segment hit (same toggle action either way
        // in run-select mode; keeps hit type accurate for cursor / future hooks).
        if (this.isRunSelectEnabled() && this.getRunnableSegmentCount() >= 2 && y >= TRACK_Y && y <= trackBottom) {
            for (let i = segs.length - 1; i >= 0; i--) {
                if (this.isFl2vMode() && !segs[i]?.isStartFrame) continue;
                const g = this._runCheckGeometry(segs[i], width);
                if (x >= g.hitX0 && x <= g.hitX1 && y >= g.hitY0 && y <= g.hitY1) {
                    return { type: "run-check", index: i };
                }
            }
        }

        // Continuity pills sit on the seam (top of clip); win over split/edge in that pad.
        if (this._showsContinuityJoints() && y >= RULER_H && y <= TRACK_Y + CONT_JOINT_H + 24) {
            for (const joint of this._continuityJointList(segs)) {
                const g = this._continuityJointGeometry(joint.frame, width);
                if (x >= g.hitX0 && x <= g.hitX1 && y >= g.hitY0 && y <= g.hitY1) {
                    return {
                        type: "continuity-joint",
                        rightIndex: joint.rightIndex,
                        leftIndex: joint.leftIndex,
                        a: joint.a,
                        b: joint.b,
                        on: joint.on,
                    };
                }
            }
        }

        // Split markers: label band + full track height, before segment/edge hits.
        // (Previously label band returned null, so diamond clicks never registered.)
        if (y >= RULER_H && y <= trackBottom) {
            const hitPad = Math.max(HANDLE_PX, 12);
            let best = null;
            let bestDist = hitPad + 1;
            for (const frame of this.getEditableSplitFrames()) {
                const sx = this.frameToX(frame, width);
                const dist = Math.abs(x - sx);
                if (dist <= hitPad && dist < bestDist) {
                    bestDist = dist;
                    best = { type: "split", frame };
                }
            }
            if (best) return best;
        }

        if (y < TRACK_Y) return null;

        // Edge handles first so fl2v/gen can drag-extend duration (repeat thumbs).
        if (y >= TRACK_Y && y <= trackBottom) {
            if (this.isFl2vMode()) {
                const flHit = this._hitTestFl2vEdge(x, y, width, segs);
                if (flHit) return flHit;
            } else {
                for (let i = 0; i < segs.length; i++) {
                    const seg = segs[i];
                    const x0 = this.frameToX(seg.start, width);
                    const x1 = this.frameToX(seg.start + seg.length, width);
                    if (Math.abs(x - x0) <= HANDLE_PX) return { type: "edge", index: i, edge: "left" };
                    if (Math.abs(x - x1) <= HANDLE_PX) return { type: "edge", index: i, edge: "right" };
                }
            }
        }

        for (let i = segs.length - 1; i >= 0; i--) {
            const seg = segs[i];
            const x0 = this.frameToX(seg.start, width);
            const x1 = this.frameToX(seg.start + seg.length, width);
            const isLast = i === segs.length - 1;
            const insideX = isLast ? (x >= x0 && x <= x1) : (x >= x0 && x < x1);
            if (insideX && y >= TRACK_Y && y <= trackBottom) {
                return { type: "segment", index: i };
            }
        }

        if (Math.abs(x - phx) <= HANDLE_PX) return { type: "playhead" };
        return null;
    },
    onMouseDown(e) {
        if (e.button !== 0) return;
        // Keep LiteGraph / node drag from eating timeline clicks.
        stopDomEvent(e);
        e.preventDefault();
        const { x, y } = this.getMousePos(e);
        const hit = this.hitTest(x, y);
        if (!hit) {
            if (
                this.isFl2vMode()
                && !(this.timeline.segments || []).length
                && y >= TRACK_Y
                && y <= TRACK_Y + TRACK_H
            ) {
                openFl2vUpload(this);
            } else if (
                this.usesBatchTimeline()
                && !(this.timeline.segments || []).length
                && y >= TRACK_Y
                && y <= TRACK_Y + TRACK_H
            ) {
                addImageBatchGroup(this);
            } else if (
                this.needsSourceVideoUpload()
                && y >= TRACK_Y
                && y <= TRACK_Y + TRACK_H
            ) {
                this.pickVideoFile();
            }
            return;
        }
        if (
            this.needsSourceVideoUpload()
            && (hit.type === "segment" || hit.type === "edge")
        ) {
            this.pickVideoFile();
            return;
        }
        const width = this.getLayoutWidth();
        if (hit.type === "playhead" || hit.type === "ruler") {
            this.currentFrame = this.xToFrame(x, width);
            this._drag = { kind: "playhead" };
            this.clearSplitSelection();
        } else if (hit.type === "run-check") {
            this.toggleSegmentRun(hit.index);
            this._drag = null;
        } else if (hit.type === "continuity-joint") {
            this.toggleContinuityJoint(hit.rightIndex);
            this._drag = null;
        } else if (hit.type === "split") {
            this.selectSplitFrame(hit.frame);
            this._drag = null;
        } else if (hit.type === "segment") {
            if (this.isFl2vMode() && hit.index !== this.selectedIndex) {
                flushFl2vPromptDraft(this);
            }
            this.selectedIndex = hit.index;
            this.clearSplitSelection();
            this.updateSelectionUI();
            if (this.isFl2vMode() || this.usesBatchTimeline() || this.timeline.segments.length >= 2) {
                // Drag body to reorder / swap clip positions; edges still resize.
                this._drag = {
                    kind: "segment-pending",
                    index: hit.index,
                    x0: x,
                    y0: y,
                    fromRank: this._visualRankFromArrayIndex(hit.index),
                };
            } else {
                this._drag = { kind: "segment" };
            }
        } else if (hit.type === "edge") {
            if (this.isFl2vMode() && hit.index !== this.selectedIndex) {
                flushFl2vPromptDraft(this);
            }
            this.selectedIndex = hit.index;
            this.clearSplitSelection();
            this.updateSelectionUI();
            this._drag = { kind: "edge", index: hit.index, edge: hit.edge };
            this._edgeSnapshot = JSON.parse(JSON.stringify(this.timeline.segments));
        }
        this.scheduleRender();
    },
    onMouseMove(e) {
        if (!this._drag) return;
        const { x, y } = this.getMousePos(e);
        const width = this.getLayoutWidth();
        const frame = this.xToFrame(x, width);

        if (this._drag.kind === "segment-pending") {
            if (Math.hypot(x - this._drag.x0, y - this._drag.y0) > 6) {
                this._drag = {
                    kind: "reorder",
                    fromRank: this._drag.fromRank,
                    index: this._drag.index,
                    pointerX: x,
                    pointerY: y,
                    originX: this._drag.x0,
                    originY: this._drag.y0,
                };
                this._reorderFromRank = this._drag.fromRank;
                this._reorderDropRank = this._drag.fromRank;
                this.canvas.classList.add("bd-grabbing");
                this.canvas.style.cursor = "grabbing";
            }
            return;
        }

        if (this._drag.kind === "playhead") {
            this.currentFrame = frame;
        } else if (this._drag.kind === "reorder") {
            this._drag.pointerX = x;
            this._drag.pointerY = y;
            this._reorderDropRank = this._computeReorderDropRank(frame, this._drag.fromRank);
            this.scheduleRender();
            return;
        } else if (this._drag.kind === "fl2v-move") {
            // Block-move: this clip + all later clips shift together (LTX ripple).
            const snap = this._edgeSnapshot || this.timeline.segments;
            const segs = snap.map((s) => ({ ...s }));
            const i = this._drag.index;
            const seg = segs[i];
            if (!seg) return;
            const width = this.getLayoutWidth();
            const frame0 = this.xToFrame(this._drag.x0, width);
            let delta = frame - frame0;
            const ordered = segs
                .map((s, idx) => ({ s, idx }))
                .sort((a, b) => a.s.start - b.s.start || a.idx - b.idx);
            const rank = ordered.findIndex((o) => o.s.id === seg.id);
            if (rank < 0) return;
            const prev = rank > 0 ? ordered[rank - 1].s : null;
            const minStart = prev ? prev.start + prev.length : 0;
            const desired = this._drag.start0 + delta;
            const clampedStart = Math.max(minStart, desired);
            delta = clampedStart - this._drag.start0;
            for (let r = rank; r < ordered.length; r++) {
                const orig = snap.find((x) => x.id === ordered[r].s.id) || ordered[r].s;
                ordered[r].s.start = Math.max(0, (parseInt(orig.start, 10) || 0) + delta);
                ordered[r].s.length = Math.max(minFrameCount("fl2v"), parseInt(orig.length, 10) || minFrameCount("fl2v"));
                ordered[r].s.frameCount = ordered[r].s.length;
            }
            this._previewSegments = segs;
        } else if (this._drag.kind === "edge") {
            const segs = this._edgeSnapshot.map((s) => ({ ...s }));
            const i = this._drag.index;
            const seg = segs[i];
            const isFl2v = this.isFl2vMode();
            const isGen = this.isGenMode();
            const isBatchTrack = this.usesBatchTimeline();
            const minLen = (isFl2v || isGen || isBatchTrack) ? minFrameCount(this.getTaskKey()) : MIN_SEG;
            if (isFl2v) {
                // LTX-style ripple: resize this clip's right edge and shift ALL later clips.
                // Left edge of a non-first clip = ripple the previous clip's right edge.
                // May extend past the sampling window (dashed overflow, not sampled).
                const ordered = [...segs]
                    .map((s, idx) => ({ s, idx }))
                    .sort((a, b) => a.s.start - b.s.start || a.idx - b.idx);
                const rank = ordered.findIndex((o) => o.s.id === seg.id);
                if (this._drag.edge === "right") {
                    const newEnd = Math.max(seg.start + minLen, frame);
                    rippleFl2vRightEdge(segs, i, newEnd, minLen, this);
                } else if (this._drag.edge === "left") {
                    if (rank > 0) {
                        const prevIdx = ordered[rank - 1].idx;
                        const prev = ordered[rank - 1].s;
                        const newEnd = Math.max(prev.start + minLen, frame);
                        rippleFl2vRightEdge(segs, prevIdx, newEnd, minLen, this);
                    }
                    // First clip's left edge stays at 0 (no negative timeline).
                }
            } else if (this._drag.edge === "left") {
                const prev = segs[i - 1];
                const minStart = prev ? prev.start + minLen : 0;
                const maxStart = seg.start + seg.length - minLen;
                seg.start = clamp(frame, minStart, maxStart);
                seg.length = (this._edgeSnapshot[i].start + this._edgeSnapshot[i].length) - seg.start;
                if (isGen || isBatchTrack) seg.frameCount = seg.length;
                if (prev) {
                    prev.length = seg.start - prev.start;
                    if (isGen || isBatchTrack) prev.frameCount = prev.length;
                }
            } else {
                const next = segs[i + 1];
                const minEnd = seg.start + minLen;
                let maxEnd;
                if (next) {
                    maxEnd = this._edgeSnapshot[i + 1].start + this._edgeSnapshot[i + 1].length;
                    if (isGen || isBatchTrack) maxEnd -= minLen;
                } else if (isGen || isBatchTrack) {
                    maxEnd = seg.start + MAX_GEN_FRAMES;
                } else {
                    maxEnd = this.getTotalFrames();
                }
                const end = clamp(frame, minEnd, maxEnd);
                seg.length = end - seg.start;
                if (isGen || isBatchTrack) seg.frameCount = seg.length;
                if (next) {
                    next.start = end;
                    next.length = (this._edgeSnapshot[i + 1].start + this._edgeSnapshot[i + 1].length) - end;
                    if (isGen || isBatchTrack) next.frameCount = next.length;
                }
            }
            this._previewSegments = segs;
            this._syncLiveDurationUiFromPreview();
        }
        this.scheduleRender();
    },
    _syncLiveDurationUiFromPreview() {
        const segs = this._previewSegments;
        if (!segs?.length) return;

        if (this.isR2vBatch() || (this.isImageBatch() && isVideoBatchTask(this.getTaskKey()))) {
            for (const input of this.batchList?.querySelectorAll("input[data-batch-sec-index]") || []) {
                if (input === document.activeElement) continue;
                const index = parseInt(input.getAttribute("data-batch-sec-index"), 10);
                if (!Number.isFinite(index)) continue;
                const seg = segs[index];
                if (!seg) continue;
                const fc = Math.max(1, parseInt(seg.frameCount ?? seg.length, 10) || 1);
                const sec = preferredDurationSecFromFrames(fc, 24);
                const play = framesToDurationSec(fc, 24);
                // Duration picker renders a label, not the raw number.
                const combo = input.__bdDurCombo;
                if (combo) {
                    combo.setSec(sec);
                    continue;
                }
                if (input.value !== String(sec)) input.value = String(sec);
                input.title = t("batch.durationTooltip", { frames: fc, play });
            }
            this.updateVideoNameLabel();
            this.updateOutputPreview();
            return;
        }

        if (this.isFl2vMode()) {
            const shots = this.timeline.shots || [];
            for (const card of this.fl2vUi?.shotsEl?.querySelectorAll(".bd-fl2v-shot") || []) {
                const index = parseInt(card.dataset.shotIndex, 10);
                const input = card.querySelector('[data-r="shot-sec"]');
                const shot = shots[index];
                if (!input || !shot || input === document.activeElement) continue;
                const sec = roundDurationSec(Number(shot.durationSec) || 0);
                const combo = input.__bdDurCombo;
                if (combo) {
                    combo.setSec(sec);
                    continue;
                }
                if (input.value !== String(sec)) input.value = String(sec);
            }
            if (this.fl2vUi?.totalInput && this.fl2vUi.totalInput !== document.activeElement) {
                this.fl2vUi.totalInput.value = String(getFl2vTotalDurationSec(this));
            }
            this.updateVideoNameLabel();
            this.updateOutputPreview();
            return;
        }

        // v2v / rv2v video timeline: live-update segment header + bounds while dragging.
        this._updateSegInfoFromSegment?.(segs[this.selectedIndex]);
        this._updateTimelineDom?.({ skipSeek: true });
        this.updateOutputPreview();
    },
    /** Build / refresh the segment panel meta line (frames, duration, ref counts). */
    _updateSegInfoFromSegment(seg) {
        if (!this.segInfo || !seg || this.isGlobalMode()) return;
        const fps = this.getFrameRate();
        const segKey = resolveTaskKey(
            seg.taskType || this.timeline.global?.taskType || this.getTaskKey(),
        );
        let info;
        if (this.isGenMode()) {
            const fc = seg.frameCount ?? seg.length;
            info = t("segment.infoFrames", { n: fc });
            if (this.isGenImage()) {
                info += seg.genImage?.imageFile ? t("segment.uploadedImage") : t("segment.noImage");
            }
        } else {
            info = t("segment.infoRange", {
                start: seg.start,
                end: seg.start + seg.length,
                length: seg.length,
                sec: (seg.length / fps).toFixed(2),
            });
            const clips = this.getVideoClips();
            if (clips.length > 1) {
                const clip = clips[this.getSegmentClipIndex(seg)];
                const clipName = clip?.fileName || clip?.videoFile
                    || t("slot.video", { n: this.getSegmentClipIndex(seg) + 1 });
                info += ` · ${clipName}`;
            }
            if (taskUsesReferenceVideo(segKey)) {
                info += seg.referenceVideo?.videoFile || seg.referenceVideo?.fileName
                    ? t("segment.refVideoUploaded")
                    : t("segment.refVideoMissing");
            }
            if (taskUsesReferenceImages(segKey) || taskUsesReferenceAudios(segKey)) {
                let imgs = 0;
                let audios = 0;
                for (const r of seg.refs || []) {
                    if (r?.imageFile || r?.imageB64) imgs += 1;
                }
                for (const r of seg.refAudios || []) {
                    if (r?.audioFile || r?.fileName) audios += 1;
                }
                info += ` · ${t("segment.refSummary", {
                    imgs,
                    maxImgs: MAX_REFERENCE_IMAGES,
                    audios,
                    maxAudios: MAX_REFERENCE_AUDIOS,
                })}`;
            }
        }
        this.segInfo.textContent = info;
    },
    onMouseUp() {
        if (
            (this._drag?.kind === "edge" || this._drag?.kind === "fl2v-move")
            && this._previewSegments
        ) {
            const preview = this._previewSegments;
            this._previewSegments = null;
            if (this.isFl2vMode()) {
                // Shot durations already updated during drag; rebuild layout from shots.
                syncFl2vDurationSecAfterDrag(this);
                updateFl2vDetailUI(this);
                this.updateVideoNameLabel();
            } else if (this.usesBatchTimeline()) {
                this.timeline.segments = preview;
                for (const seg of this.timeline.segments) {
                    const fc = Math.max(1, parseInt(seg.frameCount ?? seg.length, 10) || 1);
                    seg.frameCount = fc;
                    seg.length = fc;
                    seg.durationSec = preferredDurationSecFromFrames(fc, 24);
                }
                normalizeImageBatchSegments(this);
                this.renderImageBatchGroups();
                this.updateVideoNameLabel();
            } else {
                this._applyOuterVideoCrop(preview);
            }
            this.commit();
        } else if (this._drag?.kind === "reorder") {
            const toRank = this._reorderDropRank;
            if (toRank >= 0 && toRank !== this._drag.fromRank) {
                this.reorderSegmentsByRank(this._drag.fromRank, toRank);
                this.commit(false, { syncTimeline: true });
                if (this.isFl2vMode()) {
                    updateFl2vDetailUI(this);
                    this.updateVideoNameLabel();
                } else if (this.usesBatchTimeline()) {
                    this.renderImageBatchGroups();
                    this.updateVideoNameLabel();
                }
            }
            this._reorderDropRank = -1;
            this._reorderFromRank = -1;
            this.canvas.classList.remove("bd-grabbing");
            this.canvas.style.cursor = "";
        } else if (this._drag) {
            this.seekBar.value = this.currentFrame;
            this.scheduleRender();
        }
        this._drag = null;
        this._edgeSnapshot = null;
    },
    _applyOuterVideoCrop(preview) {
        const total = this.getTotalFrames();
        const ordered = [...(preview || [])].sort((a, b) => a.start - b.start);
        if (!ordered.length || total <= 0) {
            this.timeline.segments = preview || [];
            return false;
        }
        const cropStart = clamp(Math.round(Number(ordered[0].start) || 0), 0, total);
        const last = ordered[ordered.length - 1];
        const cropEnd = clamp(
            Math.round((Number(last.start) || 0) + (Number(last.length) || 0)),
            cropStart,
            total,
        );
        if (cropStart <= 0 && cropEnd >= total) {
            this.timeline.segments = preview;
            return false;
        }

        if (!this.getFrameMap().length) this.materializeFrameMap();
        const croppedMap = this.getFrameMap().slice(cropStart, cropEnd);
        const selectedId = this.timeline.segments?.[this.selectedIndex]?.id;
        this.setFrameMap(croppedMap);
        this.timeline.totalFrames = croppedMap.length;
        this._syncPrimaryVideoFromClips(croppedMap);
        this.timeline.videoWorkspace = null;
        this.timeline.segments = ordered.flatMap((seg) => {
            const start = Math.max(cropStart, Number(seg.start) || 0);
            const end = Math.min(cropEnd, (Number(seg.start) || 0) + (Number(seg.length) || 0));
            if (end - start < MIN_SEG) return [];
            return [{ ...seg, start: start - cropStart, length: end - start }];
        });
        this.selectedIndex = Math.max(
            0,
            this.timeline.segments.findIndex((seg) => seg.id === selectedId),
        );
        this.currentFrame = clamp(this.currentFrame - cropStart, 0, Math.max(0, croppedMap.length - 1));
        if (this.seekBar) {
            this.seekBar.max = Math.max(0, croppedMap.length - 1);
            this.seekBar.value = this.currentFrame;
        }
        if (this.totalFramesWidget) this.totalFramesWidget.value = croppedMap.length;
        this._thumbCache.clear();
        this._thumbPending.clear();
        this._prefetchSegmentThumbs(0, Math.min(croppedMap.length, THUMB_PREFETCH_BATCH * 4));
        this._syncStagePreview(this.currentFrame, { force: true });
        this.updateVideoNameLabel();
        return true;
    }
};
