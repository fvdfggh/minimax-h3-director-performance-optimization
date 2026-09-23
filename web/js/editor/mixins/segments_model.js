/** segments_model mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { logicalToSourceFrame, normalizeFrameMapEntry } from "../../core/frame_map.js";
import { MIN_SEG, THUMB_PREFETCH_BATCH } from "../../core/layout_spec.js";
import { clamp, uid } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { syncFl2vFromShots, updateFl2vDetailUI } from "../../minimax_fl2v.js";
import { minFrameCount } from "../../minimax_gen_timeline.js";
import { normalizeImageBatchSegments } from "../../minimax_image_batch.js";
export const segments_modelMixin = {
    getFrameMapEntry(logicalFrame) {
        const map = this.getFrameMap();
        if (map.length) return normalizeFrameMapEntry(map[clamp(logicalFrame, 0, map.length - 1)]);
        return { clip: 0, frame: logicalToSourceFrame(logicalFrame, this.timeline.video || {}) };
    },
    getSegmentClipIndex(seg) {
        return this.getFrameMapEntry(seg.start).clip;
    },
    getClipBoundaries() {
        const map = this.getFrameMap();
        const boundaries = [];
        for (let i = 1; i < map.length; i++) {
            const a = normalizeFrameMapEntry(map[i - 1]);
            const b = normalizeFrameMapEntry(map[i]);
            if (b.clip !== a.clip) boundaries.push(i);
        }
        return boundaries;
    },
    _segmentMetaAtFrame(frame) {
        const segs = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        for (const seg of segs) {
            if (frame >= seg.start && frame < seg.start + seg.length) {
                return {
                    prompt: seg.prompt || "",
                    taskType: seg.taskType || "",
                    refs: seg.refs ? JSON.parse(JSON.stringify(seg.refs)) : [],
                };
            }
        }
        const last = segs[segs.length - 1];
        if (last) {
            return {
                prompt: last.prompt || "",
                taskType: last.taskType || "",
                refs: last.refs ? JSON.parse(JSON.stringify(last.refs)) : [],
            };
        }
        return { prompt: "", taskType: "", refs: [] };
    },
    _buildSegmentsFromSplitPoints(points, forcedPoints = null) {
        const forced = new Set(forcedPoints || []);
        forced.add(0);
        const sorted = [...new Set(points)].sort((a, b) => a - b);
        forced.add(sorted[sorted.length - 1]);
        const newSegs = [];
        for (let i = 0; i < sorted.length - 1; i++) {
            const start = sorted[i];
            const length = sorted[i + 1] - start;
            const endsForced = forced.has(sorted[i + 1]);
            const startsForced = forced.has(start);
            if (length < MIN_SEG && !endsForced && !startsForced) continue;
            if (length < 1) continue;
            const meta = this._segmentMetaAtFrame(start);
            newSegs.push({
                id: uid(),
                start,
                length,
                prompt: meta.prompt,
                taskType: meta.taskType,
                refs: meta.refs,
            });
        }
        if (!newSegs.length) return null;
        let cursor = 0;
        return newSegs.map((seg) => {
            const s = { ...seg, start: cursor, length: seg.length };
            cursor += s.length;
            return s;
        });
    },
    _getReorderInsertFrame(dropRank, fromRank) {
        const ordered = [...this.timeline.segments].sort((a, b) => a.start - b.start);
        const lengths = ordered.map((s) => s.length);
        const without = lengths.filter((_, i) => i !== fromRank);
        let frame = 0;
        for (let i = 0; i < dropRank && i < without.length; i++) frame += without[i];
        return frame;
    },
    _orderedSegmentsWithRank() {
        return [...this.timeline.segments]
            .map((seg, arrayIndex) => ({ seg, arrayIndex }))
            .sort((a, b) => a.seg.start - b.seg.start)
            .map((item, visualRank) => ({ ...item, visualRank }));
    },
    _visualRankFromArrayIndex(arrayIndex) {
        const ordered = this._orderedSegmentsWithRank();
        return ordered.find((o) => o.arrayIndex === arrayIndex)?.visualRank ?? arrayIndex;
    },
    _computeReorderDropRank(frame, fromRank) {
        const ordered = this._orderedSegmentsWithRank();
        if (!ordered.length) return fromRank;

        // fl2v: swap slots — drop target = the clip currently under the pointer.
        if (this.isFl2vMode()) {
            for (const item of ordered) {
                const lo = item.seg.start;
                const hi = item.seg.start + item.seg.length;
                if (frame >= lo && frame < hi) return item.visualRank;
            }
            // In a gap / past the end: snap to nearest clip by center distance.
            let best = fromRank;
            let bestDist = Infinity;
            for (const item of ordered) {
                const mid = item.seg.start + item.seg.length / 2;
                const d = Math.abs(frame - mid);
                if (d < bestDist) {
                    bestDist = d;
                    best = item.visualRank;
                }
            }
            return best;
        }

        // Video / gen / batch: return the final insertion index after removing
        // the dragged item. This keeps forward moves from collapsing to no-op.
        const remaining = ordered.filter((item) => item.visualRank !== fromRank);
        for (let index = 0; index < remaining.length; index++) {
            const item = remaining[index];
            const mid = item.seg.start + item.seg.length / 2;
            if (frame < mid) return index;
        }
        return remaining.length;
    },
    reorderSegmentsByRank(fromRank, toRank) {
        const ordered = [...this.timeline.segments]
            .map((seg) => ({ seg }))
            .sort((a, b) => a.seg.start - b.seg.start);
        if (fromRank < 0 || fromRank >= ordered.length) return;
        if (toRank < 0 || toRank >= ordered.length) return;
        if (fromRank === toRank) return;
        // Ticks are positions, not identities — carry them across the move.
        this.moveRunSelectionIndex(fromRank, toRank);

        // fl2v: reorder shots[] (source of truth), then rebuild segments.
        if (this.isFl2vMode()) {
            const shots = [...(this.timeline.shots || [])];
            if (fromRank < 0 || fromRank >= shots.length) return;
            if (toRank < 0 || toRank >= shots.length) return;
            const [moved] = shots.splice(fromRank, 1);
            const insertRank = toRank;
            shots.splice(insertRank, 0, moved);
            this.timeline.shots = shots;
            syncFl2vFromShots(this);
            this.selectedIndex = insertRank;
            updateFl2vDetailUI(this);
            this.updateVideoNameLabel();
            return;
        }
        // r2v / t2v / i2v: move whole groups then renumber starts.
        if (this.usesBatchTimeline()) {
            const metas = ordered.map((o) => ({
                ...o.seg,
                refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
                refAudios: o.seg.refAudios ? JSON.parse(JSON.stringify(o.seg.refAudios)) : [],
                refVideos: o.seg.refVideos ? JSON.parse(JSON.stringify(o.seg.refVideos)) : [],
            }));
            const [mMeta] = metas.splice(fromRank, 1);
            const insertRank = toRank;
            metas.splice(insertRank, 0, mMeta);
            this.timeline.segments = metas;
            normalizeImageBatchSegments(this);
            this.selectedIndex = insertRank;
            this.updateVideoNameLabel();
            return;
        }
        // gen: no video frameMap — reorder by segment metadata only.
        if (this.isGenMode()) {
            const metas = ordered.map((o) => ({
                ...o.seg,
                refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
            }));
            const slots = ordered.map((o) => ({
                start: o.seg.start,
                length: o.seg.length || o.seg.frameCount || minFrameCount(this.getTaskKey()),
            }));
            const [mMeta] = metas.splice(fromRank, 1);
            const insertRank = toRank;
            metas.splice(insertRank, 0, mMeta);
            for (let i = 0; i < metas.length; i++) {
                const slot = slots[i] || slots[slots.length - 1];
                metas[i].start = slot.start;
                metas[i].length = slot.length;
                metas[i].frameCount = slot.length;
            }
            this.timeline.segments = metas;
            this.normalizeGenSegments();
            this.selectedIndex = insertRank;
            this.updateVideoNameLabel();
            return;
        }

        if (!this.getFrameMap().length && this.getTotalFrames() > 0) {
            this.materializeFrameMap();
        }
        const map = [...this.getFrameMap()];
        const slices = ordered.map((o) => map.slice(o.seg.start, o.seg.start + o.seg.length));
        const metas = ordered.map((o) => ({
            ...o.seg,
            refs: o.seg.refs ? JSON.parse(JSON.stringify(o.seg.refs)) : [],
        }));

        const [mSlice] = slices.splice(fromRank, 1);
        const [mMeta] = metas.splice(fromRank, 1);
        const insertRank = toRank;
        slices.splice(insertRank, 0, mSlice);
        metas.splice(insertRank, 0, mMeta);

        const newMap = slices.flat();
        let start = 0;
        const newSegs = metas.map((seg, idx) => {
            const s = { ...seg, start, length: slices[idx].length };
            start += s.length;
            return s;
        });

        this.setFrameMap(newMap);
        this.timeline.segments = newSegs;
        this._syncPrimaryVideoFromClips(newMap);
        this.selectedIndex = insertRank;
        this._prefetchSegmentThumbs(0, Math.min(newMap.length, THUMB_PREFETCH_BATCH * 4));
    }
};
