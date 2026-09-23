/** Logical ↔ source frame mapping for a Director timeline.
 *
 * A timeline may reference several clips, reorder them, and delete ranges from the
 * middle, so a *logical* frame index (the position the user sees) is not a
 * *source* frame index (what to read out of a file). ``timeline.videoClips`` plus
 * the per-segment ``frameMap`` carry that mapping; the helpers here are the only
 * place that interprets it.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { clamp } from "./utils.js";

export function deletedSourceRanges(video) {
    return video?.deletedSourceRanges || video?.deleted_source_ranges || [];
}

export function logicalToSourceFrame(logical, video) {
    const map = video?.frameMap;
    if (map?.length) {
        return normalizeFrameMapEntry(map[clamp(logical, 0, map.length - 1)]).frame;
    }
    let src = logical;
    for (const [start, end] of [...deletedSourceRanges(video)].sort((a, b) => a[0] - b[0])) {
        if (src >= start) src += end - start;
        else break;
    }
    return src;
}

export function sourceToLogicalFrame(srcFrame, video) {
    const map = video?.frameMap;
    if (map?.length) {
        let best = -1;
        for (let i = 0; i < map.length; i++) {
            const e = normalizeFrameMapEntry(map[i]);
            if (e.frame === srcFrame) return i;
            if (e.frame < srcFrame) best = i;
            else if (best < 0) return -1; // before first kept
        }
        return best;
    }
    let logical = srcFrame;
    for (const [start, end] of [...deletedSourceRanges(video)].sort((a, b) => a[0] - b[0])) {
        if (srcFrame >= end) logical -= (end - start);
        else if (srcFrame >= start) return -1;
        else break;
    }
    return Math.max(0, logical);
}

export function buildIdentityFrameMap(count) {
    return Array.from({ length: count }, (_, i) => i);
}

export function normalizeFrameMapEntry(entry, defaultClip = 0) {
    if (entry == null) return { clip: defaultClip, frame: 0 };
    if (typeof entry === "number") return { clip: defaultClip, frame: entry };
    return {
        clip: entry.clip ?? entry.videoClip ?? defaultClip,
        frame: entry.frame ?? 0,
    };
}

export function buildClipFrameMap(clipIndex, count) {
    return Array.from({ length: count }, (_, i) => ({ clip: clipIndex, frame: i }));
}
