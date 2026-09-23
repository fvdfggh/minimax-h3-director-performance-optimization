/** Output geometry: dimension snapping, segment resize, fps coercion.
 *
 * These exist because the backend is strict: width/height must be multiples of 32,
 * the frame rate must be a real number, and a segment length has to land on the
 * 17k+5 grid. The UI must never hand the backend a value it would reject, so every
 * path that produces a dimension goes through here.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { clamp } from "./utils.js";

export function snapDim(v, stride = 32) {
    return Math.max(stride, Math.round(v / stride) * stride);
}

export function snapScaledDim(dim, scale, stride = 32) {
    return Math.max(stride, Math.round((dim * scale) / stride) * stride);
}

export function resolveOutputDimensions(sourceW, sourceH, output, fallback = {}) {
    const mode = String(output?.mode || "long_edge").toLowerCase();
    const canvasStride = 32;
    if (mode === "fixed") {
        const w = snapDim(+(output?.width ?? fallback.width ?? 864), canvasStride);
        const h = snapDim(+(output?.height ?? fallback.height ?? 480), canvasStride);
        return { mode: "fixed", width: w, height: h, refMaxSize: Math.max(w, h) };
    }
    const longEdge = Math.max(canvasStride, +(output?.longEdge ?? output?.long_edge ?? fallback.refMaxSize ?? 848));
    const sw = sourceW || 0;
    const sh = sourceH || 0;
    if (!sw || !sh) {
        // Missing source: keep long-edge budget only — do not invent a 16:9 canvas
        // (that would center-crop ultrawide footage later via fit_canvas).
        return { mode: "long_edge", width: longEdge, height: canvasStride, refMaxSize: longEdge };
    }
    // Always recompute from source (even when already ≤ longEdge) so snapped
    // dims stay aspect-correct; never reuse a stale fixed W×H.
    const scale = Math.min(1, longEdge / Math.max(sw, sh));
    return {
        mode: "long_edge",
        width: snapScaledDim(sw, scale, canvasStride),
        height: snapScaledDim(sh, scale, canvasStride),
        refMaxSize: longEdge,
    };
}

export function coerceTimelineFps(value, fallback = 24) {
    const fps = Number(value);
    if (!Number.isFinite(fps) || fps <= 0) return coerceTimelineFps(fallback, 24);
    return Math.round(clamp(fps, 1, 240) * 100) / 100;
}
