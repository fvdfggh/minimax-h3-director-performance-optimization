/** Timeline ruler ticks: step selection and label formatting.
 *
 * The ruler is drawn in canvas pixels, so the tick step has to be derived from the
 * *visible* time range — a fixed step either crowds into unreadable hairlines when
 * zoomed out or vanishes when zoomed in. ``RULER_MAJOR_SEC`` is the ladder of
 * human-readable steps; the two MIN_*_PX values are the minimum on-screen spacing
 * that decides which rung is usable.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

/** Nice major steps (seconds) for CapCut-style ruler labels. */
export const RULER_MAJOR_SEC = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const RULER_MIN_MAJOR_PX = 64;
const RULER_MIN_MINOR_PX = 7;

export function pickRulerMajorStepSec(pxPerSec) {
    const pps = Math.max(0.001, Number(pxPerSec) || 0.001);
    for (const step of RULER_MAJOR_SEC) {
        if (step * pps >= RULER_MIN_MAJOR_PX) return step;
    }
    return RULER_MAJOR_SEC[RULER_MAJOR_SEC.length - 1];
}

export function pickRulerMinorStepSec(majorSec, pxPerSec) {
    const pps = Math.max(0.001, Number(pxPerSec) || 0.001);
    for (const div of [10, 5, 4, 2]) {
        const minor = majorSec / div;
        if (minor >= 1 && Number.isInteger(minor) && minor * pps >= RULER_MIN_MINOR_PX) {
            return minor;
        }
    }
    return majorSec;
}

export function formatRulerTime(sec) {
    const s = Math.max(0, Math.round(Number(sec) || 0));
    if (s < 60) return String(s);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
}

export function formatProbeFps(value) {
    const fps = Math.round(Number(value) * 100) / 100;
    if (Number.isInteger(fps)) return String(fps);
    return fps.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}
