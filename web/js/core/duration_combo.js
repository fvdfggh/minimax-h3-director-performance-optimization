/**
 * Searchable duration picker for MiniMax frame-grid durations.
 *
 * Options come from the frame grid itself (k = 1…):
 * - standalone   → 17k + 5  (22f, 39f, 56f, …)
 * - 引用上段     → 17k      (17f, 34f, 51f, …)
 * rendered as seconds (frames / fps) up to DURATION_MAX_SEC, so the user picks a
 * real, generatable clip length instead of typing one.
 */
import {
    durationOptions,
    durationToClampedMiniMaxFrames,
    formatDurationOption,
    framesToDurationSec,
} from "../minimax_gen_timeline.js";
import { t } from "../minimax_i18n.js";

function gridLabel(continuity) {
    return continuity ? "17k" : "17k+5";
}

function matchesOption(opt, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return true;
    if (opt.label.toLowerCase().includes(q)) return true;
    const n = Number(q);
    if (Number.isFinite(n)) {
        if (String(opt.frames) === q) return true;
        if (Math.abs(opt.sec - n) < 0.005) return true;
    }
    return false;
}

/**
 * @param {object}  opts
 * @param {number}  opts.sec         current duration in seconds
 * @param {boolean} opts.continuity  true when this row pins to the previous segment
 * @param {number}  opts.fps         timeline fps (default 24)
 * @param {boolean} opts.disabled    read-only (external group mode)
 * @param {object}  opts.attrs       attributes copied onto the <input> (data-* hooks)
 * @param {(sec:number, frames:number)=>void} opts.onCommit
 */
export function createDurationCombo({
    sec = 5,
    continuity = false,
    fps = 24,
    disabled = false,
    attrs = {},
    onCommit = null,
} = {}) {
    const rate = Math.max(1, Number(fps) || 24);
    const wrap = document.createElement("div");
    wrap.className = "bd-dur-combo";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "bd-dur-input bd-num";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = t("duration.searchHint");
    for (const [k, v] of Object.entries(attrs || {})) {
        if (v == null) continue;
        input.setAttribute(k, String(v));
    }
    // The popup is mounted on <body> on open: ComfyUI keeps node DOM inside a
    // CSS-transformed canvas container (a fixed popup would be offset from the
    // input), and the card list has overflow-y:auto (which clips one that stays
    // inside). Document coordinates below account for page scroll.
    const list = document.createElement("div");
    list.className = "bd-dur-list hidden";
    wrap.appendChild(input);

    let options = [];
    let current = { sec: Number(sec) || 0, frames: 0, continuity: !!continuity };
    let open = false;
    let activeIndex = -1;
    let locked = !!disabled;

    function buildOptions() {
        options = durationOptions(current.continuity, rate).map((opt) => ({
            ...opt,
            label: formatDurationOption(opt),
        }));
    }

    function displayLabel(frames) {
        const exact = options.find((o) => o.frames === frames);
        if (exact) return exact.label;
        return formatDurationOption({ sec: framesToDurationSec(frames, rate), frames });
    }

    function syncTitle() {
        input.title = t("duration.optionTitle", {
            sec: Number(current.sec || 0).toFixed(2),
            frames: current.frames,
            grid: gridLabel(current.continuity),
        });
    }

    /** Grow the field so the selected value (seconds + frames) is never clipped. */
    function fitWidth() {
        const wide = [...(input.value || "")].reduce(
            (w, ch) => w + (ch.charCodeAt(0) > 0x2e80 ? 11 : 6.2),
            0,
        );
        input.style.width = `${Math.round(Math.max(150, Math.min(260, wide + 26)))}px`;
    }

    function setValue(nextSec) {
        const resolved = durationToClampedMiniMaxFrames(nextSec, rate, current.continuity);
        current.sec = resolved.durationSec;
        current.frames = resolved.frames;
        input.value = displayLabel(resolved.frames);
        fitWidth();
        syncTitle();
    }

    function renderList(query) {
        const q = query || "";
        const nodes = list.childNodes;
        while (nodes.length) list.removeChild(nodes[0]);
        let visible = 0;
        let firstMatch = -1;
        options.forEach((opt, i) => {
            if (!matchesOption(opt, q)) return;
            if (firstMatch < 0) firstMatch = i;
            visible += 1;
            const row = document.createElement("div");
            row.className = "bd-dur-opt";
            if (opt.frames === current.frames) row.classList.add("selected");
            if (i === activeIndex) row.classList.add("cursor");
            row.textContent = opt.label;
            row.dataset.index = String(i);
            row.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (locked) return;
                select(i);
            });
            row.addEventListener("mouseenter", () => {
                if (activeIndex === i) return;
                activeIndex = i;
                list.querySelector(".bd-dur-opt.cursor")?.classList.remove("cursor");
                row.classList.add("cursor");
            });
            list.appendChild(row);
        });
        if (!visible) {
            const empty = document.createElement("div");
            empty.className = "bd-dur-opt bd-dur-empty";
            empty.textContent = t("duration.noMatch");
            list.appendChild(empty);
        }
        // Scroll the cursor row into view inside the popup (never the page).
        const cursor = list.querySelector(".bd-dur-opt.cursor");
        if (cursor) {
            const top = cursor.offsetTop;
            const bottom = top + cursor.offsetHeight;
            if (top < list.scrollTop) list.scrollTop = top;
            else if (bottom > list.scrollTop + list.clientHeight) {
                list.scrollTop = bottom - list.clientHeight;
            }
        }
        return firstMatch;
    }

    function reposition() {
        const r = input.getBoundingClientRect();
        const below = window.innerHeight - r.bottom;
        const above = r.top;
        const openUp = below < 140 && above > below;
        const scrollX = window.scrollX || window.pageXOffset || 0;
        const scrollY = window.scrollY || window.pageYOffset || 0;
        list.style.left = `${Math.round(r.left + scrollX)}px`;
        list.style.minWidth = `${Math.round(Math.max(150, r.width))}px`;
        list.style.maxHeight = `${Math.round(Math.max(120, Math.min(240, (openUp ? above : below) - 10)))}px`;
        if (openUp) {
            list.style.top = "auto";
            list.style.bottom = `${Math.round(
                document.documentElement.clientHeight - r.top + scrollY + 2,
            )}px`;
        } else {
            list.style.bottom = "auto";
            list.style.top = `${Math.round(r.bottom + scrollY + 2)}px`;
        }
    }

    function openList() {
        if (locked || open) return;
        open = true;
        if (!list.isConnected) document.body.appendChild(list);
        list.classList.remove("hidden");
        reposition();
        const idx = options.findIndex((o) => o.frames === current.frames);
        activeIndex = idx >= 0 ? idx : 0;
        renderList("");
        document.addEventListener("pointerdown", onDocPointerDown, true);
        window.addEventListener("scroll", reposition, true);
        window.addEventListener("resize", reposition);
    }

    function closeList(restore = false) {
        if (!open) return;
        open = false;
        // Detach: the picker is rebuilt on every card re-render.
        list.remove();
        list.classList.add("hidden");
        document.removeEventListener("pointerdown", onDocPointerDown, true);
        window.removeEventListener("scroll", reposition, true);
        window.removeEventListener("resize", reposition);
        if (restore) setValue(current.sec);
    }

    function onDocPointerDown(e) {
        if (wrap.contains(e.target) || list.contains(e.target)) return;
        closeList(true);
    }

    function select(i) {
        const opt = options[i];
        if (!opt) return;
        current.sec = opt.sec;
        current.frames = opt.frames;
        input.value = opt.label;
        fitWidth();
        syncTitle();
        closeList();
        input.blur();
        if (onCommit) onCommit(opt.sec, opt.frames);
    }

    function moveActive(delta) {
        if (!open) openList();
        const visible = [...list.querySelectorAll(".bd-dur-opt[data-index]")].map(
            (el) => parseInt(el.dataset.index, 10),
        );
        if (!visible.length) return;
        const pos = visible.indexOf(activeIndex);
        const nextPos = Math.min(visible.length - 1, Math.max(0, (pos < 0 ? 0 : pos) + delta));
        activeIndex = visible[nextPos];
        renderList(input.value);
    }

    input.addEventListener("focus", () => {
        if (locked) return;
        input.select?.();
        openList();
    });
    input.addEventListener("click", (e) => {
        e.stopPropagation();
        if (locked) return;
        if (!open) openList();
    });
    input.addEventListener("input", () => {
        if (locked) return;
        if (!open) openList();
        activeIndex = -1;
        const firstMatch = renderList(input.value);
        if (firstMatch >= 0) activeIndex = firstMatch;
    });
    input.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (locked) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            moveActive(1);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            moveActive(-1);
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (activeIndex >= 0) select(activeIndex);
            else closeList(true);
        } else if (e.key === "Escape") {
            e.preventDefault();
            closeList(true);
            input.blur();
        }
    });
    input.addEventListener("blur", () => closeList(true));

    buildOptions();
    setValue(sec);
    if (locked) {
        input.readOnly = true;
        input.disabled = true;
        wrap.classList.add("bd-disabled");
    }

    const api = {
        el: wrap,
        input,
        getSec: () => current.sec,
        getFrames: () => current.frames,
        setSec: (nextSec) => setValue(nextSec),
        setContinuity: (nextContinuity, nextSec) => {
            const changed = !!nextContinuity !== current.continuity;
            current.continuity = !!nextContinuity;
            if (changed) buildOptions();
            setValue(nextSec != null ? nextSec : current.sec);
            if (open) renderList("");
        },
        setDisabled: (nextDisabled) => {
            locked = !!nextDisabled;
            input.disabled = locked;
            input.readOnly = locked;
            wrap.classList.toggle("bd-disabled", locked);
            if (locked) closeList();
        },
        destroy: () => {
            closeList();
            list.remove();
            wrap.remove();
        },
    };
    input.__bdDurCombo = api;
    return api;
}
