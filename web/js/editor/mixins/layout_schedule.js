/** layout_schedule mixin for the Director editor (layout_schedule).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { ensureDirectorNodeFitsContent, getDirectorUiHeight, healOversizedDirectorNode } from "../../core/editor_lifecycle.js";
import { DIRECTOR_MIN_WIDTH } from "../../core/layout_spec.js";
import { bindDomWidgetContentComputeSize, contentDomWidgetMinHeight, syncBatchPanelFillHeight } from "../../minimax_image_batch.js";
export const layout_scheduleMixin = {
    _observeViewportResize() {
        if (!this.viewport || typeof ResizeObserver === "undefined") return;
        this._resizeObserver?.disconnect();
        this._resizeObserver = new ResizeObserver(() => {
            if (this.isPlaying || this._pauseSettling) return;
            this.scheduleRender();
        });
        this._resizeObserver.observe(this.viewport);
        if (this.container && this.container !== this.viewport) {
            this._resizeObserver.observe(this.container);
        }
    },
    _measureDrawWidth() {
        if (this.isPlaying && this._playCanvasWidth > 0) return this._playCanvasWidth;
        if (this.getTimelineZoom() > 1) {
            const zoomed = this.canvas?.clientWidth || this.canvas?.offsetWidth || 0;
            if (zoomed > 0) return zoomed;
        }
        return this.viewport?.clientWidth
            || this.canvas?.clientWidth
            || this.canvas?.offsetWidth
            || this.container?.clientWidth
            || this.root?.clientWidth
            || 0;
    },
    /** Redraw after layout/zoom settles (first mount often measures before the node finishes sizing). */
    scheduleSettleRender() {
        this.scheduleRender();
        if (this._settleRenderTimer != null) return;
        this._settleRenderTimer = setTimeout(() => {
            this._settleRenderTimer = null;
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    if (!this.isPlaying) this.scheduleRender();
                });
            });
        }, 0);
        // Extra pass after ComfyUI node size / graph zoom finishes applying.
        clearTimeout(this._settleRenderLateTimer);
        this._settleRenderLateTimer = setTimeout(() => {
            this._settleRenderLateTimer = null;
            if (!this.isPlaying) this.scheduleRender();
        }, 100);
    },
    _capturePlayCanvasWidth() {
        const w = this.viewport?.clientWidth
            || this.container?.offsetWidth
            || this.node?.size?.[0]
            || DIRECTOR_MIN_WIDTH;
        if (w > 0) this._playCanvasWidth = w;
        return this._playCanvasWidth;
    },
    _lockPlayLayout() {
        this._capturePlayCanvasWidth();
    },
    _resetLayoutStyles() {
        if (this.isPlaying) return;
        for (const el of [this.container, this.root, this.viewport]) {
            if (!el) continue;
            el.style.removeProperty("width");
            el.style.removeProperty("min-width");
            el.style.removeProperty("max-width");
        }
        this._playCanvasWidth = 0;
        this.applyZoomWidth();
    },
    _releasePlayLayoutLock() {
        this._resetLayoutStyles();
    },
    getDirectorUiMinHeight() {
        return getDirectorUiHeight(this);
    },
    updateDomWidgetHeight(opts = {}) {
        const h = contentDomWidgetMinHeight(this) || getDirectorUiHeight(this);
        this.container?.style.setProperty("--comfy-widget-min-height", `${h}px`);
        if (this.container) this.container.style.minHeight = `${h}px`;
        // Content min only — never bake node.size / stretch into computeSize.
        bindDomWidgetContentComputeSize(this);
        const runActive = !!this.runStatusEl?.classList?.contains("active");
        // Grow only when content needs more room (e.g. mode switch). Never shrink
        // a user-enlarged node (#7). During live progress: never grow; heal runaway.
        if (!this.isPlaying) {
            if (runActive) healOversizedDirectorNode(this.node, this);
            else ensureDirectorNodeFitsContent(this.node, this);
        }
        syncBatchPanelFillHeight(this, {
            settle: opts.settle !== false && !runActive,
        });
    },
    /** Patch batch card `.running` without tearing down the list (progress path). */
    _syncBatchRunHighlight() {
        if (!this.isImageBatch?.() || !this.batchList) return;
        const runningIdx = this._runHighlightSeg;
        this.batchList.querySelectorAll(".bd-batch-card").forEach((card) => {
            const i = parseInt(card.dataset.batchIndex, 10);
            card.classList.toggle("running", Number.isFinite(i) && i === runningIdx);
        });
        this.batchPicker?.querySelectorAll?.(".bd-batch-pick").forEach((chip) => {
            const i = parseInt(chip.dataset.batchIndex, 10);
            chip.classList.toggle("running", Number.isFinite(i) && i === runningIdx);
        });
        this._syncR2vCardSelection?.();
    },
    scheduleRender() {
        if (this._renderPending) return;
        this._renderPending = true;
        this._resizeRaf = requestAnimationFrame(() => {
            this._renderPending = false;
            if (this.isPlaying) this.renderTimelineOnly();
            else this.render();
        });
    },
    onNodeResize() {
        if (this.isPlaying || this._pauseSettling) return;
        // Growable layout (no computeSize) → LiteGraph puts free space into computedHeight.
        bindDomWidgetContentComputeSize(this);
        this._resetLayoutStyles();
        this.applyZoomWidth();
        syncBatchPanelFillHeight(this);
        // Re-fill after LiteGraph finishes arranging widgets for the new node size.
        requestAnimationFrame(() => syncBatchPanelFillHeight(this));
        this.scheduleSettleRender();
    }
};
