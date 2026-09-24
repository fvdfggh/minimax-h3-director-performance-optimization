/** events mixin for the Director editor (events).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { coerceTimelineFps } from "../../core/dims.js";
import { parseTimeline, stopDomEvent } from "../../core/editor_lifecycle.js";
import { clamp } from "../../core/utils.js";
import { ensureFl2vTimeline, flushFl2vPromptDraft, openFl2vUpload, updateFl2vDetailUI } from "../../minimax_fl2v.js";
import { DEFAULT_MEGAPIXELS, clampMegapixels, getDirectorMode, normalizeRefImageSize, parseMegapixelsInput, resolveTaskKey } from "../../minimax_gen_timeline.js";
import { t, toggleLocale } from "../../minimax_i18n.js";
import { addImageBatchGroup, ensureImageBatchTimeline } from "../../minimax_image_batch.js";
import { bindPackActions } from "../../minimax_pack.js";
import { mountPromptImageMentions } from "../../minimax_prompt_mentions.js";
export const eventsMixin = {
    bindEvents() {
        const bind = (sel, fn) => {
            const el = this.root.querySelector(sel);
            if (!el) return;
            el.onclick = (e) => { stopDomEvent(e); fn(); };
        };
        bind('[data-a="video"]', () => this.pickVideoFile());
        bind('[data-a="video-existing"]', () => { void this.pickExistingVideoFile(); });
        bind('[data-a="fl2v-add-shot"]', () => openFl2vUpload(this));
        bind('[data-a="r2v-add-group"]', () => addImageBatchGroup(this));
        bind('[data-a="video-append"]', () => this.pickAppendVideoFile());
        bind('[data-a="split"]', () => this.splitAtFrame(this.currentFrame));
        bind('[data-a="equal"]', () => this.equalSplit());
        bind('[data-a="smart-split"]', () => { void this.smartSplit(); });
        bind('[data-a="del-split"]', () => this.deleteSelectedSplitPoint());
        bind('[data-a="run-select-toggle"]', () => this.toggleRunSelectMode());
        bind('[data-a="seg-export"]', () => { void this.openSegmentExportPicker(); });
        bind('[data-a="second-sample"]', () => { void this.openSecondSamplePicker(); });
        bind('[data-a="audio-extract"]', () => { void this.openAudioExtractPicker(); });
        bind('[data-a="del"]', () => this.deleteSelectedSegment());
        bind('[data-a="mode-global"]', () => this.setEditMode("global"));
        bind('[data-a="mode-segment"]', () => this.setEditMode("segment"));
        bind('[data-a="lang-toggle"]', () => toggleLocale());
        bind('[data-a="zoom-toggle"]', () => this.toggleTimelineZoom());
        bindPackActions(this);
        bind('[data-a="play"]', () => this.togglePlay());
        bind('[data-a="loop"]', () => this.toggleLoop());
        bind('[data-a="live-tae-preview"]', () => this.toggleLiveTaePreview());
        bind('[data-a="frame-prev"]', () => this.stepFrame(-1));
        bind('[data-a="frame-next"]', () => this.stepFrame(1));
        this.refreshLiveTaePreviewButton();
        this.updateLiveSamplePanel();

        this.seekBar.oninput = () => {
            this.seekToFrame(+this.seekBar.value, { fromUi: true });
        };
        if (this.frameInputEl) {
            const applyFrameInput = () => {
                const total = this.getTotalFrames();
                if (total < 1) return;
                const raw = parseInt(this.frameInputEl.value, 10);
                if (!Number.isFinite(raw)) {
                    this.frameInputEl.value = String(this.currentFrame + 1);
                    return;
                }
                // UI is 1-based; internal currentFrame is 0-based.
                this.seekToFrame(raw - 1, { fromUi: true });
            };
            this.frameInputEl.addEventListener("keydown", (e) => {
                e.stopPropagation();
                if (e.key === "Enter") {
                    e.preventDefault();
                    applyFrameInput();
                    this.frameInputEl.blur();
                } else if (e.key === "Escape") {
                    e.preventDefault();
                    this.frameInputEl.value = String(this.currentFrame + 1);
                    this.frameInputEl.blur();
                }
            });
            this.frameInputEl.addEventListener("change", applyFrameInput);
            this.frameInputEl.addEventListener("focus", () => {
                if (this.isPlaying) this._stopPlay();
                this.frameInputEl.select();
            });
        }
        if (this.stageBadge) {
            this.stageBadge.title = t("player.badgeJump");
            this.stageBadge.addEventListener("click", (e) => {
                e.stopPropagation();
                if (this.isPlaying) this._stopPlay();
                this.frameInputEl?.focus();
                this.frameInputEl?.select();
            });
        }
        if (this.zoomSlider) {
            const zoomMin = () => Number(this.zoomSlider.min) || 1;
            const zoomMax = () => Number(this.zoomSlider.max) || 10;
            const applySliderZoom = () => {
                if (!this.zoomEnabled) return;
                this.zoom = clamp(+this.zoomSlider.value, zoomMin(), zoomMax());
                this.applyZoomWidth();
                this.scheduleRender();
            };
            const zoomFromClientX = (clientX) => {
                const rect = this.zoomSlider.getBoundingClientRect();
                const min = zoomMin();
                const max = zoomMax();
                const t = rect.width > 1 ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0;
                return min + t * (max - min);
            };
            const scrubZoom = (e) => {
                this.zoomSlider.value = String(zoomFromClientX(e.clientX));
                applySliderZoom();
            };
            this.zoomSlider.oninput = applySliderZoom;
            const onZoomPointerDown = (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation?.();
                if (!this.zoomEnabled) return;
                if (e.button != null && e.button !== 0) return;
                this._zoomPointerId = e.pointerId;
                try { this.zoomSlider.setPointerCapture(e.pointerId); } catch { /* ignore */ }
                this.zoomSlider.focus({ preventScroll: true });
                scrubZoom(e);
            };
            const onZoomPointerMove = (e) => {
                if (this._zoomPointerId == null || e.pointerId !== this._zoomPointerId) return;
                e.preventDefault();
                e.stopPropagation();
                scrubZoom(e);
            };
            const onZoomPointerUp = (e) => {
                if (this._zoomPointerId == null || e.pointerId !== this._zoomPointerId) return;
                e.stopPropagation();
                this._zoomPointerId = null;
                try { this.zoomSlider.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
            };
            this.zoomSlider.addEventListener("pointerdown", onZoomPointerDown, true);
            this.zoomSlider.addEventListener("pointermove", onZoomPointerMove);
            this.zoomSlider.addEventListener("pointerup", onZoomPointerUp, true);
            this.zoomSlider.addEventListener("pointercancel", onZoomPointerUp, true);
            this.zoomSlider.addEventListener("mousedown", (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, true);
            this.zoomSlider.addEventListener("wheel", (e) => e.stopPropagation(), true);
        }
        this.viewport?.addEventListener("wheel", (e) => {
            if (this.getTimelineZoom() <= 1) return;
            e.stopPropagation();
            if (e.deltaX === 0 && e.deltaY !== 0) {
                e.preventDefault();
                this.viewport.scrollLeft += e.deltaY;
            }
        }, { passive: false });
        if (this.runSelectAllCb) {
            this.runSelectAllCb.onchange = (e) => {
                stopDomEvent(e);
                if (!this.isRunSelectEnabled()) return;
                this.setRunSelectionAll(this.runSelectAllCb.checked);
            };
        }
        this.globalTask.onchange = () => {
            this.onGlobalField("taskType", this.globalTask.value);
        };
        this.globalPrompt.oninput = () => this.onGlobalField("prompt", this.globalPrompt.value);
        // 「启用公共参数」/「收起公共参数」在 r2v 下已移除（公共素材成为常驻页），
        // 这里不再绑定任何点击行为，仅保留节点引用以防其他代码访问。

        if (this.continuousRefCb) {
            this.continuousRefCb.onchange = () => {
                this.timeline.global = this.timeline.global || { refs: [], referenceVideo: {} };
                this.timeline.global.continuousReference = !!this.continuousRefCb.checked;
                this.scheduleTimelineSync();
            };
        }
        this.segPrompt.oninput = () => this.onSegField("prompt", this.segPrompt.value);
        this.globalNegative.oninput = () => this.onNegativePrompt(this.globalNegative.value);
        this.segNegative.oninput = () => this.onNegativePrompt(this.segNegative.value);

        mountPromptImageMentions(this);

        this.outMode.onchange = () => this.onOutputField("mode", this.outMode.value);
        if (this.outAspect) {
            this.outAspect.onchange = () => this.onOutputField("aspectRatio", this.outAspect.value);
        }
        if (this.outMp) {
            // Do not coerce incomplete drafts ("0", "0.") — that snaps back to 0.4 mid-typing.
            const applyMp = ({ force = false } = {}) => {
                const parsed = parseMegapixelsInput(this.outMp.value);
                if (parsed == null) {
                    if (!force) return;
                    const restored = clampMegapixels(
                        this.timeline.output?.megapixels ?? DEFAULT_MEGAPIXELS,
                    );
                    this.outMp.value = String(restored);
                    this.onOutputField("megapixels", restored);
                    return;
                }
                this.onOutputField("megapixels", parsed);
            };
            this.outMp.onchange = () => applyMp({ force: true });
            this.outMp.onblur = () => applyMp({ force: true });
            this.outMp.oninput = () => {
                clearTimeout(this._mpInputTimer);
                this._mpInputTimer = setTimeout(() => applyMp({ force: false }), 280);
            };
            this.outMp.addEventListener("keydown", (e) => e.stopPropagation());
        }
        this.outLong.onchange = () => this.onOutputField("longEdge", +this.outLong.value);
        this.outW.onchange = () => this.onOutputField("width", +this.outW.value);
        this.outH.onchange = () => this.onOutputField("height", +this.outH.value);
        this.fpsInput.onchange = () => this.onFrameRateChanged(this.fpsInput.value);
        this.fpsInput.oninput = () => {
            clearTimeout(this._fpsInputTimer);
            this._fpsInputTimer = setTimeout(() => this.onFrameRateChanged(this.fpsInput.value), 350);
        };
        this.outMaxFrames.onchange = () => this.onOutputField("maxExportFrames", +this.outMaxFrames.value);
        this.outExportMode.onchange = () => this.onOutputField("exportMode", this.outExportMode.value);
        if (this.outAudioMode) {
            this.outAudioMode.onchange = () => this.onOutputField("audioMode", this.outAudioMode.value);
        }
        if (this.segRefImageSize) {
            this.segRefImageSize.onchange = () => {
                const seg = this.timeline.segments?.[this.selectedIndex];
                if (!seg) return;
                seg.refImageSize = normalizeRefImageSize(this.segRefImageSize.value);
                this.commit(true);
            };
        }
        if (this.segmentContinuityCb) {
            this.segmentContinuityCb.onchange = () => {
                this.onOutputField("continuityEnabled", this.segmentContinuityCb.checked);
                this.updateSegmentContinuityUI();
            };
        }
        if (this.segmentContinuityOverlap) {
            const applyOverlap = () => this.onOutputField("continuityOverlapFrames", +this.segmentContinuityOverlap.value);
            this.segmentContinuityOverlap.onchange = applyOverlap;
            this.segmentContinuityOverlap.oninput = applyOverlap;
            this.segmentContinuityOverlap.addEventListener("keydown", (e) => e.stopPropagation());
            this.segmentContinuityOverlap.addEventListener("keyup", (e) => e.stopPropagation());
        }
        if (this.segContinuityFromPrevCb) {
            this.segContinuityFromPrevCb.onchange = () => {
                const seg = this.timeline.segments?.[this.selectedIndex];
                if (!seg || this.selectedIndex <= 0) return;
                seg.continuityFromPrev = !!this.segContinuityFromPrevCb.checked;
                this.commit(true);
            };
            this.segContinuityFromPrevWrap?.setAttribute(
                "title",
                t("tooltip.segmentContinuityFromPrev"),
            );
        }

        this.genGlobalImg?.addEventListener("click", (e) => { stopDomEvent(e); this.pickGenSrcImage(true); });
        this.genSegImg?.addEventListener("click", (e) => { stopDomEvent(e); this.pickGenSrcImage(false); });
        this.root.querySelector('[data-r="global-refs-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRef(true);
        });
        this.root.querySelector('[data-r="seg-refs-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRef(false);
        });
        this.root.querySelector('[data-r="global-videos-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingR2vCommonVideo();
        });
        this.root.querySelector('[data-r="global-audios-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRefAudio(true);
        });
        this.root.querySelector('[data-r="seg-audios-pick"]')?.addEventListener("click", (e) => {
            stopDomEvent(e);
            void this.pickExistingRefAudio(false);
        });
        this.genDefaultFc?.addEventListener("change", () => this.onGenDefaultFcChange());
        this.genSegFc?.addEventListener("change", () => this.onGenSegFcChange());

        this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
        this.canvas.addEventListener("dblclick", (e) => {
            stopDomEvent(e);
            e.preventDefault();
            // fl2v: double-click still replaces the start frame. Other modes do not split.
            if (!this.isFl2vMode()) return;
            const { x, y } = this.getMousePos(e);
            const hit = this.hitTest(x, y);
            if (hit?.type === "segment" || hit?.type === "edge") {
                const idx = hit.index ?? this.selectedIndex;
                if (idx !== this.selectedIndex) flushFl2vPromptDraft(this);
                this.selectedIndex = idx;
                this.updateSelectionUI();
                updateFl2vDetailUI(this);
                this._fl2vUploadMode = "slot";
                this._fl2vSlotKind = "start";
                this._fl2vSlotShotIndex = idx;
                this.fl2vUi?.fileInput?.click();
            }
        });
        this.canvas.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            if (this.isFl2vMode()) return;
            this.addSplitAtMouse(e);
        });
        this._onMouseMove = (e) => this.onMouseMove(e);
        this._onMouseUp = () => this.onMouseUp();
        this._onCanvasHover = (e) => {
            if (this._drag || this.isPlaying) return;
            const { x, y } = this.getMousePos(e);
            const hit = this.hitTest(x, y);
            this.canvas.classList.remove("bd-grab");
            if (hit?.type === "run-check" || hit?.type === "split" || hit?.type === "continuity-joint") {
                this.canvas.style.cursor = "pointer";
                if (hit.type === "continuity-joint") {
                    this.canvas.title = t(
                        hit.on ? "tooltip.continuityJointOn" : "tooltip.continuityJointOff",
                        { a: hit.a, b: hit.b },
                    );
                } else {
                    this.canvas.title = "";
                }
            } else if (hit?.type === "edge") {
                // Edge drag is always horizontal (change start/length); keep ↔ cursor.
                this.canvas.style.cursor = "ew-resize";
                this.canvas.title = this.isFl2vMode()
                    ? t("tooltip.dragFl2vDuration")
                    : "";
            } else if (this.needsSourceVideoUpload?.() && (hit?.type === "segment" || hit?.type === "edge" || !hit)) {
                this.canvas.style.cursor = "pointer";
                this.canvas.title = t("canvas.clickUploadVideo");
            } else if (hit?.type === "segment" && (this.isFl2vMode() || this.usesBatchTimeline() || this.timeline.segments.length >= 2)) {
                this.canvas.classList.add("bd-grab");
                this.canvas.style.cursor = "";
                this.canvas.title = this.isFl2vMode()
                    ? t("tooltip.dragFl2vSwap")
                    : (this.isR2vBatch()
                        ? t("tooltip.dragR2vOrder")
                        : (this.usesBatchTimeline()
                            ? t("tooltip.dragPromptGroupOrder")
                            : t("tooltip.dragSegmentOrder")));
            } else {
                this.canvas.style.cursor = "";
                this.canvas.title = "";
            }
        };
        window.addEventListener("mousemove", this._onMouseMove);
        window.addEventListener("mouseup", this._onMouseUp);
        this.canvas.addEventListener("mousemove", this._onCanvasHover);
        this.canvas.addEventListener("mouseleave", () => {
            this.canvas.classList.remove("bd-grab");
            this.canvas.style.cursor = "";
            this.canvas.title = "";
        });

        this.root.addEventListener("mouseenter", () => { this._isHovering = true; });
        this.root.addEventListener("mouseleave", () => { this._isHovering = false; });
        this._onKeyDown = (e) => {
            if (!this._isHovering) return;
            const el = document.activeElement;
            const tag = el?.tagName;
            // Prompt editors are contenteditable DIVs (bd-token-editor), not TEXTAREA —
            // must ignore typing shortcuts there or Backspace deletes the asset group.
            if (
                tag === "INPUT"
                || tag === "TEXTAREA"
                || tag === "SELECT"
                || el?.isContentEditable
                || el?.closest?.('[contenteditable="true"], .bd-token-editor')
            ) {
                return;
            }
            if ((e.key === "Delete" || e.key === "Backspace") && this.timeline.segments.length >= 1) {
                // Split points only delete via the toolbar button; Delete removes segments.
                if (this.selectedSplitFrame != null) {
                    e.preventDefault();
                    return;
                }
                this.deleteSelectedSegment();
                e.preventDefault();
            } else if (e.code === "Space") {
                this.togglePlay(); e.preventDefault();
            } else if (e.key === "ArrowLeft") {
                this.stepFrame(e.shiftKey ? -10 : -1);
                e.preventDefault();
            } else if (e.key === "ArrowRight") {
                this.stepFrame(e.shiftKey ? 10 : 1);
                e.preventDefault();
            }
        };
        window.addEventListener("keydown", this._onKeyDown, true);

        this.root.addEventListener("dragover", (e) => e.preventDefault());
        this.root.addEventListener("drop", (e) => {
            e.preventDefault();
            // Slot-to-slot moves are handled on .bd-ref; don't also treat as new upload.
            const types = [...(e.dataTransfer?.types || [])];
            if (types.includes("application/x-minimax-ref-slot")) return;
            if (types.includes("application/x-minimax-fl2v-slot")) return;
            if (types.includes("application/x-minimax-fl2v-shot")) return;
            if (e.target.closest?.(".bd-ref, .bd-batch-ref, .bd-batch-src, .bd-batch-video, .bd-batch-audio, .bd-batch-videos, .bd-batch-audios, .bd-fl2v-slot, .bd-fl2v-shot")) return;
            const f = e.dataTransfer.files?.[0];
            if (f?.type.startsWith("video/")) this.loadVideoFile(f);
            else if (f?.type.startsWith("image/")) {
                if (this.isImageBatch?.() && e.target.closest?.(".bd-batch-ref")) return;
                if (this.isImageBatch?.()) return;
                this.addRefFromFile(f, this.getRefTarget());
            }
        });
    },
    /** Replace the whole timeline with an imported director pack (Opt or upstream). */
    applyImportedTimeline(timeline, widgets = {}) {
        const data = timeline && typeof timeline === "object" ? timeline : {};
        const opts = widgets && typeof widgets === "object" ? widgets : {};
        // Replace, do not merge: drop in-memory drafts from the previous task so
        // later t2v/r2v/v2v switches restore pack workspaces, not stale slots.
        this._batchWsMem = {};
        this._videoWsMem = {};
        this._lastOutputWasBatchFixed = false;
        this._legacyFrames = [];
        this._clearPreviewVideos?.(true);
        const taskType = opts.task_type || opts.taskType || data.global?.taskType || "";
        if (this.taskTypeWidget && taskType) this.taskTypeWidget.value = taskType;
        if (this.globalTask && taskType) this.globalTask.value = taskType;
        for (const name of ["steps", "sampler", "scheduler", "cfg", "shift_video", "shift_audio", "seed"]) {
            if (opts[name] == null || opts[name] === "") continue;
            const w = this.widget(name);
            if (w) w.value = opts[name];
        }
        const out = data.output && typeof data.output === "object" ? data.output : {};
        if (this.widthWidget && out.width) this.widthWidget.value = out.width;
        if (this.heightWidget && out.height) this.heightWidget.value = out.height;
        if (this.frameRateWidget && (data.frameRate || out.frameRate)) {
            this.frameRateWidget.value = data.frameRate || out.frameRate;
        }
        if (this.refMaxWidget && (data.refMaxSize || out.longEdge)) {
            this.refMaxWidget.value = data.refMaxSize || out.longEdge;
        }
        if (this.globalPromptWidget && data.global?.prompt != null) {
            this.globalPromptWidget.value = data.global.prompt;
        }
        if (this.timelineWidget) this.timelineWidget.value = JSON.stringify(data);
        const initTotal = Math.max(0, parseInt(this.totalFramesWidget?.value || data.totalFrames || 124, 10));
        const initFps = coerceTimelineFps(this.frameRateWidget?.value || data.frameRate || 24);
        this.timeline = parseTimeline(this.timelineWidget?.value, initTotal, initFps);
        this.syncFrameRateUI?.(this.timeline.frameRate);
        const prevMode = this._directorMode;
        const prevKey = this._taskKey;
        const nextMode = getDirectorMode(this.taskTypeWidget?.value || taskType);
        this._directorMode = nextMode;
        this._taskKey = resolveTaskKey(this.taskTypeWidget?.value || taskType);
        if (nextMode === "video") {
            this.restoreVideoFromTimeline();
        } else if (nextMode === "prompt_batch" || nextMode === "image_batch") {
            ensureImageBatchTimeline(this);
        } else if (nextMode === "fl2v") {
            ensureFl2vTimeline(this);
        } else {
            this.ensureGenTimeline();
        }
        this.applyTaskLayout(prevMode, prevKey);
        this.populateTaskSelect(this.globalTask, this.taskTypeWidget?.value);
        this.setEditMode(this.timeline.editMode || "global");
        this.selectedIndex = 0;
        this.updateModeUI?.();
        this.updateSelectionUI();
        this.applyZoomWidth?.();
        this.commit(true, { syncTimeline: true });
        this._externalGroupsSyncSig = null;
        this.syncExternalGroupsTimeline?.();
        this.scheduleSettleRender?.();
        this.updateDomWidgetHeight?.();
    }
};
