/** live_playback mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { healOversizedDirectorNode, syncDirectorNodeSize } from "../../core/editor_lifecycle.js";
import { deletedSourceRanges } from "../../core/frame_map.js";
import { clamp } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { isVideoBatchTask } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
export const live_playbackMixin = {
    isLiveTaePreviewEnabled() {
        return this.timeline?.liveTaePreview !== false;
    },
    /** fl2v / v2v / rv2v (and aliases): show dedicated live-sample panel when toggle is on. */
    needsLiveSamplePanel() {
        if (!this.isLiveTaePreviewEnabled()) return false;
        if (this.isImageBatch?.()) return false;
        if (this.isFl2vMode?.()) return true;
        const key = this.getTaskKey?.() || "";
        return key === "v2v" || key === "mv2v" || key === "ads2v"
            || key === "rv2v" || key === "vrc2v" || key === "vi2v";
    },
    toggleLiveTaePreview() {
        this.timeline.liveTaePreview = !this.isLiveTaePreviewEnabled();
        this.refreshLiveTaePreviewButton();
        this.updateLiveSamplePanel();
        this.scheduleTimelineSync();
        this.updateDomWidgetHeight?.();
        syncDirectorNodeSize(this.node, this);
    },
    refreshLiveTaePreviewButton() {
        const btn = this.root?.querySelector('[data-a="live-tae-preview"]');
        if (!btn) return;
        const on = this.isLiveTaePreviewEnabled();
        btn.classList.toggle("active", on);
        btn.textContent = t("toolbar.liveTaePreview");
        btn.title = on ? t("tooltip.liveTaePreviewOn") : t("tooltip.liveTaePreviewOff");
        btn.setAttribute("data-i18n", "toolbar.liveTaePreview");
        btn.removeAttribute("data-i18n-title");
    },
    _clearEmbeddedLiveLayoutClasses() {
        this.globalPromptLayout?.classList.remove("bd-v2v-with-live", "bd-rv2v-with-live");
        this.segPromptLayout?.classList.remove("bd-v2v-with-live", "bd-rv2v-with-live");
    },
    _activePromptLayout() {
        return this.isGlobalMode?.() ? this.globalPromptLayout : this.segPromptLayout;
    },
    _placeLiveSamplePanel() {
        const panel = this.liveSampleEl;
        if (!panel) return;

        if (this.isFl2vMode?.() && this.fl2vUi?.workbench && this.fl2vUi?.shotsEl) {
            this._clearEmbeddedLiveLayoutClasses();
            if (this._liveSampleHost !== "fl2v" || panel.parentElement !== this.fl2vUi.workbench) {
                this.fl2vUi.workbench.insertBefore(panel, this.fl2vUi.shotsEl);
                this._liveSampleHost = "fl2v";
            }
            return;
        }

        const layout = this._activePromptLayout();

        // v2v: preview sits to the right of the prompt column.
        if (this.usesV2vPromptStyle?.() && this.isLiveTaePreviewEnabled() && layout) {
            if (panel.parentElement !== layout) layout.appendChild(panel);
            this.globalPromptLayout?.classList.toggle("bd-v2v-with-live", layout === this.globalPromptLayout);
            this.segPromptLayout?.classList.toggle("bd-v2v-with-live", layout === this.segPromptLayout);
            this.globalPromptLayout?.classList.remove("bd-rv2v-with-live");
            this.segPromptLayout?.classList.remove("bd-rv2v-with-live");
            this._liveSampleHost = "v2v";
            return;
        }

        // rv2v: preview under the prompt (same stack as r2v right column).
        if (this.usesRv2vRefStyle?.() && this.isLiveTaePreviewEnabled() && layout) {
            const promptCol = layout.querySelector(".bd-prompt-col");
            if (promptCol) {
                if (panel.parentElement !== promptCol) promptCol.appendChild(panel);
                this.globalPromptLayout?.classList.toggle("bd-rv2v-with-live", layout === this.globalPromptLayout);
                this.segPromptLayout?.classList.toggle("bd-rv2v-with-live", layout === this.segPromptLayout);
                this.globalPromptLayout?.classList.remove("bd-v2v-with-live");
                this.segPromptLayout?.classList.remove("bd-v2v-with-live");
                this._liveSampleHost = "rv2v";
                return;
            }
        }

        this._clearEmbeddedLiveLayoutClasses();
        if (this.outputBarEl && (this._liveSampleHost !== "main" || panel.parentElement !== this.mainBody)) {
            this.outputBarEl.insertAdjacentElement("afterend", panel);
            this._liveSampleHost = "main";
        }
    },
    updateLiveSamplePanel() {
        const panel = this.liveSampleEl;
        if (!panel) return;
        const show = this.needsLiveSamplePanel();
        this._placeLiveSamplePanel();
        panel.classList.toggle("hidden", !show);
        if (!show) {
            panel.classList.remove("receiving");
            return;
        }
        if (!this._liveSampleB64) {
            this.liveSampleImg?.classList.add("hidden");
            this.liveSampleEmpty?.classList.remove("hidden");
            this.liveSampleBadge?.classList.add("hidden");
            if (this.liveSampleMeta) this.liveSampleMeta.textContent = t("liveSample.idleHint");
        }
    },
    clearLiveSamplePreview() {
        this._liveSampleB64 = "";
        this._liveSampleStep = null;
        this._liveSampleTotal = null;
        this._liveSampleSeg = null;
        this.liveSampleEl?.classList.remove("receiving");
        if (this.liveSampleImg) {
            this.liveSampleImg.removeAttribute("src");
            this.liveSampleImg.classList.add("hidden");
        }
        this.liveSampleEmpty?.classList.remove("hidden");
        this.liveSampleBadge?.classList.add("hidden");
        if (this.liveSampleMeta) this.liveSampleMeta.textContent = t("liveSample.idleHint");
    },
    setLiveSamplePreview(detail = {}) {
        if (!this.needsLiveSamplePanel()) return;
        const b64 = detail.image_b64 || detail.imageB64 || "";
        if (!b64) return;
        this._placeLiveSamplePanel();
        this.liveSampleEl?.classList.remove("hidden");
        this._liveSampleB64 = b64;
        this._liveSampleStep = detail.step ?? null;
        this._liveSampleTotal = detail.total_steps ?? detail.totalSteps ?? null;
        this._liveSampleSeg = detail.segment_index ?? detail.segmentIndex ?? null;

        const src = b64.startsWith("data:") ? b64 : `data:image/jpeg;base64,${b64}`;
        if (this.liveSampleImg) {
            this.liveSampleImg.src = src;
            this.liveSampleImg.classList.remove("hidden");
        }
        this.liveSampleEmpty?.classList.add("hidden");
        this.liveSampleEl?.classList.toggle("receiving", !!detail.live);

        const step = this._liveSampleStep;
        const total = this._liveSampleTotal;
        const seg = this._liveSampleSeg;
        let badge = "";
        if (step && total) badge = t("batch.generatingStep", { step, total });
        else if (detail.live) badge = t("batch.generating");
        if (this.liveSampleBadge) {
            this.liveSampleBadge.textContent = badge;
            this.liveSampleBadge.classList.toggle("hidden", !badge);
        }
        if (this.liveSampleMeta) {
            const unit = this.isFl2vMode?.() ? t("unit.shot") : t("unit.segment");
            const segLabel = (seg != null && seg !== "")
                ? t("liveSample.segmentHint", { unit, n: Number(seg) + 1 })
                : "";
            this.liveSampleMeta.textContent = detail.live
                ? (segLabel || t("liveSample.sampling"))
                : (segLabel || t("liveSample.done"));
        }
    },
    /** Keep timeline / 素材组 selection on the segment the run is currently on. */
    _followRunSelection(timelineSeg1Based) {
        const segs = this.timeline?.segments || [];
        if (!segs.length) return;
        const idx = clamp(Math.round(Number(timelineSeg1Based) || 1) - 1, 0, segs.length - 1);
        if (this.selectedIndex === idx) return;
        this.selectedIndex = idx;
        this.updateSelectionUI();
        const seg = segs[idx];
        if (seg && Number.isFinite(seg.start) && !this.isPlaying) {
            const f = Math.max(0, Math.round(Number(seg.start) || 0));
            this.currentFrame = f;
            if (this.seekBar) this.seekBar.value = f;
            this._syncStagePreview?.(f, { force: true });
        }
        this.scheduleRender();
    },
    setRunProgress(detail) {
        if (!this.runStatusEl) return;
        const timelineTotal = this.timeline?.segments?.length || 0;
        const runTotal = Math.max(detail.segment_total || this.getRunProgressSegmentTotal(), 1);
        const runSeg = Math.max(1, detail.segment || 1);
        const timelineSeg = detail.timeline_segment ?? runSeg;
        const partialRun = !!detail.partial_run
            || (this.isRunSelectEnabled?.() && runTotal < timelineTotal);
        const phaseLabel = detail.phase_label || detail.phase || t("run.phase.default");
        const overallPct = detail.overall_max > 0
            ? Math.round((100 * detail.overall_value) / detail.overall_max)
            : 0;
        const phasePct = detail.phase_max > 0
            ? Math.round((100 * detail.phase_value) / detail.phase_max)
            : 0;
        const remain = Math.max(0, runTotal - runSeg);

        if (detail.phase === "finish") {
            this.runStatusEl.className = "bd-run-status done";
            this.runTitleEl.textContent = t("run.titleDone");
            this.runDetailEl.textContent = runTotal
                ? (this.isImageBatch()
                    ? (isVideoBatchTask(this.getTaskKey())
                        ? t("run.detailDoneVideos", { n: runTotal })
                        : t("run.detailDoneImages", { n: runTotal }))
                    : (partialRun
                        ? t("run.detailDoneSegmentsPartial", { n: runTotal })
                        : t("run.detailDoneSegments", { n: runTotal })))
                : t("run.detailDoneGeneric");
            this.runOverallEl.style.width = "100%";
            this.runPhaseEl.style.width = "100%";
            this._runHighlightSeg = -1;
            this._runProgressSegKey = null;
            this._followRunSelection(timelineSeg);
            this.updateRunSelectUI();
            if (this.isImageBatch()) this.renderImageBatchGroups();
            else this.scheduleRender();
            return;
        }

        this.runStatusEl.className = "bd-run-status active";
        // Hide the pre-run "将运行 N 段" chip while progress is live — it sits
        // under the title in the same green accent and reads as a layout glitch.
        this.runSelectBar?.classList.add("hidden");
        this._runHighlightSeg = timelineSeg - 1;
        this._followRunSelection(timelineSeg);
        let title;
        if (detail.phase === "plan") {
            title = runTotal > 1 ? t("run.titlePlanning", { n: runTotal, phase: phaseLabel }) : phaseLabel;
        } else if (this.isImageBatch()) {
            // Partial run: show timeline card number (e.g. group 4), not compact run order.
            title = partialRun
                ? t("run.titleBatchGroupPartial", {
                    timeline: timelineSeg, i: runSeg, n: runTotal, phase: phaseLabel,
                })
                : t("run.titleBatchGroup", { i: runSeg, n: runTotal, phase: phaseLabel });
        } else if (partialRun) {
            title = t("run.titleSegmentPartial", { timeline: timelineSeg, i: runSeg, n: runTotal, phase: phaseLabel });
        } else {
            title = t("run.titleSegment", { i: runSeg, n: runTotal, phase: phaseLabel });
        }
        if (phasePct > 0 && detail.phase !== "plan") {
            title += ` · ${phasePct}%`;
        }
        this.runTitleEl.textContent = title;
        const parts = [];
        if (detail.frames_label) parts.push(detail.frames_label);
        if (detail.task_key) parts.push(detail.task_key);
        parts.push(t("run.detailOverall", { pct: overallPct }));
        if (runTotal > 1) {
            parts.push(this.isImageBatch()
                ? t("run.detailRemainingGroups", { n: remain })
                : t("run.detailRemainingSegments", { n: remain }));
        }
        if (partialRun && timelineTotal > runTotal) {
            parts.push(t("run.detailTimelineTotal", { n: timelineTotal }));
        }
        this.runDetailEl.textContent = parts.join(" · ");
        this.runOverallEl.style.width = `${overallPct}%`;
        this.runPhaseEl.style.width = `${phasePct}%`;
        // Do NOT syncDirectorNodeSize / full batch rebuild every tick — that was the
        // cross-mode (t2v/i2v/r2v/…) infinite-height feedback loop. Status has a fixed
        // min-height; live previews patch in place via minimax_director_opt_preview.
        const segKey = `${timelineSeg}|${detail.phase}|${runSeg}`;
        const segChanged = this._runProgressSegKey !== segKey;
        this._runProgressSegKey = segKey;
        if (this.isImageBatch()) {
            this._syncBatchRunHighlight();
            if (segChanged) healOversizedDirectorNode(this.node, this);
        } else if (segChanged) {
            this.scheduleRender();
            healOversizedDirectorNode(this.node, this);
        }
    },
    clearRunProgress(title, detail) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status idle";
        this.runTitleEl.textContent = title || t("run.titleIdle");
        this.runDetailEl.textContent = detail || t("run.detailIdle");
        this.runOverallEl.style.width = "0%";
        this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this._runProgressSegKey = null;
        this.updateRunSelectUI();
        if (this.isImageBatch()) this.renderImageBatchGroups();
        else this.scheduleRender();
    },
    setRunError(message) {
        if (!this.runStatusEl) return;
        this.runStatusEl.className = "bd-run-status error";
        this.runTitleEl.textContent = t("run.titleError");
        this.runDetailEl.textContent = message || t("run.detailError");
        if (this.runOverallEl) this.runOverallEl.style.width = "0%";
        if (this.runPhaseEl) this.runPhaseEl.style.width = "0%";
        this._runHighlightSeg = -1;
        this._runProgressSegKey = null;
        this.updateRunSelectUI();
        this.scheduleRender();
    },
    _stopPlay() {
        this.isPlaying = false;
        this._playHandoff = false;
        this._nativePlayFailed = false;
        this._pauseSettling = true;
        cancelAnimationFrame(this._playRaf);
        this._playRaf = null;
        this.stageVideo?.pause();
        this.root.querySelector('[data-a="play"]').textContent = "▶";
        this._resizeObserver?.disconnect();

        const w = this._playCanvasWidth;
        this._releasePlayLayoutLock();

        if (w) this._drawTimelineCanvas(w);
        this._updateTimelineDom({ skipSeek: true });
        this._syncStagePreview(this.currentFrame, { force: true });

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (+this.seekBar.value !== this.currentFrame) {
                    this.seekBar.value = this.currentFrame;
                }
                this._observeViewportResize();
                const drawW = this._measureDrawWidth() || this.viewport?.clientWidth || w;
                if (drawW) this._drawTimelineCanvas(drawW);
                this._syncStagePreview(this.currentFrame, { force: true });
                this._pauseSettling = false;
            });
        });
    },
    async _beginNativePlay() {
        const total = this.getTotalFrames();
        if (total < 1) return;
        if (this.currentFrame >= total) this.currentFrame = 0;
        await this._ensureStageReadyForFrame(this.currentFrame);
        if (!this.isPlaying) return;
        const v = this.stageVideo;
        if (!v) return;
        try {
            await v.play();
        } catch {
            // Native play blocked/failed — keep isPlaying but drive via frame clock.
            this._nativePlayFailed = true;
        }
    },
    async _advanceNativePlayToNextClipOrEnd() {
        if (this._playHandoff || !this.isPlaying) return;
        this._playHandoff = true;
        try {
            const total = this.getTotalFrames();
            const range = this._logicalRangeForClip(this._stageClipIndex);
            const next = range.end < total ? range.end : -1;
            if (next >= 0) {
                this.currentFrame = next;
                await this._beginNativePlay();
                return;
            }
            if (this.isLooping) {
                this.currentFrame = 0;
                await this._beginNativePlay();
                return;
            }
            this.currentFrame = Math.max(0, total - 1);
            this._stopPlay();
        } finally {
            this._playHandoff = false;
        }
    },
    togglePlay() {
        if (this.isPlaying) {
            this._stopPlay();
            return;
        }
        const total = this.getTotalFrames();
        if (total < 1) return;

        this.isPlaying = true;
        this._nativePlayFailed = false;
        this.root.querySelector('[data-a="play"]').textContent = "⏸";
        this._lockPlayLayout();
        this._resizeObserver?.disconnect();

        if (this.currentFrame >= total) this.currentFrame = 0;
        this.renderTimelineOnly();
        this._updateTimelineDom();

        const useNative = !this._legacyFrames.length && !!this.stageVideo;
        if (useNative) {
            this._beginNativePlay();
        } else {
            this._syncStagePreview(this.currentFrame, { force: true });
        }

        const tick = () => {
            if (!this.isPlaying) return;
            const fps = Math.max(0.001, this.getFrameRate());

            if (useNative && this.stageVideo && !this._nativePlayFailed) {
                const v = this.stageVideo;
                const clipIndex = this._stageClipIndex >= 0 ? this._stageClipIndex : 0;
                const range = this._logicalRangeForClip(clipIndex);
                const lastLogical = Math.max(range.start, range.end - 1);
                const lastTime = this.getFrameMapEntry(lastLogical).frame / fps;
                const atMappedEnd = v.currentTime >= Math.max(0, lastTime - 0.04);
                const hasTimelineEdits = !!(
                    this.getFrameMap().length
                    || deletedSourceRanges(this.timeline.video || {}).length
                );
                // With deletes, file duration still includes removed tails — trust mapped end.
                const atMediaEnd = !hasTimelineEdits && (
                    v.ended || (v.duration > 0 && v.currentTime >= v.duration - 0.04)
                );

                if ((atMappedEnd || atMediaEnd) && !v.seeking && !this._playHandoff) {
                    this.currentFrame = lastLogical;
                    this.renderTimelineOnly();
                    this._updateTimelineDom();
                    this._advanceNativePlayToNextClipOrEnd();
                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                    return;
                }

                if (!v.paused) {
                    const srcFrame = Math.max(0, Math.round(v.currentTime * fps));
                    let logical = this._logicalFromStageTime(clipIndex, v.currentTime);
                    const jumpToKept = () => {
                        const nextLogical = this._nextLogicalAfterSourceFrame(clipIndex, srcFrame);
                        if (nextLogical >= 0) {
                            const nextSrc = this.getFrameMapEntry(nextLogical).frame;
                            try { v.currentTime = nextSrc / fps; } catch { /* seek race */ }
                            return nextLogical;
                        }
                        return -1;
                    };
                    // Sparse deleted gap, or mid/leading gap vs mapped source.
                    if (logical < 0) {
                        const next = jumpToKept();
                        if (next < 0) {
                            this.currentFrame = lastLogical;
                            this.renderTimelineOnly();
                            this._updateTimelineDom();
                            this._advanceNativePlayToNextClipOrEnd();
                            if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                            return;
                        }
                        logical = next;
                    } else {
                        const mapped = this.getFrameMapEntry(logical);
                        if (mapped.clip === clipIndex && mapped.frame !== srcFrame) {
                            // leading gap (mapped > src) or mid gap (mapped < src)
                            if (mapped.frame > srcFrame || mapped.frame < srcFrame) {
                                const next = mapped.frame > srcFrame ? logical : jumpToKept();
                                if (next < 0) {
                                    this.currentFrame = clamp(logical, 0, total - 1);
                                    this.renderTimelineOnly();
                                    this._updateTimelineDom();
                                    this._advanceNativePlayToNextClipOrEnd();
                                    if (this.isPlaying) this._playRaf = requestAnimationFrame(tick);
                                    return;
                                }
                                if (mapped.frame > srcFrame) {
                                    try { v.currentTime = mapped.frame / fps; } catch { /* seek race */ }
                                }
                                logical = next;
                            }
                        }
                    }
                    this.currentFrame = clamp(logical, 0, total - 1);
                    this.renderTimelineOnly();
                    const now = performance.now();
                    if (now - this._lastSeekUiMs > 66) {
                        this._updateTimelineDom();
                        this._lastSeekUiMs = now;
                    }
                }
            } else {
                // Legacy embedded frames (or native play unavailable): step by logical frame.
                this.currentFrame += 1;
                if (this.currentFrame >= total) {
                    if (this.isLooping) this.currentFrame = 0;
                    else {
                        this.currentFrame = total - 1;
                        this._stopPlay();
                        return;
                    }
                }
                this.renderTimelineOnly();
                this._syncStagePreview(this.currentFrame, { force: true });
                const now = performance.now();
                if (now - this._lastSeekUiMs > 80) {
                    this._updateTimelineDom();
                    this._lastSeekUiMs = now;
                }
            }
            this._playRaf = requestAnimationFrame(tick);
        };
        this._playRaf = requestAnimationFrame(tick);
    }
};
