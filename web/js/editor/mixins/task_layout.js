/** task_layout mixin for MiniMaxH3DirectorOptEditor.
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { clamp, uid } from "../../core/utils.js";
import { MiniMaxH3DirectorOptEditor } from "../editor.js";
import { ensureFl2vTimeline, setFl2vToolbar, updateFl2vDetailUI, updateFl2vToolbarBtns } from "../../minimax_fl2v.js";
import { DEFAULT_ASPECT_RATIO, DEFAULT_MEGAPIXELS, MAX_GEN_FRAMES, MINIMAX_CANVAS_MULTIPLE, NO_VIDEO_UPLOAD_TASKS, defaultFrameCount, genLayoutHint, getDirectorMode, isCustomAspectRatio, isVideoBatchTask, minFrameCount, normalizeAspectRatioLabel, resolveTaskKey, sumFrameCounts, taskUsesReferenceAudios, taskUsesReferenceImages, taskUsesReferenceVideo } from "../../minimax_gen_timeline.js";
import { t } from "../../minimax_i18n.js";
import { ensureImageBatchTimeline, setR2vToolbar, setToolbarDisabledForBatch, updateR2vToolbarBtns } from "../../minimax_image_batch.js";
export const task_layoutMixin = {
    ensureGenTimeline() {
        const key = this.getTaskKey();
        this.timeline.gen = this.timeline.gen || {};
        const defFc = defaultFrameCount(key);
        if (!this.timeline.segments?.length || !sumFrameCounts(this.timeline.segments)) {
            this.timeline.segments = [{
                id: uid(), start: 0, length: defFc, frameCount: defFc,
                prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
            }];
        }
        for (const seg of this.timeline.segments) {
            if (seg.frameCount == null) seg.frameCount = seg.length ?? defFc;
            seg.genImage = seg.genImage || { imageFile: seg.imageFile || "" };
        }
        this.timeline.global = this.timeline.global || { refs: [] };
        this.timeline.global.genImage = this.timeline.global.genImage || { imageFile: "" };
        if (this.isGenBlank()) {
            this.timeline.output = this.timeline.output || {};
            this.timeline.output.mode = "fixed";
        }
        this.normalizeGenSegments();
    },
    normalizeGenSegments() {
        const key = this.getTaskKey();
        const minFc = minFrameCount(key);
        let start = 0;
        const fixed = [];
        for (const seg of [...this.timeline.segments]) {
            let fc = clamp(parseInt(seg.frameCount ?? seg.length, 10) || defaultFrameCount(key), minFc, MAX_GEN_FRAMES);
            fixed.push({
                ...seg,
                start,
                length: fc,
                frameCount: fc,
                refs: seg.refs || [],
                genImage: seg.genImage || { imageFile: "" },
            });
            start += fc;
        }
        if (!fixed.length) {
            const fc = defaultFrameCount(key);
            fixed.push({
                id: uid(), start: 0, length: fc, frameCount: fc,
                prompt: "", taskType: "", refs: [], genImage: { imageFile: "" },
            });
        }
        this.timeline.segments = fixed;
        this.timeline.totalFrames = start || fixed[0].frameCount;
        this.selectedIndex = clamp(this.selectedIndex, 0, fixed.length - 1);
    },
    /** rv2v (and video-timeline tasks with refs) use the polished r2v-like asset stage. */
    usesRv2vRefStyle(taskKey = this.getTaskKey()) {
        const key = resolveTaskKey(taskKey);
        // r2v shared panel reuses the polished image/audio slot chrome.
        return key === "rv2v" || key === "vrc2v" || key === "vi2v" || key === "r2v";
    },
    /** v2v prompt-only video edit — full-width polished prompt stage. */
    usesV2vPromptStyle(taskKey = this.getTaskKey()) {
        const key = resolveTaskKey(taskKey);
        return key === "v2v" || key === "mv2v";
    },
    syncRv2vRefLayoutClasses({ hideTimeline = false, seg = null } = {}) {
        const globalKey = this.getTaskKey();
        const segKey = resolveTaskKey(
            seg?.taskType || this.timeline.global?.taskType || this.globalTask?.value || globalKey,
        );
        const globalRefStyle = !hideTimeline && this.usesRv2vRefStyle(globalKey);
        const segRefStyle = !hideTimeline && this.usesRv2vRefStyle(segKey);
        const globalV2vStyle = !hideTimeline && this.usesV2vPromptStyle(globalKey);
        const segV2vStyle = !hideTimeline && this.usesV2vPromptStyle(segKey);

        this.globalPanel?.classList.toggle("bd-rv2v-panel", globalRefStyle);
        this.segmentPanel?.classList.toggle("bd-rv2v-panel", segRefStyle);
        this.globalPanel?.classList.toggle("bd-v2v-panel", globalV2vStyle);
        this.segmentPanel?.classList.toggle("bd-v2v-panel", segV2vStyle);

        this.globalPromptLayout?.classList.toggle("bd-rv2v-layout", globalRefStyle);
        this.segPromptLayout?.classList.toggle("bd-rv2v-layout", segRefStyle);
        this.globalPromptLayout?.classList.toggle("bd-v2v-layout", globalV2vStyle);
        this.segPromptLayout?.classList.toggle("bd-v2v-layout", segV2vStyle);

        for (const wrap of [this.globalRefsImagesWrap, this.globalRefAudiosWrap, this.globalRefVideosWrap]) {
            wrap?.classList.toggle("bd-r2v-section", globalRefStyle);
        }
        for (const wrap of [this.segRefsImagesWrap, this.segRefAudiosWrap]) {
            wrap?.classList.toggle("bd-r2v-section", segRefStyle);
        }
        const gLabel = this.root.querySelector('[data-r="global-refs-label"]');
        if (gLabel) {
            const key = globalRefStyle ? "batch.r2v.sectionPictures" : "panel.refImages";
            gLabel.textContent = t(key);
            gLabel.setAttribute("data-i18n", key);
        }
        const sLabel = this.root.querySelector('[data-r="seg-refs-label"]');
        if (sLabel) {
            const key = segRefStyle ? "batch.r2v.sectionPictures" : "panel.segmentRefImages";
            sLabel.textContent = t(key);
            sLabel.setAttribute("data-i18n", key);
        }
    },
    updateReferenceImageVisibility({ hideTimeline = false, seg = null } = {}) {
        const globalKey = this.getTaskKey();
        const showGlobalRefs = !hideTimeline && taskUsesReferenceImages(globalKey);
        const showGlobalRefAudios = !hideTimeline && taskUsesReferenceAudios(globalKey);
        // r2v common panel: multi-slot ref videos (distinct from ads2v single referenceVideo).
        const showGlobalR2vVideos = !hideTimeline && this.usesR2vCommonPanel();
        const showGlobalRefVideo = !hideTimeline && taskUsesReferenceVideo(globalKey);

        this.globalRefsCol?.classList.toggle(
            "hidden",
            !showGlobalRefs && !showGlobalRefVideo && !showGlobalRefAudios && !showGlobalR2vVideos,
        );
        this.globalRefsImagesWrap?.classList.toggle("hidden", !showGlobalRefs);
        this.globalRefVideosWrap?.classList.toggle("hidden", !showGlobalR2vVideos);
        this.globalRefAudiosWrap?.classList.toggle("hidden", !showGlobalRefAudios);
        this.globalRefVideoCol?.classList.toggle("hidden", !showGlobalRefVideo);
        if (this.globalPanelTitle) {
            let titleKey = "panel.globalPromptOnly";
            if (this.usesR2vCommonPanel()) {
                titleKey = "panel.r2vCommonParams";
            } else if (showGlobalRefVideo) {
                titleKey = "panel.globalPromptAndRefVideo";
            } else if (showGlobalRefs || showGlobalRefAudios) {
                titleKey = showGlobalRefAudios
                    ? "panel.globalPromptAndRefsMedia"
                    : "panel.globalPromptAndRefs";
            }
            this.globalPanelTitle.textContent = t(titleKey);
            this.globalPanelTitle.setAttribute("data-i18n", titleKey);
        }
        this.syncR2vCommonCollapse();

        const segKey = resolveTaskKey(
            seg?.taskType || this.timeline.global?.taskType || this.globalTask?.value || globalKey,
        );
        const showSegRefs = !hideTimeline && taskUsesReferenceImages(segKey);
        const showSegRefAudios = !hideTimeline && taskUsesReferenceAudios(segKey);
        const showSegRefVideo = !hideTimeline && taskUsesReferenceVideo(segKey);
        this.segRefsCol?.classList.toggle(
            "hidden",
            !showSegRefs && !showSegRefVideo && !showSegRefAudios,
        );
        this.segRefsImagesWrap?.classList.toggle("hidden", !showSegRefs);
        this.segRefAudiosWrap?.classList.toggle("hidden", !showSegRefAudios);
        this.segRefVideoCol?.classList.toggle("hidden", !showSegRefVideo);
        const showContinuousRef = !hideTimeline
            && this.isGlobalMode()
            && showGlobalRefVideo
            && globalKey === "ads2v";
        this.continuousRefWrap?.classList.toggle("hidden", !showContinuousRef);
        if (this.continuousRefCb) {
            this.continuousRefCb.checked = !!this.timeline.global?.continuousReference;
        }
        this.syncRv2vRefLayoutClasses({ hideTimeline, seg });
        if (showGlobalRefVideo || showSegRefVideo) this.renderRefVideoSlot();
        if (showGlobalR2vVideos) this.renderR2vCommonVideoSlots();
        if (showGlobalRefAudios || showSegRefAudios) this.renderRefAudioSlots();
    },
    _stashFl2vWorkspace() {
        const shots = this.timeline.shots || [];
        const segs = this.timeline.segments || [];
        const keys = this.timeline.keyframes || [];
        if (!shots.length && !segs.length && !keys.length) return;
        this.timeline.fl2vWorkspace = {
            shots: JSON.parse(JSON.stringify(shots)),
            segments: JSON.parse(JSON.stringify(segs)),
            keyframes: JSON.parse(JSON.stringify(keys)),
            durationSec: this.timeline.durationSec,
            totalFrames: this.timeline.totalFrames,
            selectedIndex: this.selectedIndex,
            runSelectEnabled: !!this.timeline.runSelectEnabled,
            runSelection: Array.isArray(this.timeline.runSelection)
                ? [...this.timeline.runSelection]
                : [],
            output: this.timeline.output
                ? JSON.parse(JSON.stringify(this.timeline.output))
                : undefined,
        };
    },
    _restoreFl2vWorkspace() {
        const ws = this.timeline.fl2vWorkspace;
        if (!ws) return false;
        const hasShots = Array.isArray(ws.shots) && ws.shots.length;
        const hasSegs = Array.isArray(ws.segments) && ws.segments.length;
        const hasKeys = Array.isArray(ws.keyframes) && ws.keyframes.length;
        if (!hasShots && !hasSegs && !hasKeys) return false;
        if (hasShots) this.timeline.shots = JSON.parse(JSON.stringify(ws.shots));
        if (hasSegs) this.timeline.segments = JSON.parse(JSON.stringify(ws.segments));
        if (hasKeys) this.timeline.keyframes = JSON.parse(JSON.stringify(ws.keyframes));
        if (ws.durationSec != null) this.timeline.durationSec = ws.durationSec;
        if (ws.totalFrames != null) this.timeline.totalFrames = ws.totalFrames;
        if (ws.selectedIndex != null) this.selectedIndex = ws.selectedIndex;
        if (ws.runSelectEnabled != null) this.timeline.runSelectEnabled = !!ws.runSelectEnabled;
        if (Array.isArray(ws.runSelection)) this.timeline.runSelection = [...ws.runSelection];
        if (ws.output && typeof ws.output === "object") {
            this.timeline.output = { ...(this.timeline.output || {}), ...JSON.parse(JSON.stringify(ws.output)) };
        }
        this.timeline.fl2vWorkspace = null;
        return true;
    },
    applyTaskLayout(prevMode, prevTaskKey) {
        const mode = this.getDirectorMode();
        const prev = prevMode || "video";
        const wasBatch = prev === "prompt_batch" || prev === "image_batch";
        const isBatch = mode === "prompt_batch";
        const wasFl2v = prev === "fl2v";
        const isFl2v = mode === "fl2v";
        const wasGen = prev !== "video" && prev !== "prompt_batch" && prev !== "image_batch" && prev !== "fl2v";
        const isGen = mode !== "video" && mode !== "prompt_batch" && mode !== "fl2v";
        const currentKey = this.getTaskKey();
        const stashBatchKey = prevTaskKey && isVideoBatchTask(prevTaskKey) ? prevTaskKey : null;
        const stashVideoKey = prevTaskKey && getDirectorMode(prevTaskKey) === "video" ? prevTaskKey : null;

        if (this.isPlaying) this._stopPlay();

        if (isFl2v) {
            if (prev === "video") {
                this._stashVideoWorkspace(stashVideoKey);
                this._clearLiveRunSelection();
            } else if (wasBatch) {
                this._stashBatchWorkspace(stashBatchKey);
                this._clearLiveRunSelection();
            }
            if (!this._restoreFl2vWorkspace()) {
                ensureFl2vTimeline(this);
                this._clearLiveRunSelection();
            } else {
                ensureFl2vTimeline(this);
            }
        } else if (isBatch) {
            if (!wasBatch) {
                if (wasFl2v) {
                    this._stashFl2vWorkspace();
                    this._clearLiveRunSelection();
                }
                // Keep v2v/rv2v video + segments so switching back can restore them.
                // Run-select is per workspace: stash video's, then clear live so i2v/batch
                // does not inherit「选择运行」from rv2v.
                if (prev === "video") {
                    this._stashVideoWorkspace(stashVideoKey);
                    this._clearLiveRunSelection();
                    this._clearVideoState();
                }
                this._switchToBatchTaskWorkspace(null, currentKey);
            } else if (prevTaskKey && prevTaskKey !== currentKey) {
                // t2v / i2v / r2v used to share one segment list — isolate per task.
                this._switchToBatchTaskWorkspace(prevTaskKey, currentKey);
            }
            ensureImageBatchTimeline(this);
        } else if (isGen) {
            if (wasBatch) {
                this._stashBatchWorkspace(stashBatchKey);
                this._clearLiveRunSelection();
            }
            if (wasFl2v) {
                this._stashFl2vWorkspace();
                this._clearLiveRunSelection();
            }
            if (!wasGen && !wasBatch && !wasFl2v) {
                if (prev === "video") {
                    this._stashVideoWorkspace(stashVideoKey);
                    this._clearLiveRunSelection();
                }
                const key = this.getTaskKey();
                const defFc = defaultFrameCount(key);
                const keepPrompt = this.timeline.global?.prompt || "";
                this.timeline.segments = [{
                    id: uid(),
                    start: 0,
                    length: defFc,
                    frameCount: defFc,
                    prompt: keepPrompt,
                    taskType: "",
                    refs: [],
                    genImage: { imageFile: "" },
                }];
            }
            this.ensureGenTimeline();
        } else if (prev !== "video") {
            // Leaving batch/gen/fl2v for video — stash before video restore.
            if (wasBatch) {
                this._stashBatchWorkspace(stashBatchKey);
                this._clearLiveRunSelection();
            }
            if (wasFl2v) {
                this._stashFl2vWorkspace();
                this._clearLiveRunSelection();
            }
            this.timeline.timelineMode = "video";
            this._switchToVideoTaskWorkspace(stashVideoKey, currentKey);
        } else if (prevTaskKey && prevTaskKey !== currentKey) {
            // v2v ↔ rv2v share director mode "video" but keep separate workspaces.
            this._switchToVideoTaskWorkspace(prevTaskKey, currentKey);
        }
        this.timeline.timelineMode = mode;
        this._directorMode = mode;
        const taskKey = currentKey;
        this._taskKey = taskKey;

        const isR2v = isBatch && taskKey === "r2v";
        const showBatchTrack = isBatch && isVideoBatchTask(taskKey);
        // fl2v / t2v / i2v / r2v use the main timeline track; image batch + gen hide it.
        const hideTimeline = (isBatch && !showBatchTrack) || isGen;
        const hideVideoUpload = hideTimeline || NO_VIDEO_UPLOAD_TASKS.has(taskKey) || isR2v;
        const showBatchExport = (isBatch && isVideoBatchTask(taskKey)) || isFl2v;
        // t2v / i2v / r2v: never show source-video upload (fl2v keeps "上传图片").
        this.btnVideo?.classList.toggle("hidden", (hideVideoUpload && !isFl2v) || isR2v);
        this.btnVideoExisting?.classList.toggle("hidden", hideVideoUpload || isFl2v || isR2v);
        this.btnVideoAppend?.classList.toggle("hidden", hideVideoUpload || isFl2v || isR2v);
        // Playback / seek / zoom are for source-video (v2v). fl2v / t2v / batch have no source clip.
        this.controlsBar?.classList.toggle("hidden", hideTimeline || isBatch || isFl2v);
        this.boundsEl?.classList.toggle("hidden", hideTimeline || isBatch || isFl2v);
        this.timecodeEl?.classList.toggle("hidden", hideTimeline || isBatch || isFl2v);
        this.viewport?.classList.toggle("hidden", isBatch && !showBatchTrack);
        this.tlZoomWrap?.classList.toggle("hidden", isBatch && !showBatchTrack);
        this.updateStageVisibility();
        this.updateLiveSamplePanel();
        this.syncExternalGroupsTimeline();
        // r2v keeps bd-split visible so the shared「公共参数」panel can sit above batch cards.
        this.root.querySelector(".bd-split")?.classList.toggle("hidden", (isBatch && !isR2v) || isFl2v);
        this.batchPanel?.classList.toggle("hidden", !isBatch);
        this.root?.classList.toggle("bd-batch-fill", !!isBatch);
        this.fl2vUi?.root?.classList.toggle("hidden", !isFl2v);
        this.fl2vTotalWrap?.classList.toggle("hidden", !isFl2v);
        if (isFl2v) {
            setR2vToolbar(this, false);
            setFl2vToolbar(this, true);
            setToolbarDisabledForBatch(this, false);
            // Re-apply fl2v-specific disables after clearing batch disables.
            setFl2vToolbar(this, true);
        } else if (isR2v) {
            setFl2vToolbar(this, false);
            setToolbarDisabledForBatch(this, false);
            setR2vToolbar(this, true);
            if (this.btnVideo) {
                this.btnVideo.textContent = t("toolbar.uploadVideo");
                this.btnVideo.setAttribute("data-i18n", "toolbar.uploadVideo");
            }
            updateFl2vToolbarBtns(this);
        } else {
            setFl2vToolbar(this, false);
            setR2vToolbar(this, false);
            setToolbarDisabledForBatch(this, isBatch);
            if (this.btnVideo) {
                this.btnVideo.textContent = t("toolbar.uploadVideo");
                this.btnVideo.setAttribute("data-i18n", "toolbar.uploadVideo");
            }
            const del = this.root?.querySelector('[data-a="del"]');
            if (del) {
                if (showBatchTrack) {
                    const externalLocked = !!(this.hasExternalI2vGroups?.() || this.hasExternalR2vGroups?.());
                    if (externalLocked) {
                        del.classList.add("hidden");
                        del.disabled = true;
                    } else {
                        del.disabled = false;
                        del.classList.remove("bd-disabled", "hidden");
                        del.textContent = t("toolbar.deleteSelectedGroup");
                        del.setAttribute("data-i18n", "toolbar.deleteSelectedGroup");
                        del.setAttribute("data-i18n-title", "tooltip.deleteSelectedPromptGroup");
                        del.title = t("tooltip.deleteSelectedPromptGroup");
                    }
                } else {
                    del.textContent = t("toolbar.deleteSegment");
                    del.setAttribute("data-i18n", "toolbar.deleteSegment");
                    del.setAttribute("data-i18n-title", "tooltip.deleteSegment");
                }
            }
            updateFl2vToolbarBtns(this);
            updateR2vToolbarBtns(this);
        }

        // Side ref panels stay hidden for most batch modes (refs live in cards).
        // r2v shows a collapsible「公共参数」bar; refs only when enabled/expanded.
        this.syncR2vCommonCollapse();
        this.updateReferenceImageVisibility({
            hideTimeline: (isBatch && !this.isR2vCommonEnabled()) || isGen,
        });

        const showGenImg = mode === "gen_image";
        this.genGlobalImg?.classList.toggle("hidden", !showGenImg || !this.isGlobalMode());
        this.genSegImg?.classList.toggle("hidden", !showGenImg || this.isGlobalMode());
        this.genGlobalFcRow?.classList.toggle("hidden", !isGen || !this.isGlobalMode());
        this.genSegFcRow?.classList.toggle("hidden", !isGen || this.isGlobalMode());

        if (isBatch || isGen || isFl2v || NO_VIDEO_UPLOAD_TASKS.has(taskKey)) {
            this.timeline.output = this.timeline.output || {};
            this.timeline.output.mode = "fixed";
            if (isBatch && !isVideoBatchTask(taskKey)) this.timeline.output.exportMode = "all";
            if (!this.timeline.output.aspectRatio) this.timeline.output.aspectRatio = DEFAULT_ASPECT_RATIO;
            else this.timeline.output.aspectRatio = normalizeAspectRatioLabel(this.timeline.output.aspectRatio);
            if (this.timeline.output.megapixels == null) this.timeline.output.megapixels = DEFAULT_MEGAPIXELS;
            if (this.timeline.output.multiple == null) this.timeline.output.multiple = MINIMAX_CANVAS_MULTIPLE;
            if (isCustomAspectRatio(this.timeline.output.aspectRatio)) {
                this.applyCustomResolution();
            } else {
                this.applyResolutionSelector();
            }
            this.updateOutputModeUI();
        } else if (this.outMode) {
            this.outMode.disabled = false;
            // Video edit (v2v/rv2v): prefer long-edge so ultrawide sources are not
            // center-cropped into a leftover 16:9 fixed canvas from batch modes.
            this.timeline.output = this.timeline.output || {};
            if (!this.timeline.output.mode || this.timeline.output.mode === "fixed") {
                const fromBatchFixed = this._lastOutputWasBatchFixed;
                if (fromBatchFixed || !this.timeline.output.mode) {
                    this.timeline.output.mode = "long_edge";
                    if (!this.timeline.output.longEdge) this.timeline.output.longEdge = 848;
                }
            }
            this._lastOutputWasBatchFixed = false;
            this.updateOutputModeUI();
        }
        if (isBatch || isGen || isFl2v) this._lastOutputWasBatchFixed = true;

        if (this.outHint) {
            const isVideoEdit = taskKey === "v2v" || taskKey === "rv2v";
            const showHint = isGen || isBatch || isFl2v || isVideoEdit;
            this.outHint.classList.toggle("hidden", !showHint);
            this.outHint.textContent = showHint ? genLayoutHint(this.getTaskKey()) : "";
        }
        const isVideoEditTask = taskKey === "v2v" || taskKey === "rv2v";
        // 声音控制对所有任务可见：生成类任务可“生成声音/静音”，编辑类任务额外可用“使用原声”
        this.outAudioWrap?.classList.remove("hidden");
        if (this.outAudioMode) {
            const srcOpt = this.outAudioMode.querySelector('option[value="source"]');
            if (srcOpt) srcOpt.disabled = !isVideoEditTask;
        }
        if (this.outExportMode) {
            this.outExportMode.disabled = (isBatch || isFl2v) && !showBatchExport;
            this.outExportMode.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
            this.outExportMode.previousElementSibling?.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
        }
        if (this.outMaxFrames) {
            this.outMaxFrames.disabled = (isBatch || isFl2v) && !showBatchExport;
            this.outMaxFrames.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
            this.outMaxFrames.previousElementSibling?.classList.toggle("hidden", (isBatch || isFl2v) && !showBatchExport);
        }

        if ((isGen || isBatch || isFl2v) && prev === "video") {
            this.currentFrame = 0;
        }
        this.updateVideoNameLabel();
        if (isFl2v) {
            this.timeline.editMode = "segment";
            ensureFl2vTimeline(this);
            this.updateSelectionUI();
            updateFl2vDetailUI(this);
            this.updateVideoNameLabel();
        } else if (isBatch) {
            this.timeline.editMode = "segment";
            this.renderImageBatchGroups();
            // Must refresh globalPanel display — r2v common params stay display:none
            // if we only ran updateModeUI in the non-batch branch (segment → r2v).
            this.updateModeUI();
            if (showBatchTrack) {
                this.updateSelectionUI();
                this._syncR2vCardSelection();
            }
        } else {
            this.updateModeUI();
            this.updateSelectionUI();
        }
        this.updateDomWidgetHeight();
        this.syncOutputUIFromTimeline();
        this.seekBar.max = Math.max(0, this.getTotalFrames() - 1);
        if (!isBatch || showBatchTrack) this.scheduleRender();
        this.scheduleTimelineSync();
        this.updateRunSelectUI();
    }
};
