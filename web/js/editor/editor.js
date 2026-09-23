



import { coerceTimelineFps } from "../core/dims.js";
import { hideWidget, parseTimeline } from "../core/editor_lifecycle.js";


import { HIDDEN_WIDGETS, RULER_H, SEG_LABEL_H, TRACK_H } from "../core/layout_spec.js";


import { uploadToInput, uploadToInputSmart } from "../core/upload.js";
import { relPath } from "../core/utils.js";


import { refViewUrl, videoRelativePath } from "./urls.js";

import { MAX_REFERENCE_AUDIOS, MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, getDirectorMode, refAudioLabel, refImageLabel, refVideoLabel, resolveTaskKey } from "../minimax_gen_timeline.js";
import { onLocaleChange, t } from "../minimax_i18n.js";
import { bindDomWidgetContentComputeSize, bindR2vMediaPlayback, ensureImageBatchTimeline, formatMediaDuration, rebaseR2vGroupSlotsForCommon, syncBatchPanelFillHeight, wireMediaDuration } from "../minimax_image_batch.js";

import { refreshPromptTokenEditors } from "../minimax_prompt_mentions.js";
import { hasDuplicateReferenceAudio, prepareLocalReferenceAudio } from "../minimax_ref_audio.js";
import { canvasMixin } from "./mixins/canvas.js";
import { interactionMixin } from "./mixins/interaction.js";
import { dom_shellMixin } from "./mixins/dom_shell.js";
import { eventsMixin } from "./mixins/events.js";
import { timeline_payloadMixin } from "./mixins/timeline_payload.js";
import { run_selectionMixin } from "./mixins/run_selection.js";
import { export_pickersMixin } from "./mixins/export_pickers.js";
import { external_groupsMixin } from "./mixins/external_groups.js";
import { layout_scheduleMixin } from "./mixins/layout_schedule.js";
import { fieldsMixin } from "./mixins/fields.js";
import { modesMixin } from "./mixins/modes.js";
import { workspacesMixin } from "./mixins/workspaces.js";
import { task_layoutMixin } from "./mixins/task_layout.js";
import { ref_video_slotMixin } from "./mixins/ref_video_slot.js";
import { media_thumbsMixin } from "./mixins/media_thumbs.js";
import { video_loadMixin } from "./mixins/video_load.js";
import { segments_modelMixin } from "./mixins/segments_model.js";
import { frame_map_ioMixin } from "./mixins/frame_map_io.js";
import { task_uiMixin } from "./mixins/task_ui.js";
import { output_uiMixin } from "./mixins/output_ui.js";
import { widget_syncMixin } from "./mixins/widget_sync.js";
import { stage_previewMixin } from "./mixins/stage_preview.js";
import { thumbnailsMixin } from "./mixins/thumbnails.js";
import { dialogsMixin } from "./mixins/dialogs.js";
import { media_pickerMixin } from "./mixins/media_picker.js";
import { segment_editMixin } from "./mixins/segment_edit.js";
import { live_playbackMixin } from "./mixins/live_playback.js";

export class MiniMaxH3DirectorOptEditor {
    constructor(node, container, domWidget) {
        this.node = node;
        this.container = container;
        this.domWidget = domWidget;
        this.zoom = 1;
        this.zoomEnabled = false;
        this.selectedIndex = 0;
        /** @type {number|null} Selected editable split-point frame (logical). */
        this.selectedSplitFrame = null;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.isLooping = false;
        this._playRaf = null;
        this._drag = null;
        this._previewSegments = null;
        this._edgeSnapshot = null;
        this._isHovering = false;
        this._thumbCache = new Map();
        this._thumbPending = new Set();
        this._seekChain = Promise.resolve();
        this._legacyFrames = [];
        this._storageWidth = 0;
        this._storageHeight = 0;
        this._previewVideo = null;
        this._previewVideos = new Map();
        this._thumbCanvas = null;
        this._syncTimer = null;
        this._resizeRaf = null;
        this._renderPending = false;
        this._settleRenderTimer = null;
        this._settleRenderLateTimer = null;
        this._promptRenderTimer = null;
        this._lastSeekUiMs = 0;
        this._playCanvasWidth = 0;
        this._pauseSettling = false;
        this._runHighlightSeg = -1;
        this._modalEl = null;
        this._modalKeyHandler = null;
        this._drawWidth = 0;
        this._reorderDropRank = -1;
        this._reorderFromRank = -1;
        this.canvasHeight = RULER_H + SEG_LABEL_H + TRACK_H;
        this._stageClipIndex = -1;
        this._stageSyncMs = 0;
        this._playHandoff = false;

        for (const w of node.widgets || []) {
            if (HIDDEN_WIDGETS.includes(w.name)) hideWidget(w);
        }

        this.timelineWidget = this.widget("timeline_data");
        this.totalFramesWidget = this.widget("total_frames");
        this.frameRateWidget = this.widget("frame_rate");
        this.taskTypeWidget = this.widget("task_type");
        this.globalPromptWidget = this.widget("global_prompt");
        this.negativePromptWidget = null;
        this.widthWidget = this.widget("width");
        this.heightWidget = this.widget("height");
        this.refMaxWidget = this.widget("ref_max_size");

        const initTotal = Math.max(0, parseInt(this.totalFramesWidget?.value || 124, 10));
        const initFps = coerceTimelineFps(this.frameRateWidget?.value || 24);
        this.timeline = parseTimeline(this.timelineWidget?.value, initTotal, initFps);
        this.buildDOM();
        this.bindEvents();
        this._unsubLocale = onLocaleChange(() => this.applyLocale());
        this.applyLocale();
        this._directorMode = getDirectorMode(this.taskTypeWidget?.value);
        this._taskKey = resolveTaskKey(this.taskTypeWidget?.value);
        if (this._directorMode === "video") {
            this.restoreVideoFromTimeline();
        } else if (this._directorMode === "prompt_batch" || this._directorMode === "image_batch") {
            ensureImageBatchTimeline(this);
        } else {
            this.ensureGenTimeline();
        }
        this.applyTaskLayout(this._directorMode);

        this.updateDomWidgetHeight();
        this.applyZoomWidth();
        this.syncFromWidgets();
        this.updateModeUI();
        this.updateSelectionUI();
        this.commit(true, { syncTimeline: false });
        this._observeViewportResize();
        this.syncExternalGroupsTimeline();
        this.scheduleSettleRender();
    }





    /**
     * Push a Director-card prompt edit into the matching external Group node
     * widget so execution (and the next sync) don't revive stale graph text.
     */



    /**
     * CSS layout width for the timeline bitmap.
     * Must NOT use getBoundingClientRect — ComfyUI graph zoom transforms inflate/deflate
     * that value while width:100% still follows clientWidth, and object-fit:fill then
     * stretches segment thumbnails.
     */

















































    /**
     * Re-base「选择运行」after the group at ``removedIndex`` is removed.
     *
     * ``runSelection`` is a plain list of array positions, so removing a group in
     * the middle used to hand its tick to the next group and silently drop the
     * last one. Everything above the removed position shifts down by one; the
     * removed position itself is dropped.
     */

    /**
     * Re-base「选择运行」after a group moves from ``fromRank`` to ``toRank``
     * (splice-out then splice-in). Without this the ticks stay on the old
     * positions and end up selecting whichever group landed there.
     */

    /**
     * Ask the backend to drop the cache files of the group at ``index``.
     *
     * Fire-and-forget: the cache is content-addressed now, so a failed call only
     * leaves orphaned files behind — the next run's slot sync removes them.
     */



    // ---------------------------------------------------------------------
    // 对齐下段 (align-to-next) availability
    // ---------------------------------------------------------------------


    /**
     * Refresh which segments may enable「对齐下段」.
     *
     * Asked of the backend because under「选择运行」the plan index is the
     * compact run order, so the frontend cannot infer "the next segment" from
     * the card list. Best-effort: on failure every segment stays disabled.
     */

    /**
     * Whether segment ``index`` may tick「对齐下段」: the next segment must hold
     * a cached AV latent. Unticked by default — this is an opt-in, cache-driven
     * middle-out mode.
     */






    // ---------------------------------------------------------------------
    // 分段导出 (segment export)
    // ---------------------------------------------------------------------






    // ---------------------------------------------------------------------
    // 二次采样 (second sample) — 复用分段导出弹窗结构，但写入 secondSample，
    // 执行时走节点 execute 的二采分支（需已加载 model / upscale_model / VAE）。
    // ---------------------------------------------------------------------













    /**
     * Turn the persisted「分段导出」flag back off.
     *
     * Export is a one-shot action, but ``resolveSegmentExport`` stores
     * ``enabled`` on ``timeline.output`` where it survives re-renders and page
     * reloads. Left set, every later run exports instead of generating — the run
     * silently becomes an export-only pass (no sampling, just a few seconds).
     * Mode/indices are kept so the picker reopens with the last selection.
     */












    /**
     * 方案B: 让「分段导出 / 二次采样」相关 UI 与「选择运行」状态彻底解耦。
     * 卡片进入 run-skipped(未选运行)时, 不应把卡片内/工具栏里的导出·二采元素
     * 一并灰化或禁用 —— 它们的可用与否只由各自功能的缓存/启用状态决定。
     * 这里作为 CSS 豁免之外的双保险(强制 opacity/disabled 不受 run-skipped 影响)。
     */







































































    /**
     * Shared params are always on in r2v — there is no「启用公共参数」toggle any
     * more: the shared asset page is a permanent page of the paginator, so the
     * runtime merge (concat prompt + merge refs) must always see it.
     */


    /**
     * Stable workflow id used by every cache path (``<root>/<slug>/node_<id>``).
     * Read-only routes（分段导出状态 / 二采状态 / segment_clip）must use this —
     * the hidden ``workflow_name`` widget is only synced at queue time, so
     * reading it here can still yield "" and point at the bare ``node_<id>`` dir.
     */
















































































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











    /**
     * Touching clip pairs in visual (time) order. `rightIndex` is the array index
     * whose `continuityFromPrev` flag owns this joint.
     */







    /**
     * fl2v edge handles: top half → previous clip's right edge;
     * bottom half → next clip's left edge.
     */




    /**
     * While dragging a segment edge, keep group seconds / toolbar / output preview in sync
     * without rebuilding the whole batch DOM (that would break the drag).
     */































    renderRefSlots(refs, box, isGlobal) {
        if (!box) return;
        box.innerHTML = "";
        const target = isGlobal
            ? this.timeline.global
            : this.timeline.segments[this.selectedIndex];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const wrap = isGlobal ? this.globalRefsImagesWrap : this.segRefsImagesWrap;
        const countEl = isGlobal ? this.globalRefsCount : this.segRefsCount;
        const PIC_STEP = 3;
        const PIC_SLOTS = MAX_REFERENCE_IMAGES;

        let filled = 0;
        let highestFilled = -1;
        for (const r of refs || []) {
            const idx = Number(r.index ?? r.slot);
            const has = !!(r?.imageFile || r?.imageB64);
            if (!has || !Number.isFinite(idx) || idx < 0 || idx >= PIC_SLOTS) continue;
            filled += 1;
            highestFilled = Math.max(highestFilled, idx);
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${PIC_SLOTS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-refs-pick"]' : '[data-r="seg-refs-pick"]',
            filled >= PIC_SLOTS,
        );

        if (!this._rv2vPicsVisible) this._rv2vPicsVisible = {};
        const visKey = isGlobal ? "global" : `seg:${target?.id ?? this.selectedIndex}`;
        const minVisible = highestFilled >= 0
            ? Math.min(PIC_SLOTS, Math.ceil((highestFilled + 1) / PIC_STEP) * PIC_STEP)
            : PIC_STEP;
        let visible = polished
            ? (Number(this._rv2vPicsVisible[visKey]) || PIC_STEP)
            : PIC_SLOTS;
        if (polished) {
            visible = Math.max(PIC_STEP, Math.min(PIC_SLOTS, visible));
            if (visible < minVisible) visible = minVisible;
            this._rv2vPicsVisible[visKey] = visible;
        }

        for (let i = 0; i < PIC_SLOTS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref";
            if (polished && i >= visible) el.classList.add("bd-r2v-pic-hidden");
            el.dataset.refSlot = String(i);
            el.dataset.refKind = "image";
            el.dataset.refIndex = String(i);
            el.dataset.refScope = isGlobal ? "global" : "seg";
            const label = refImageLabel(i);
            el.title = t("ref.slotTitle", { label });
            const ref = (refs || []).find((r) => Number(r.index ?? r.slot) === i);
            const tag = document.createElement("span");
            tag.className = polished ? "cap" : "bd-ref-tag";
            tag.textContent = label;
            el.appendChild(tag);
            if (ref?.imageFile) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = refViewUrl(ref.imageFile);
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            } else if (ref?.imageB64) {
                el.classList.add("has-img");
                const img = document.createElement("img");
                img.src = ref.imageB64.startsWith("data:") ? ref.imageB64 : `data:image/png;base64,${ref.imageB64}`;
                img.draggable = false;
                el.appendChild(img);
                if (polished) {
                    const dot = document.createElement("span");
                    dot.className = "dot";
                    el.appendChild(dot);
                }
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRef(target, i);
                };
                el.appendChild(x);
            }
            this._bindRefSlotDnD(el, target, i, isGlobal);
            el.onclick = () => {
                if (this._refDragMoved) {
                    this._refDragMoved = false;
                    return;
                }
                this.pickRef(target, i, isGlobal);
            };
            box.appendChild(el);
        }

        wrap?.querySelectorAll(".bd-r2v-pics-toggle").forEach((btn) => btn.remove());
        if (polished && wrap) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "bd-r2v-pics-toggle";
            const syncToggleLabel = () => {
                if (visible < PIC_SLOTS) {
                    const next = Math.min(PIC_STEP, PIC_SLOTS - visible);
                    toggle.textContent = t("batch.r2v.expandPics", { n: next });
                } else {
                    toggle.textContent = t("batch.r2v.collapsePics");
                }
            };
            syncToggleLabel();
            toggle.onclick = (e) => {
                e.stopPropagation();
                if (visible < PIC_SLOTS) {
                    visible = Math.min(PIC_SLOTS, visible + PIC_STEP);
                } else {
                    visible = Math.max(PIC_STEP, minVisible);
                }
                this._rv2vPicsVisible[visKey] = visible;
                box.querySelectorAll(".bd-ref").forEach((el, i) => {
                    el.classList.toggle("bd-r2v-pic-hidden", i >= visible);
                });
                syncToggleLabel();
                this.updateDomWidgetHeight?.();
            };
            wrap.appendChild(toggle);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    _bindRefSlotDnD(el, target, slotIndex, isGlobal) {
        const hasImg = el.classList.contains("has-img");
        el.draggable = hasImg;
        el.addEventListener("dragstart", (e) => {
            if (!hasImg) {
                e.preventDefault();
                return;
            }
            this._refDragMoved = false;
            const payload = JSON.stringify({
                scope: isGlobal ? "global" : "seg",
                segIndex: isGlobal ? -1 : this.selectedIndex,
                from: slotIndex,
            });
            e.dataTransfer.setData("application/x-minimax-ref-slot", payload);
            e.dataTransfer.setData("text/plain", payload);
            e.dataTransfer.effectAllowed = "move";
        });
        el.addEventListener("dragend", () => {
            // click may fire after dragend; keep suppress for one tick
            setTimeout(() => { this._refDragMoved = false; }, 0);
        });
        el.addEventListener("dragover", (e) => {
            const types = e.dataTransfer?.types || [];
            if (![...types].includes("application/x-minimax-ref-slot") && ![...types].includes("Files")) {
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = [...types].includes("application/x-minimax-ref-slot")
                ? "move"
                : "copy";
        });
        el.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const raw = e.dataTransfer.getData("application/x-minimax-ref-slot")
                || e.dataTransfer.getData("text/plain");
            if (raw) {
                try {
                    const data = JSON.parse(raw);
                    const scope = isGlobal ? "global" : "seg";
                    if (data.scope !== scope) return;
                    if (!isGlobal && data.segIndex !== this.selectedIndex) return;
                    this._refDragMoved = true;
                    this.moveRefSlot(target, Number(data.from), slotIndex, isGlobal);
                    return;
                } catch (_) { /* fall through to file drop */ }
            }
            const f = e.dataTransfer.files?.[0];
            if (f?.type?.startsWith("image/")) {
                this.addRefFromFile(f, target, slotIndex, isGlobal);
            }
        });
    }

    moveRefSlot(target, fromIndex, toIndex, isGlobal) {
        if (!target || fromIndex === toIndex) return;
        const refs = [...(target.refs || [])];
        const fromRef = refs.find((r) => Number(r.index ?? r.slot) === fromIndex);
        if (!fromRef) return;
        const toRef = refs.find((r) => Number(r.index ?? r.slot) === toIndex);
        target.refs = refs.filter((r) => {
            const idx = Number(r.index ?? r.slot);
            return idx !== fromIndex && idx !== toIndex;
        });
        target.refs.push({ ...fromRef, index: toIndex, slot: undefined });
        if (toRef) {
            target.refs.push({ ...toRef, index: fromIndex, slot: undefined });
        }
        if (isGlobal) {
            this.timeline.global = target;
            if (this.isR2vCommonEnabled()) rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    }

    removeRef(target, index) {
        target.refs = (target.refs || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
    }

    renderRefAudioSlots() {
        const isGlobal = this.usesGlobalRefPanel();
        const box = isGlobal ? this.globalRefAudiosBox : this.segRefAudiosBox;
        if (!box) return;
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const taskKey = isGlobal
            ? this.getTaskKey()
            : resolveTaskKey(
                target?.taskType || this.timeline.global?.taskType || this.globalTask?.value || this.getTaskKey(),
            );
        const polished = this.usesRv2vRefStyle(taskKey);
        const countEl = isGlobal ? this.globalAudiosCount : this.segAudiosCount;
        let filled = 0;
        for (const r of target.refAudios) {
            if (r?.audioFile || r?.fileName) filled += 1;
        }
        if (countEl) countEl.textContent = polished ? `${filled}/${MAX_REFERENCE_AUDIOS}` : "";
        this._syncPickExistingDisabled(
            isGlobal ? '[data-r="global-audios-pick"]' : '[data-r="seg-audios-pick"]',
            filled >= MAX_REFERENCE_AUDIOS,
        );

        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_AUDIOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-audio";
            el.dataset.audioSlot = String(i);
            el.dataset.refKind = "audio";
            el.dataset.refIndex = String(i);
            const label = refAudioLabel(i);
            const ref = (target.refAudios || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.audioFile || ref?.fileName || "";
            el.title = file
                ? t("ref.audioTitleFilled", { label, file })
                : t("ref.audioTitleEmpty", { label });
            if (polished) {
                const thumb = document.createElement("div");
                thumb.className = "bd-r2v-thumb";
                const meta = document.createElement("div");
                meta.className = "bd-r2v-meta";
                const tag = document.createElement("span");
                tag.className = "tag";
                tag.textContent = label;
                meta.appendChild(tag);
                el.appendChild(thumb);
                el.appendChild(meta);
                if (file) {
                    el.classList.add("has-audio");
                    const playBtn = document.createElement("button");
                    playBtn.type = "button";
                    playBtn.className = "bd-r2v-play";
                    playBtn.title = t("batch.r2v.play");
                    playBtn.textContent = "▶";
                    thumb.appendChild(playBtn);
                    const dur = document.createElement("span");
                    dur.className = "bd-r2v-dur";
                    dur.textContent = ref?.durationSec != null
                        ? formatMediaDuration(ref.durationSec)
                        : "--:--";
                    meta.appendChild(dur);
                    const progress = document.createElement("div");
                    progress.className = "bd-r2v-progress";
                    progress.title = t("batch.r2v.seek");
                    progress.innerHTML = `<div class="bd-r2v-progress-fill"></div>`;
                    el.appendChild(progress);
                    const audio = document.createElement("audio");
                    audio.preload = "metadata";
                    audio.src = refViewUrl(file);
                    audio.className = "bd-r2v-media";
                    el.appendChild(audio);
                    bindR2vMediaPlayback(audio, playBtn, progress);
                    wireMediaDuration(audio, dur, (sec) => {
                        if (ref) ref.durationSec = sec;
                    });
                    const x = document.createElement("span");
                    x.className = "x";
                    x.textContent = "×";
                    x.onclick = (e) => {
                        e.stopPropagation();
                        this.removeRefAudio(target, i);
                    };
                    el.appendChild(x);
                } else {
                    thumb.textContent = "♪";
                    const hint = document.createElement("span");
                    hint.className = "name";
                    hint.textContent = t("batch.r2v.uploadHint");
                    meta.appendChild(hint);
                }
            } else if (file) {
                el.classList.add("has-audio");
                const tag = document.createElement("span");
                tag.textContent = label;
                el.appendChild(tag);
                const name = document.createElement("span");
                name.className = "bd-ref-audio-name";
                name.textContent = file.split("/").pop() || file;
                el.appendChild(name);
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeRefAudio(target, i);
                };
                el.appendChild(x);
            } else {
                el.textContent = t("ref.audioUpload", { label });
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-progress, .x")) return;
                this.pickRefAudio(target, i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    removeRefAudio(target, index) {
        if (!target) return;
        target.refAudios = (target.refAudios || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled() && target === this.timeline.global) {
            rebaseR2vGroupSlotsForCommon(this);
        }
        this.commit();
        this.renderRefAudioSlots();
    }

    pickRefAudio(target, index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.mts,.ts";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefAudioFromFile(file, target, index);
        };
        input.click();
    }

    async addRefAudioFromFile(file, target, slotIndex = null) {
        if (!target || !file) return;
        target.refAudios = target.refAudios || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_AUDIOS }, (_, i) => i)
                .find((i) => !target.refAudios.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const prepared = await prepareLocalReferenceAudio(file);
            const relPath = prepared.relPath;
            if (hasDuplicateReferenceAudio(target.refAudios, relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: relPath,
                fileName: prepared.fileName || file.name,
                type: prepared.type || "input",
                subfolder: prepared.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && target === this.timeline.global) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio upload failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    }

    /** r2v common panel: multi-slot global.refVideos (1–3), merged into groups at run time. */
    renderR2vCommonVideoSlots() {
        const box = this.globalRefVideosBox;
        if (!box || !this.usesR2vCommonPanel()) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let filled = 0;
        for (const r of target.refVideos) {
            if (r?.videoFile || r?.fileName || r?.previewImageFile || r?.previewImageUrl || r?.linked) {
                filled += 1;
            }
        }
        if (this.globalVideosCount) {
            this.globalVideosCount.textContent = `${filled}/${MAX_REFERENCE_VIDEOS}`;
        }
        this._syncPickExistingDisabled('[data-r="global-videos-pick"]', filled >= MAX_REFERENCE_VIDEOS);
        box.innerHTML = "";
        for (let i = 0; i < MAX_REFERENCE_VIDEOS; i++) {
            const el = document.createElement("div");
            el.className = "bd-ref-video";
            el.dataset.videoSlot = String(i);
            el.dataset.refKind = "video";
            el.dataset.refIndex = String(i);
            const label = refVideoLabel(i);
            const ref = (target.refVideos || []).find((r) => Number(r.index ?? r.slot) === i);
            const file = ref?.videoFile || "";
            const posterSrc = ref?.previewImageUrl
                || (ref?.previewImageFile ? refViewUrl(ref.previewImageFile) : "");
            const hasMedia = !!(file || posterSrc || ref?.linked);
            const titleFile = file || ref?.fileName || ref?.previewImageFile || "";
            el.title = hasMedia
                ? t("ref.videoTitleFilled", { label, file: titleFile || label })
                : t("ref.videoTitleEmpty", { label });
            const thumb = document.createElement("div");
            thumb.className = "bd-r2v-thumb bd-r2v-thumb-video";
            const meta = document.createElement("div");
            meta.className = "bd-r2v-meta";
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.textContent = label;
            meta.appendChild(tag);
            el.appendChild(thumb);
            el.appendChild(meta);
            if (file) {
                el.classList.add("has-video");
                const video = document.createElement("video");
                video.preload = "metadata";
                video.muted = true;
                video.playsInline = true;
                video.src = refViewUrl(file);
                video.className = "bd-r2v-media";
                thumb.appendChild(video);
                const playBtn = document.createElement("button");
                playBtn.type = "button";
                playBtn.className = "bd-r2v-play";
                playBtn.title = t("batch.r2v.play");
                playBtn.textContent = "▶";
                thumb.appendChild(playBtn);
                const dur = document.createElement("span");
                dur.className = "bd-r2v-dur";
                dur.textContent = ref?.durationSec != null
                    ? formatMediaDuration(ref.durationSec)
                    : "--:--";
                meta.appendChild(dur);
                bindR2vMediaPlayback(video, playBtn);
                playBtn.addEventListener("click", () => { video.muted = false; });
                wireMediaDuration(video, dur, (sec) => {
                    if (ref) ref.durationSec = sec;
                });
                video.addEventListener("loadeddata", () => {
                    if (video.readyState >= 2 && video.currentTime < 0.05) {
                        try {
                            video.currentTime = Math.min(0.1, (video.duration || 1) * 0.05);
                        } catch (_) { /* ignore */ }
                    }
                }, { once: true });
                const x = document.createElement("span");
                x.className = "x";
                x.textContent = "×";
                x.onclick = (e) => {
                    e.stopPropagation();
                    this.removeR2vCommonVideo(i);
                };
                el.appendChild(x);
            } else if (posterSrc) {
                el.classList.add("has-video");
                const img = document.createElement("img");
                img.className = "bd-r2v-media";
                img.src = posterSrc;
                img.alt = label;
                thumb.appendChild(img);
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.externalPoster");
                meta.appendChild(hint);
            } else {
                thumb.textContent = "▶";
                const hint = document.createElement("span");
                hint.className = "name";
                hint.textContent = t("batch.r2v.uploadHint");
                meta.appendChild(hint);
            }
            el.onclick = (e) => {
                if (e.target?.closest?.(".bd-r2v-play, .bd-r2v-dur, .x, video")) return;
                if (file && e.target?.closest?.(".bd-r2v-thumb")) {
                    el.querySelector(".bd-r2v-play")?.click();
                    return;
                }
                this.pickR2vCommonVideo(i);
            };
            box.appendChild(el);
        }
        refreshPromptTokenEditors(this.root || document);
    }

    removeR2vCommonVideo(index) {
        const target = this.timeline.global;
        if (!target) return;
        target.refVideos = (target.refVideos || []).filter((r) => Number(r.index ?? r.slot) !== index);
        if (this.isR2vCommonEnabled()) {
            rebaseR2vGroupSlotsForCommon(this);
            this.renderImageBatchGroups?.();
        }
        this.commit();
        this.renderR2vCommonVideoSlots();
    }

    pickR2vCommonVideo(index) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*,.mp4,.mov,.webm,.mkv";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addR2vCommonVideoFromFile(file, index);
        };
        input.click();
    }

    async pickExistingR2vCommonVideo() {
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        const index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
            .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i && (r.videoFile || r.fileName)));
        if (index == null) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseVideoInput({
                title: t("mediaPicker.pickReferenceVideo"),
            });
            if (!picked?.relPath) return;
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video pick failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    }

    async addR2vCommonVideoFromFile(file, slotIndex = null) {
        if (!file) return;
        const target = (this.timeline.global = this.timeline.global || {
            refs: [], refAudios: [], refVideos: [],
        });
        target.refVideos = target.refVideos || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_VIDEOS }, (_, i) => i)
                .find((i) => !target.refVideos.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInputSmart(file);
            const relPath = videoRelativePath(uploaded);
            target.refVideos = target.refVideos.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refVideos.push({
                index,
                videoFile: relPath,
                fileName: uploaded?.name || file.name,
                type: "input",
                subfolder: uploaded?.subfolder || "",
            });
            if (this.isR2vCommonEnabled()) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderR2vCommonVideoSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] common ref video upload failed:", err);
            alert(t("upload.refVideoBatchFailed", { err: err?.message || err }));
        }
    }

    pickRef(target, index, isGlobal) {
        const input = document.createElement("input");
        input.type = "file"; input.accept = "image/*";
        input.onchange = () => {
            const file = input.files?.[0];
            if (file) this.addRefFromFile(file, target, index, isGlobal);
        };
        input.click();
    }

    _nextEmptyMediaSlot(items, max, hasFn) {
        for (let i = 0; i < max; i++) {
            const hit = (items || []).find((r) => Number(r.index ?? r.slot) === i);
            if (!hasFn(hit)) return i;
        }
        return -1;
    }

    _syncPickExistingDisabled(selector, disabled) {
        const btn = this.root?.querySelector(selector);
        if (!btn) return;
        btn.disabled = !!disabled;
        btn.title = disabled ? t("mediaPicker.slotsFull") : t("mediaPicker.pickExistingHint");
    }

    async pickExistingRef(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refs = target.refs || [];
        const index = this._nextEmptyMediaSlot(
            target.refs,
            MAX_REFERENCE_IMAGES,
            (r) => !!(r?.imageFile || r?.imageB64),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseImageInput({
                title: t("mediaPicker.pickReferenceImage"),
            });
            if (!picked?.imageFile) return;
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: picked.imageFile, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
            this.renderRefSlots(
                target.refs,
                isGlobal ? this.globalRefsBox : this.segRefsBox,
                isGlobal,
            );
        } catch (err) {
            console.error("[MiniMax H3Director] ref pick failed:", err);
        }
    }

    async pickExistingRefAudio(isGlobal) {
        const target = isGlobal
            ? (this.timeline.global = this.timeline.global || { refs: [], refAudios: [] })
            : this.timeline.segments[this.selectedIndex];
        if (!target) return;
        target.refAudios = target.refAudios || [];
        const index = this._nextEmptyMediaSlot(
            target.refAudios,
            MAX_REFERENCE_AUDIOS,
            (r) => !!(r?.audioFile || r?.fileName),
        );
        if (index < 0) {
            alert(t("mediaPicker.slotsFull"));
            return;
        }
        try {
            const picked = await this.chooseAudioInput({
                title: t("mediaPicker.pickReferenceAudio"),
            });
            if (!picked?.relPath) return;
            if (hasDuplicateReferenceAudio(target.refAudios, picked.relPath, index)) {
                alert(t("ref.audioDuplicate"));
                return;
            }
            target.refAudios = target.refAudios.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refAudios.push({
                index,
                audioFile: picked.relPath,
                fileName: picked.fileName || picked.relPath,
                type: picked.type || "input",
                subfolder: picked.subfolder || "",
            });
            if (this.isR2vCommonEnabled() && isGlobal) {
                rebaseR2vGroupSlotsForCommon(this);
                this.renderImageBatchGroups?.();
            }
            this.commit();
            this.renderRefAudioSlots();
        } catch (err) {
            console.error("[MiniMax H3Director] ref audio pick failed:", err);
            alert(t("upload.refAudioFailed", { err: err?.message || err }));
        }
    }

    async addRefFromFile(file, target, slotIndex = null, isGlobal = null) {
        target.refs = target.refs || [];
        let index = slotIndex;
        if (index == null) {
            index = Array.from({ length: MAX_REFERENCE_IMAGES }, (_, i) => i)
                .find((i) => !target.refs.some((r) => Number(r.index ?? r.slot) === i));
            if (index == null) return;
        }
        try {
            const uploaded = await uploadToInput(file);
            const relPath = videoRelativePath(uploaded);
            target.refs = target.refs.filter((r) => Number(r.index ?? r.slot) !== index);
            target.refs.push({ index, imageFile: relPath, imageB64: "" });
            if (isGlobal) {
                this.timeline.global = target;
                if (this.isR2vCommonEnabled()) {
                    rebaseR2vGroupSlotsForCommon(this);
                    this.renderImageBatchGroups?.();
                }
            }
            this.commit();
        } catch (err) {
            console.error("[MiniMax H3Director] ref upload failed:", err);
        }
    }
























}

Object.assign(MiniMaxH3DirectorOptEditor.prototype, live_playbackMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, segment_editMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, media_pickerMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, dialogsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, thumbnailsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, stage_previewMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, widget_syncMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, output_uiMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, task_uiMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, frame_map_ioMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, segments_modelMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, video_loadMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, media_thumbsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, ref_video_slotMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, task_layoutMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, workspacesMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, modesMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, fieldsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, layout_scheduleMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, external_groupsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, export_pickersMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, run_selectionMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, timeline_payloadMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, eventsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, dom_shellMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, interactionMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, canvasMixin);
