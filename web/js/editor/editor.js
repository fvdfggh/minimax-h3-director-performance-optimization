



import { coerceTimelineFps } from "../core/dims.js";
import { hideWidget, parseTimeline } from "../core/editor_lifecycle.js";


import { HIDDEN_WIDGETS, RULER_H, SEG_LABEL_H, TRACK_H } from "../core/layout_spec.js";








import { getDirectorMode, resolveTaskKey } from "../minimax_gen_timeline.js";
import { onLocaleChange, t } from "../minimax_i18n.js";
import { ensureImageBatchTimeline } from "../minimax_image_batch.js";



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
import { ref_slotsMixin } from "./mixins/ref_slots.js";
import { audio_pickersMixin } from "./mixins/audio_pickers.js";
import { asrCheckMixin } from "./mixins/asr_check.js";

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









































































}

Object.assign(MiniMaxH3DirectorOptEditor.prototype, layout_scheduleMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, ref_slotsMixin);

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


Object.assign(MiniMaxH3DirectorOptEditor.prototype, external_groupsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, export_pickersMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, audio_pickersMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, asrCheckMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, run_selectionMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, timeline_payloadMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, eventsMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, dom_shellMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, interactionMixin);

Object.assign(MiniMaxH3DirectorOptEditor.prototype, canvasMixin);
