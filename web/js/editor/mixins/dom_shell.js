/** dom_shell mixin for the Director editor (dom_shell).
 *
 * Extracted verbatim from editor.js; methods are unchanged apart from the
 * trailing comma an object literal needs.
 */

import { STYLES } from "../styles.js";
import { bindFl2vEvents, mountFl2vPanel } from "../../minimax_fl2v.js";
import { CUSTOM_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, DEFAULT_MEGAPIXELS, MAX_GEN_FRAMES, RESOLUTION_ASPECTS } from "../../minimax_gen_timeline.js";
import { aspectDisplayLabel } from "../../minimax_i18n.js";
import { bindImageBatchEvents, mountImageBatchPanel, normalizeImageBatchSegments, renderImageBatchGroups, wireBatchRunSelectControls } from "../../minimax_image_batch.js";
import { teardownPromptImageMentions } from "../../minimax_prompt_mentions.js";
export const dom_shellMixin = {
    buildDOM() {
        this.root = document.createElement("div");
        this.root.className = "bd-wrap";
        this.root.innerHTML = `<style>${STYLES}</style>`;

        const toolbarWrap = document.createElement("div");
        toolbarWrap.className = "bd-toolbar-wrap";
        toolbarWrap.innerHTML = `
            <div class="bd-toolbar">
                <div class="bd-actions">
                    <button type="button" class="bd-btn bd-btn-primary hidden" data-a="r2v-add-group" data-i18n="toolbar.addRefGroup" data-i18n-title="tooltip.addRefGroup">添加素材组</button>
                    <button type="button" class="bd-btn bd-btn-primary" data-a="video" data-i18n="toolbar.uploadVideo">上传视频</button>
                    <button type="button" class="bd-btn" data-a="video-existing" data-i18n="mediaPicker.pickExistingVideo" data-i18n-title="mediaPicker.pickExistingHint">选已有视频</button>
                    <button type="button" class="bd-btn bd-btn-primary hidden" data-a="fl2v-add-shot" data-i18n="toolbar.addShot" data-i18n-title="tooltip.addShot">添加一组</button>
                    <button type="button" class="bd-btn" data-a="video-append" data-i18n="toolbar.appendVideo" data-i18n-title="tooltip.appendVideo">追加视频</button>
                    <button type="button" class="bd-btn" data-a="split" data-i18n="toolbar.split">+ 分割</button>
                    <input type="number" class="bd-num" data-r="equal-n" min="2" max="64" value="2" data-i18n-title="tooltip.equalSplitN">
                    <button type="button" class="bd-btn" data-a="equal" data-i18n="toolbar.equalSplit">均分</button>
                    <button type="button" class="bd-btn" data-a="smart-split" data-i18n="toolbar.smartSplit" data-i18n-title="tooltip.smartSplit">智能分割</button>
                    <button type="button" class="bd-btn" data-a="run-select-toggle" data-i18n="toolbar.runSelect" data-i18n-title="tooltip.runSelect">选择运行</button>
                    <label class="bd-run-select-all-wrap hidden" data-r="run-select-all-wrap" data-i18n-title="tooltip.runSelectAll">
                        <input type="checkbox" data-r="run-select-all-cb">
                        <span data-i18n="toolbar.selectAll">全选</span>
                    </label>
                    <button type="button" class="bd-btn" data-a="seg-export" data-i18n="toolbar.segmentExport" data-i18n-title="tooltip.segmentExport">分段导出</button>
                    <button type="button" class="bd-btn" data-a="second-sample" title="二次采样：对已有缓存片段做 放大+重采样+连续出片">二次采样</button>
                    <button type="button" class="bd-btn bd-btn-danger" data-a="del" data-i18n="toolbar.deleteSegment" data-i18n-title="tooltip.deleteSegment">删除片段</button>
                    <div class="bd-mode">
                        <button type="button" data-a="mode-global" class="active" data-i18n="toolbar.modeGlobal">全局模式</button>
                        <button type="button" data-a="mode-segment" data-i18n="toolbar.modeSegment">分段模式</button>
                    </div>
                    <select class="bd-select" data-r="global-task" title="task_type"></select>
                    <span class="bd-video-tag" data-r="video-name" data-i18n="toolbar.noVideo">未上传视频</span>
                </div>
                <div class="bd-right">
                    <div class="bd-tl-zoom" data-r="tl-zoom">
                        <button type="button" class="bd-btn bd-btn-zoom" data-a="zoom-toggle" data-i18n="toolbar.timelineZoom" data-i18n-title="toolbar.timelineZoomTitle">放大</button>
                        <input type="range" class="bd-tl-zoom-slider hidden" data-r="zoom" min="1" max="10" step="any" value="1" data-i18n-title="tooltip.timelineZoom">
                    </div>
                    <button type="button" class="bd-btn" data-a="pack-import" data-i18n="toolbar.importPack" data-i18n-title="tooltip.importPack">导入导演包</button>
                    <button type="button" class="bd-btn" data-a="pack-export" data-i18n="toolbar.exportPack" data-i18n-title="tooltip.exportPack">导出导演包</button>
                    <button type="button" class="bd-btn" data-a="lang-toggle" data-i18n="toolbar.langToggle" data-i18n-title="toolbar.langToggleTitle">EN</button>
                    <div class="bd-bounds" data-r="bounds">起点: 0.00 | 终点: -</div>
                    <div class="bd-timecode" data-r="timecode">0.00s</div>
                </div>
            </div>
            <div class="bd-smart-split-msg hidden" data-r="smart-split-msg" role="status"></div>
            <div class="bd-external-groups-msg hidden" data-r="external-groups-msg" role="status"></div>`;
        this.root.appendChild(toolbarWrap);
        this.smartSplitMsgEl = toolbarWrap.querySelector('[data-r="smart-split-msg"]');
        this.externalGroupsMsgEl = toolbarWrap.querySelector('[data-r="external-groups-msg"]');
        this.langToggleBtn = toolbarWrap.querySelector('[data-a="lang-toggle"]');

        this.mainBody = document.createElement("div");
        this.mainBody.className = "bd-main";
        this.root.appendChild(this.mainBody);

        const stage = document.createElement("div");
        stage.className = "bd-stage hidden";
        stage.setAttribute("data-r", "video-stage");
        stage.innerHTML = `
            <video class="bd-stage-video hidden" data-r="stage-video" muted playsinline preload="auto"></video>
            <img class="bd-stage-img hidden" data-r="stage-img" alt="">
            <div class="bd-stage-empty" data-r="stage-empty" data-i18n="stage.empty">上传视频后可在此预览播放</div>
            <div class="bd-stage-badge hidden" data-r="stage-badge"></div>`;
        this.mainBody.appendChild(stage);

        // Playback bar sits between video stage and timeline edit area.
        const controls = document.createElement("div");
        controls.className = "bd-controls";
        controls.innerHTML = `
            <div class="bd-player">
                <button type="button" class="bd-icon-btn" data-a="play" data-i18n-title="player.playPause">▶</button>
                <button type="button" class="bd-icon-btn" data-a="loop" data-i18n-title="player.loopOn">⟳</button>
                <button type="button" class="bd-icon-btn" data-a="frame-prev" data-i18n-title="player.framePrev">‹</button>
                <button type="button" class="bd-icon-btn" data-a="frame-next" data-i18n-title="player.frameNext">›</button>
                <span class="bd-frame-jump" data-i18n-title="player.frameJump">
                    <span data-i18n="player.frame">帧</span>
                    <input type="number" class="bd-frame-input" data-r="frame-input" min="1" step="1" value="1">
                    <span>/</span>
                    <span class="bd-frame-total" data-r="frame-total">0</span>
                </span>
                <div class="bd-timecode" data-r="player-timecode">0.00 / 0.00</div>
                <input type="range" class="bd-seek" data-r="seek" min="0" value="0" step="1">
            </div>`;
        this.mainBody.appendChild(controls);

        // Appears above the timeline when a split point is selected.
        const splitEditBar = document.createElement("div");
        splitEditBar.className = "bd-split-edit-bar hidden";
        splitEditBar.setAttribute("data-r", "split-edit-bar");
        splitEditBar.innerHTML = `
            <span class="bd-split-edit-hint" data-r="split-edit-hint" data-i18n="split.selectedHint">已选中分割点</span>
            <button type="button" class="bd-btn bd-btn-del-split" data-a="del-split" data-i18n="toolbar.deleteSplitPoint" data-i18n-title="tooltip.deleteSplitPoint">删除分割点</button>`;
        this.mainBody.appendChild(splitEditBar);
        this.splitEditBarEl = splitEditBar;
        this.splitEditHintEl = splitEditBar.querySelector('[data-r="split-edit-hint"]');

        this.viewport = document.createElement("div");
        this.viewport.className = "bd-viewport";
        this.canvas = document.createElement("canvas");
        this.canvas.className = "bd-canvas";
        this.viewport.appendChild(this.canvas);
        this.mainBody.appendChild(this.viewport);
        this.ctx = this.canvas.getContext("2d");

        const outputBar = document.createElement("div");
        outputBar.className = "bd-output";
        outputBar.innerHTML = `
            <span class="bd-fl2v-total-wrap hidden" data-r="fl2v-total-wrap" data-i18n-title="tooltip.fl2vTotalDuration">
                <label data-i18n="output.totalDurationSec">总时长（秒）</label>
                <input type="number" class="bd-num" data-r="fl2v-total" min="1" max="99999" step="0.1" value="5" style="width:64px" disabled data-i18n-title="tooltip.fl2vTotalInput">
            </span>
            <label data-i18n="output.resolution">输出分辨率</label>
            <select class="bd-select" data-r="out-aspect" data-i18n-title="tooltip.aspectRatio" style="max-width:200px">
                ${RESOLUTION_ASPECTS.map(([label]) => `<option value="${label}"${label === DEFAULT_ASPECT_RATIO ? " selected" : ""}>${aspectDisplayLabel(label)}</option>`).join("")}
                <option value="${CUSTOM_ASPECT_RATIO}">${aspectDisplayLabel(CUSTOM_ASPECT_RATIO)}</option>
            </select>
            <span class="bd-out-mp-wrap" data-r="out-mp-wrap" data-i18n-title="tooltip.megapixels">
                <label data-i18n="output.megapixels">百万像素</label>
                <input type="number" class="bd-num" data-r="out-mp" min="0.1" max="16" step="0.1" value="${DEFAULT_MEGAPIXELS}" style="width:56px">
            </span>
            <span class="bd-out-long hidden" data-r="out-long-wrap">
                <label data-i18n="output.longEdge">最长边</label>
                <input type="number" class="bd-num" data-r="out-long" min="32" max="8192" step="1" value="864" style="width:56px" data-i18n-title="tooltip.longEdge">
            </span>
            <span class="bd-out-fixed hidden" data-r="out-fixed-wrap" data-i18n-title="tooltip.customWH">
                <label data-i18n="output.width">宽</label>
                <input type="number" class="bd-num" data-r="out-w" min="32" max="8192" step="32" value="864" style="width:56px">
                <label data-i18n="output.height">高</label>
                <input type="number" class="bd-num" data-r="out-h" min="32" max="8192" step="32" value="480" style="width:56px">
            </span>
            <select class="bd-select hidden" data-r="out-mode" data-i18n-title="tooltip.outputMode">
                <option value="long_edge" data-i18n="output.mode.longEdge">最长边缩放</option>
                <option value="fixed" data-i18n="output.mode.fixed">固定宽高</option>
            </select>
            <label data-i18n="output.fpsLabel" data-i18n-title="tooltip.fps">帧率</label>
            <input type="number" class="bd-num" data-r="timeline-fps" min="1" max="240" step="0.01" value="24" style="width:64px" data-i18n-title="tooltip.timelineFps">
            <span class="bd-out-audio-wrap" data-r="out-audio-wrap" data-i18n-title="tooltip.audioMode">
                <label data-i18n="output.audio.label">声音</label>
                <select class="bd-select" data-r="out-audio-mode" style="max-width:120px">
                    <option value="generate" data-i18n="output.audio.generate">生成声音</option>
                    <option value="source" data-i18n="output.audio.source">使用原声</option>
                    <option value="mute" data-i18n="output.audio.mute">静音</option>
                </select>
            </span>
            <span class="bd-meta" data-r="out-preview">—</span>
            <span class="bd-meta hidden" data-r="out-hint"></span>
            <label data-i18n="output.exportMode.label" data-i18n-title="tooltip.exportMode">导出方式</label>
            <select class="bd-select" data-r="out-export-mode" data-i18n-title="tooltip.exportMode">
                <option value="all" data-i18n="output.exportMode.all">全部导出</option>
                <option value="segments" data-i18n="output.exportMode.segments">分段导出</option>
            </select>
            <span class="hidden" data-r="out-max-frames-wrap" hidden aria-hidden="true">
                <label data-i18n="output.maxFrames">最大帧数</label>
                <input type="number" class="bd-num" data-r="out-max-frames" min="0" max="999999" step="1" value="0" style="width:64px">
            </span>
            <span class="bd-continuous-ref hidden" data-r="segment-continuity-wrap" hidden aria-hidden="true" title="">
                <label><input type="checkbox" data-r="segment-continuity-cb"><span data-i18n="output.segmentContinuity">段间引导</span></label>
                <span class="bd-meta" data-i18n="output.continuityOverlap">上下文帧数</span>
                <select class="bd-num" data-r="segment-continuity-overlap" style="width:64px">
                    <option value="5">5</option>
                    <option value="22" selected>22</option>
                    <option value="39">39</option>
                    <option value="56">56</option>
                </select>
            </span>
            <button type="button" class="bd-btn bd-btn-live-preview active" data-a="live-tae-preview" data-i18n="toolbar.liveTaePreview" data-i18n-title="tooltip.liveTaePreview">实时预览</button>`;
        this.mainBody.appendChild(outputBar);
        this.outputBarEl = outputBar;

        const liveSample = document.createElement("div");
        liveSample.className = "bd-live-sample hidden";
        liveSample.setAttribute("data-r", "live-sample");
        liveSample.innerHTML = `
            <div class="bd-live-sample-head">
                <b data-i18n="liveSample.title">采样预览</b>
                <span class="bd-meta" data-r="live-sample-meta" data-i18n="liveSample.idleHint">开启后，采样过程中显示实时画面</span>
            </div>
            <div class="bd-live-sample-body">
                <img class="hidden" data-r="live-sample-img" alt="live preview">
                <div class="bd-live-sample-empty" data-r="live-sample-empty" data-i18n="liveSample.waiting">等待采样…</div>
                <div class="bd-live-sample-badge hidden" data-r="live-sample-badge"></div>
            </div>`;
        this.mainBody.appendChild(liveSample);
        this.liveSampleEl = liveSample;
        this.liveSampleImg = liveSample.querySelector('[data-r="live-sample-img"]');
        this.liveSampleEmpty = liveSample.querySelector('[data-r="live-sample-empty"]');
        this.liveSampleBadge = liveSample.querySelector('[data-r="live-sample-badge"]');
        this.liveSampleMeta = liveSample.querySelector('[data-r="live-sample-meta"]');
        this._liveSampleHost = "main";

        const bottom = document.createElement("div");
        bottom.className = "bd-split";
        bottom.innerHTML = `
            <div class="bd-panel" data-r="global-panel">
                <div class="bd-r2v-common-head" data-r="r2v-common-head">
                    <div class="bd-r2v-common-titles">
                        <b data-r="global-panel-title" data-i18n="panel.globalPromptAndRefs">全局提示词 & 参考图 (图片1–9)</b>
                        <span class="bd-r2v-common-status" data-r="r2v-common-status" data-i18n="panel.r2vCommonOff">未启用 · 各组独立素材与提示词</span>
                    </div>
                    <div class="bd-r2v-common-actions">
                        <button type="button" class="bd-btn bd-r2v-common-fold hidden" data-r="r2v-common-fold" data-i18n="panel.r2vCommonCollapse">收起公共参数</button>
                        <button type="button" class="bd-btn bd-r2v-common-toggle" data-r="r2v-common-toggle" data-i18n="panel.r2vCommonEnable">启用公共参数</button>
                    </div>
                </div>
                <div class="bd-r2v-common-body" data-r="r2v-common-body">
                    <div class="bd-meta bd-r2v-common-hint hidden" data-r="r2v-common-hint" data-i18n="panel.r2vCommonHint">公共参考图/视频/音频供各组读取；公共提示词会与每组提示词拼接成完整提示词。同槽位以组内素材优先。</div>
                    <div class="bd-prompt-layout" data-r="global-prompt-layout">
                        <div class="bd-refs-col" data-r="global-refs-col">
                            <div class="bd-refs-images-wrap" data-r="global-refs-images-wrap">
                                <div class="bd-r2v-section-head" data-r="global-refs-head">
                                    <span class="bd-label bd-r2v-section-title" data-r="global-refs-label" data-i18n="panel.refImages">参考图 (图片1–9)</span>
                                    <span class="bd-r2v-section-actions">
                                        <button type="button" class="bd-r2v-pick-existing" data-r="global-refs-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                        <span class="bd-r2v-section-count" data-r="global-refs-count"></span>
                                    </span>
                                </div>
                                <div class="bd-refs" data-r="global-refs"></div>
                            </div>
                            <div class="bd-ref-videos-wrap hidden" data-r="global-ref-videos-wrap">
                                <div class="bd-r2v-section-head" data-r="global-videos-head">
                                    <span class="bd-label bd-r2v-section-title" data-i18n="batch.r2v.sectionVideos">参考视频</span>
                                    <span class="bd-r2v-section-actions">
                                        <button type="button" class="bd-r2v-pick-existing" data-r="global-videos-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                        <span class="bd-r2v-section-count" data-r="global-videos-count"></span>
                                    </span>
                                </div>
                                <div class="bd-ref-videos" data-r="global-ref-videos"></div>
                            </div>
                            <div class="bd-ref-audios-wrap hidden" data-r="global-ref-audios-wrap">
                                <div class="bd-r2v-section-head" data-r="global-audios-head">
                                    <span class="bd-label bd-r2v-section-title" data-i18n="batch.r2v.sectionAudios">参考音频</span>
                                    <span class="bd-r2v-section-actions">
                                        <button type="button" class="bd-r2v-pick-existing" data-r="global-audios-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                        <span class="bd-r2v-section-count" data-r="global-audios-count"></span>
                                    </span>
                                </div>
                                <div class="bd-ref-audios" data-r="global-ref-audios"></div>
                            </div>
                            <div class="bd-ref-video-col hidden" data-r="global-ref-video-col">
                                <span class="bd-label" data-i18n="panel.refVideo">参考视频（植入内容）</span>
                                <div class="bd-gen-src" data-r="global-ref-video" data-i18n="panel.uploadRefVideo" data-i18n-title="tooltip.uploadRefVideo">点击上传参考视频</div>
                                <span class="bd-meta bd-ref-video-name" data-r="global-ref-video-name"></span>
                                <label class="bd-continuous-ref hidden" data-r="continuous-ref-wrap" data-i18n-title="tooltip.continuousRef">
                                    <input type="checkbox" data-r="continuous-ref-cb">
                                    <span data-i18n="panel.continuousRef">连续参考</span>
                                </label>
                            </div>
                            <div class="bd-gen-src hidden" data-r="gen-global-img" data-i18n="panel.uploadSourceImage" data-i18n-title="tooltip.uploadSourceImage">点击上传源图片</div>
                        </div>
                        <div class="bd-prompt-col">
                            <span class="bd-label" data-i18n="panel.prompt">提示词</span>
                            <textarea class="bd-prompt" data-r="global-prompt" data-i18n-placeholder="placeholder.globalPrompt" placeholder=""></textarea>
                            <textarea class="bd-prompt bd-prompt-negative hidden" data-r="global-negative" hidden aria-hidden="true"></textarea>
                        </div>
                    </div>
                    <div class="bd-gen-fc-row hidden" data-r="gen-global-fc-row">
                        <span class="bd-label" data-i18n="panel.defaultSegmentFrames">默认片段帧数</span>
                        <input type="number" class="bd-num" data-r="gen-default-fc" min="1" max="${MAX_GEN_FRAMES}" value="124" style="width:72px">
                    </div>
                </div>
            </div>
            <div class="bd-panel" data-r="segment-panel" style="display:none">
                <div class="bd-seg-head">
                    <b data-r="seg-label">片段 1</b>
                    <label class="bd-seg-continuity hidden" data-r="seg-continuity-from-prev-wrap" hidden>
                        <input type="checkbox" data-r="seg-continuity-from-prev">
                        <span data-i18n="batch.continuityFromPrev">引用上段</span>
                    </label>
                    <div class="bd-meta" data-r="seg-info"></div>
                    <label class="bd-seg-refsize hidden" data-r="seg-ref-image-size-wrap" hidden data-i18n-title="tooltip.refImageSize">
                        <span data-i18n="output.refImageSize.label">参考图尺寸</span>
                        <select class="bd-select" data-r="seg-ref-image-size">
                            <option value="match" data-i18n="output.refImageSize.match">match</option>
                            <option value="max" data-i18n="output.refImageSize.max">max</option>
                        </select>
                    </label>
                </div>
                <div class="bd-prompt-layout" data-r="seg-prompt-layout">
                    <div class="bd-refs-col" data-r="seg-refs-col">
                        <div class="bd-refs-images-wrap" data-r="seg-refs-images-wrap">
                            <div class="bd-r2v-section-head" data-r="seg-refs-head">
                                <span class="bd-label bd-r2v-section-title" data-r="seg-refs-label" data-i18n="panel.segmentRefImages">片段参考图 (图片1–9)</span>
                                <span class="bd-r2v-section-actions">
                                    <button type="button" class="bd-r2v-pick-existing" data-r="seg-refs-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                    <span class="bd-r2v-section-count" data-r="seg-refs-count"></span>
                                </span>
                            </div>
                            <div class="bd-refs" data-r="seg-refs"></div>
                        </div>
                        <div class="bd-ref-audios-wrap hidden" data-r="seg-ref-audios-wrap">
                            <div class="bd-r2v-section-head" data-r="seg-audios-head">
                                <span class="bd-label bd-r2v-section-title" data-i18n="batch.r2v.sectionAudios">参考音频</span>
                                <span class="bd-r2v-section-actions">
                                    <button type="button" class="bd-r2v-pick-existing" data-r="seg-audios-pick" data-i18n="mediaPicker.pickExisting" data-i18n-title="mediaPicker.pickExistingHint">选已有</button>
                                    <span class="bd-r2v-section-count" data-r="seg-audios-count"></span>
                                </span>
                            </div>
                            <div class="bd-ref-audios" data-r="seg-ref-audios"></div>
                        </div>
                        <div class="bd-ref-video-col hidden" data-r="seg-ref-video-col">
                            <span class="bd-label" data-i18n="panel.segmentRefVideo">片段参考视频（植入内容）</span>
                            <div class="bd-gen-src" data-r="seg-ref-video" data-i18n="panel.uploadRefVideo" data-i18n-title="tooltip.uploadRefVideo">点击上传参考视频</div>
                            <span class="bd-meta bd-ref-video-name" data-r="seg-ref-video-name"></span>
                        </div>
                        <div class="bd-gen-src hidden" data-r="gen-seg-img" data-i18n="panel.uploadSegmentSourceImage" data-i18n-title="tooltip.uploadSourceImage">点击上传源图片</div>
                    </div>
                    <div class="bd-prompt-col">
                        <span class="bd-label" data-i18n="panel.prompt">提示词</span>
                        <textarea class="bd-prompt" data-r="seg-prompt" data-i18n-placeholder="placeholder.segmentPrompt" placeholder=""></textarea>
                        <textarea class="bd-prompt bd-prompt-negative hidden" data-r="seg-negative" hidden aria-hidden="true"></textarea>
                    </div>
                </div>
                <div class="bd-gen-fc-row hidden" data-r="gen-seg-fc-row">
                    <span class="bd-label" data-i18n="panel.segmentFrames">片段帧数</span>
                    <input type="number" class="bd-num" data-r="gen-seg-fc" min="1" max="${MAX_GEN_FRAMES}" value="124" style="width:72px">
                </div>
            </div>`;
        this.mainBody.appendChild(bottom);
        this.splitEl = bottom;

        const batchUi = mountImageBatchPanel(this.mainBody);
        this.batchPanel = batchUi.panel;
        this.batchList = batchUi.list;
        this.batchHint = batchUi.hint;
        this.batchI2vNotice = batchUi.i2vNotice;
        this.batchAddBtn = batchUi.addBtn;
        this.batchPicker = batchUi.picker;
        wireBatchRunSelectControls(this, batchUi);

        this.fl2vUi = mountFl2vPanel(this.mainBody);
        this.fl2vTotalWrap = this.root.querySelector('[data-r="fl2v-total-wrap"]');
        if (this.fl2vUi) {
            this.fl2vUi.totalInput = this.root.querySelector('[data-r="fl2v-total"]');
        }
        bindFl2vEvents(this);

        const runStatus = document.createElement("div");
        runStatus.className = "bd-run-status idle";
        runStatus.dataset.r = "run-status";
        runStatus.innerHTML = `
            <div class="bd-run-title" data-r="run-title" data-i18n="run.titleIdle">运行状态：待命</div>
            <div class="bd-run-detail" data-r="run-detail" data-i18n="run.detailIdle">队列执行时将显示当前片段与阶段进度</div>
            <div class="bd-run-select-bar hidden" data-r="run-select-bar">
                <span data-r="run-select-summary" data-i18n="run.summaryAllSegments">将运行全部片段</span>
            </div>
            <div class="bd-run-bars">
                <div class="bd-run-bar" data-i18n-title="run.bar.overall"><div class="bd-run-bar-fill" data-r="run-overall" style="width:0%"></div></div>
                <div class="bd-run-bar bd-run-bar-sub" data-i18n-title="run.bar.phase"><div class="bd-run-bar-fill" data-r="run-phase" style="width:0%"></div></div>
            </div>`;
        this.root.appendChild(runStatus);

        if (this.container) {
            for (const wrap of [...this.container.querySelectorAll(":scope > .bd-wrap")]) {
                wrap.remove();
            }
            this.container.appendChild(this.root);
        }

        this._previewVideo = document.createElement("video");
        this._previewVideo.crossOrigin = "anonymous";
        this._previewVideo.muted = true;
        this._previewVideo.playsInline = true;
        this._previewVideo.preload = "auto";
        this._previewVideo.style.cssText = "position:fixed;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
        document.body.appendChild(this._previewVideo);

        this._thumbCanvas = document.createElement("canvas");
        this._thumbCtx = this._thumbCanvas.getContext("2d", { alpha: false });

        this.videoNameEl = this.root.querySelector('[data-r="video-name"]');
        this.equalCountInput = this.root.querySelector('[data-r="equal-n"]');
        this.boundsEl = this.root.querySelector('[data-r="bounds"]');
        this.timecodeEl = this.root.querySelector('[data-r="timecode"]');
        this.playerTimecodeEl = this.root.querySelector('[data-r="player-timecode"]');
        this.frameInputEl = this.root.querySelector('[data-r="frame-input"]');
        this.frameTotalEl = this.root.querySelector('[data-r="frame-total"]');
        this.seekBar = this.root.querySelector('[data-r="seek"]');
        this.tlZoomWrap = this.root.querySelector('[data-r="tl-zoom"]');
        this.zoomToggleBtn = this.root.querySelector('[data-a="zoom-toggle"]');
        this.zoomSlider = this.root.querySelector('[data-r="zoom"]');
        this.stageEl = this.root.querySelector('[data-r="video-stage"]');
        this.stageVideo = this.root.querySelector('[data-r="stage-video"]');
        this.stageImg = this.root.querySelector('[data-r="stage-img"]');
        this.stageEmpty = this.root.querySelector('[data-r="stage-empty"]');
        this.stageBadge = this.root.querySelector('[data-r="stage-badge"]');
        if (this.stageVideo) {
            this.stageVideo.crossOrigin = "anonymous";
            this.stageVideo.muted = true;
            this.stageVideo.playsInline = true;
        }
        this.globalTask = this.root.querySelector('[data-r="global-task"]');
        this.globalPanel = this.root.querySelector('[data-r="global-panel"]');
        this.globalPanelTitle = this.globalPanel?.querySelector('[data-r="global-panel-title"]')
            || this.globalPanel?.querySelector("b");
        this.r2vCommonHead = this.root.querySelector('[data-r="r2v-common-head"]');
        this.r2vCommonBody = this.root.querySelector('[data-r="r2v-common-body"]');
        this.r2vCommonHint = this.root.querySelector('[data-r="r2v-common-hint"]');
        this.r2vCommonStatus = this.root.querySelector('[data-r="r2v-common-status"]');
        this.r2vCommonFold = this.root.querySelector('[data-r="r2v-common-fold"]');
        this.r2vCommonToggle = this.root.querySelector('[data-r="r2v-common-toggle"]');
        this.segmentPanel = this.root.querySelector('[data-r="segment-panel"]');
        this.globalPrompt = this.root.querySelector('[data-r="global-prompt"]');
        this.globalNegative = this.root.querySelector('[data-r="global-negative"]');
        this.globalPromptLayout = this.root.querySelector('[data-r="global-prompt-layout"]');
        this.segPromptLayout = this.root.querySelector('[data-r="seg-prompt-layout"]');
        this.globalRefsBox = this.root.querySelector('[data-r="global-refs"]');
        this.globalRefsImagesWrap = this.root.querySelector('[data-r="global-refs-images-wrap"]');
        this.globalRefsCount = this.root.querySelector('[data-r="global-refs-count"]');
        this.globalAudiosCount = this.root.querySelector('[data-r="global-audios-count"]');
        this.segRefsImagesWrap = this.root.querySelector('[data-r="seg-refs-images-wrap"]');
        this.segRefsCount = this.root.querySelector('[data-r="seg-refs-count"]');
        this.segAudiosCount = this.root.querySelector('[data-r="seg-audios-count"]');
        this.globalRefAudiosWrap = this.root.querySelector('[data-r="global-ref-audios-wrap"]');
        this.globalRefAudiosBox = this.root.querySelector('[data-r="global-ref-audios"]');
        this.globalRefVideosWrap = this.root.querySelector('[data-r="global-ref-videos-wrap"]');
        this.globalRefVideosBox = this.root.querySelector('[data-r="global-ref-videos"]');
        this.globalVideosCount = this.root.querySelector('[data-r="global-videos-count"]');
        this.segRefAudiosWrap = this.root.querySelector('[data-r="seg-ref-audios-wrap"]');
        this.segRefAudiosBox = this.root.querySelector('[data-r="seg-ref-audios"]');
        this.segLabel = this.root.querySelector('[data-r="seg-label"]');
        this.segContinuityFromPrevWrap = this.root.querySelector('[data-r="seg-continuity-from-prev-wrap"]');
        this.segContinuityFromPrevCb = this.root.querySelector('[data-r="seg-continuity-from-prev"]');
        this.segRefImageSizeWrap = this.root.querySelector('[data-r="seg-ref-image-size-wrap"]');
        this.segRefImageSize = this.root.querySelector('[data-r="seg-ref-image-size"]');
        this.segInfo = this.root.querySelector('[data-r="seg-info"]');
        this.segPrompt = this.root.querySelector('[data-r="seg-prompt"]');
        this.segNegative = this.root.querySelector('[data-r="seg-negative"]');
        this.segRefsBox = this.root.querySelector('[data-r="seg-refs"]');
        this.globalRefsCol = this.root.querySelector('[data-r="global-refs-col"]');
        this.segRefsCol = this.root.querySelector('[data-r="seg-refs-col"]');
        this.globalRefVideoCol = this.root.querySelector('[data-r="global-ref-video-col"]');
        this.globalRefVideo = this.root.querySelector('[data-r="global-ref-video"]');
        this.globalRefVideoNameEl = this.root.querySelector('[data-r="global-ref-video-name"]');
        this.segRefVideoCol = this.root.querySelector('[data-r="seg-ref-video-col"]');
        this.segRefVideo = this.root.querySelector('[data-r="seg-ref-video"]');
        this.segRefVideoNameEl = this.root.querySelector('[data-r="seg-ref-video-name"]');
        this.continuousRefWrap = this.root.querySelector('[data-r="continuous-ref-wrap"]');
        this.continuousRefCb = this.root.querySelector('[data-r="continuous-ref-cb"]');
        this.genGlobalImg = this.root.querySelector('[data-r="gen-global-img"]');
        this.genSegImg = this.root.querySelector('[data-r="gen-seg-img"]');
        this.genGlobalFcRow = this.root.querySelector('[data-r="gen-global-fc-row"]');
        this.genSegFcRow = this.root.querySelector('[data-r="gen-seg-fc-row"]');
        this.genDefaultFc = this.root.querySelector('[data-r="gen-default-fc"]');
        this.genSegFc = this.root.querySelector('[data-r="gen-seg-fc"]');
        this.controlsBar = this.root.querySelector(".bd-controls");
        this.btnVideo = this.root.querySelector('[data-a="video"]');
        this.btnVideoExisting = this.root.querySelector('[data-a="video-existing"]');
        this.btnFl2vAddShot = this.root.querySelector('[data-a="fl2v-add-shot"]');
        this.btnVideoAppend = this.root.querySelector('[data-a="video-append"]');
        this.outHint = this.root.querySelector('[data-r="out-hint"]');
        this.outMode = this.root.querySelector('[data-r="out-mode"]');
        this.outAspect = this.root.querySelector('[data-r="out-aspect"]');
        this.outMpWrap = this.root.querySelector('[data-r="out-mp-wrap"]');
        this.outMp = this.root.querySelector('[data-r="out-mp"]');
        this.outLongWrap = this.root.querySelector('[data-r="out-long-wrap"]');
        this.outFixedWrap = this.root.querySelector('[data-r="out-fixed-wrap"]');
        this.outLong = this.root.querySelector('[data-r="out-long"]');
        this.outW = this.root.querySelector('[data-r="out-w"]');
        this.outH = this.root.querySelector('[data-r="out-h"]');
        this.fpsInput = this.root.querySelector('[data-r="timeline-fps"]');
        this.outAudioWrap = this.root.querySelector('[data-r="out-audio-wrap"]');
        this.outAudioMode = this.root.querySelector('[data-r="out-audio-mode"]');
        this.outMaxFrames = this.root.querySelector('[data-r="out-max-frames"]');
        this.outExportMode = this.root.querySelector('[data-r="out-export-mode"]');
        this.segmentContinuityWrap = this.root.querySelector('[data-r="segment-continuity-wrap"]');
        this.segmentContinuityCb = this.root.querySelector('[data-r="segment-continuity-cb"]');
        this.segmentContinuityOverlap = this.root.querySelector('[data-r="segment-continuity-overlap"]');
        this.outPreview = this.root.querySelector('[data-r="out-preview"]');
        this.runStatusEl = this.root.querySelector('[data-r="run-status"]');
        this.runTitleEl = this.root.querySelector('[data-r="run-title"]');
        this.runDetailEl = this.root.querySelector('[data-r="run-detail"]');
        this.runOverallEl = this.root.querySelector('[data-r="run-overall"]');
        this.runPhaseEl = this.root.querySelector('[data-r="run-phase"]');
        this.runSelectBar = this.root.querySelector('[data-r="run-select-bar"]');
        this.runSelectSummary = this.root.querySelector('[data-r="run-select-summary"]');
        this.btnRunSelectToggle = this.root.querySelector('[data-a="run-select-toggle"]');
        this.runSelectAllWrap = this.root.querySelector('[data-r="run-select-all-wrap"]');
        this.runSelectAllCb = this.root.querySelector('[data-r="run-select-all-cb"]');

        this.populateTaskSelect(this.globalTask, this.taskTypeWidget?.value);
        this.syncNegativeFromWidget();
        this.syncOutputUIFromTimeline();
        bindImageBatchEvents(this);
    },
    renderImageBatchGroups() {
        renderImageBatchGroups(this);
    },
    normalizeImageBatchSegments() {
        normalizeImageBatchSegments(this);
    },
    syncNegativeFromWidget() {
        const v = this.negativePromptWidget?.value ?? "";
        if (this.globalNegative) this.globalNegative.value = v;
        if (this.segNegative) this.segNegative.value = v;
    },
    destroy() {
        clearTimeout(this._syncTimer);
        clearTimeout(this._settleRenderTimer);
        clearTimeout(this._settleRenderLateTimer);
        clearTimeout(this._promptRenderTimer);
        this._promptRenderTimer = null;
        this._settleRenderTimer = null;
        this._settleRenderLateTimer = null;
        cancelAnimationFrame(this._resizeRaf);
        cancelAnimationFrame(this._playRaf);
        this._resizeObserver?.disconnect();
        this._unsubLocale?.();
        this._unsubLocale = null;
        this._closeBdModal();
        teardownPromptImageMentions(this.root);
        this._clearPreviewVideos(true);
        this._previewVideos?.clear();
        try {
            this._previewVideo?.pause();
            this._previewVideo?.removeAttribute("src");
            this._previewVideo?.load();
        } catch { /* ignore */ }
        this._previewVideo?.remove();
        this._previewVideo = null;
        window.removeEventListener("mousemove", this._onMouseMove);
        window.removeEventListener("mouseup", this._onMouseUp);
        this.canvas?.removeEventListener("mousemove", this._onCanvasHover);
        this.canvas?.classList.remove("bd-grab", "bd-grabbing");
        window.removeEventListener("keydown", this._onKeyDown, true);
        this.root?.remove();
        this.root = null;
        if (this.node?._minimaxEditor === this) this.node._minimaxEditor = null;
        if (this.domWidget?._minimaxEditor === this) this.domWidget._minimaxEditor = null;
    },
    widget(name) { return this.node.widgets?.find((w) => w.name === name); }
};
