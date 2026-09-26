/** The editor style block, injected once per page.
 *
 * The timeline is drawn on canvas, but the panels, buttons and dialogs around it
 * are plain DOM, so the plugin ships one style block for them. It is a single JS
 * string rather than a .css file because ComfyUI has no CSS entry point for
 * extensions -- this module is the one place that owns it.
 *
 * Moved verbatim from minimax_timeline.js.
 */


import { FL2V_STYLES } from "../minimax_fl2v.js";
import { IMAGE_BATCH_STYLES } from "../minimax_image_batch.js";
export const STYLES = `/* min-height = content only; height:100% fills LiteGraph free space without raising
   getMinHeight (avoids Vue-node ResizeObserver feedback growth).
   overflow:hidden keeps run-status from painting past the node bottom edge. */
.mmx-host{width:100%;box-sizing:border-box;display:flex;flex-direction:column;min-height:var(--comfy-widget-min-height,0px);height:100%;max-height:100%;overflow:hidden}
/* Default: fill allocated box. Batch-fill mode stretches list into leftover space. */
.bd-wrap{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;color:#e0e0e0;font-size:11px;display:flex;flex-direction:column;gap:6px;width:100%;box-sizing:border-box;position:relative;min-height:0;height:100%;flex:1 1 auto;overflow:hidden}
.bd-wrap.bd-batch-fill{height:100%!important;min-height:0!important;max-height:100%;flex:1 1 0;overflow:hidden}
.bd-main{flex:0 1 auto;min-height:0;display:flex;flex-direction:column;gap:6px;width:100%}
/*
 * Batch is inside .bd-main (sibling of .bd-split which holds 公共参数).
 * Main grows with the node; .bd-split may shrink/scroll so .bd-batch always keeps space.
 */
.bd-wrap.bd-batch-fill .bd-main{flex:1 1 0;min-height:0;overflow:hidden}
.bd-wrap.bd-batch-fill .bd-main>:not(.bd-batch):not(.bd-split){flex:0 0 auto}
/* 公共参数区：可收缩+内部滚动，避免展开后把素材组挤出视口 */
.bd-wrap.bd-batch-fill .bd-main>.bd-split{
  flex:0 1 auto;min-height:0;max-height:42%;overflow:auto;width:100%
}
.bd-wrap.bd-batch-fill .bd-main>.bd-batch:not(.hidden){
  flex:1 1 0;min-height:0;overflow:hidden;display:flex;flex-direction:column
}
.bd-wrap.bd-batch-fill .bd-batch-toolbar,.bd-wrap.bd-batch-fill .bd-batch-i2v-notice,.bd-wrap.bd-batch-fill .bd-batch-picker{flex-shrink:0}
.bd-wrap.bd-batch-fill .bd-batch-list{
  flex:1 1 0;min-height:0;max-height:none!important;overflow-y:auto;height:auto;
  display:flex;flex-direction:column
}
.bd-wrap.bd-batch-fill .bd-run-status{flex:0 0 auto;margin-top:0;flex-shrink:0;
  position:sticky;bottom:0;z-index:3;background:var(--bg-panel, #1a1a2e);border-top:1px solid #333}
/* Fixed min so progress text wrap does not change node chrome height every tick. */
.bd-run-status{min-height:52px;box-sizing:border-box}
/* Solo material group fills the viewport by default, but may grow beyond it when
   the user drags the rich prompt editor. The list then scrolls instead of
   clipping the editor or forcing its height back to auto. */
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card{flex:1 1 auto;min-height:0;align-self:stretch}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-r2v{
  display:flex;flex-direction:column;flex:0 0 auto!important;min-height:100%;height:auto!important
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-r2v-body{
  flex:0 0 auto;min-height:0;max-height:none;align-items:start
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-r2v-main{
  flex:0 0 auto;min-height:320px;height:auto;max-height:none
}
/* 素材组列：保持内部滚动、不被外层高度裁剪。
   具体高度（1280）由卡片内 .bd-batch-r2v .bd-batch-r2v-assets 统一定义，
   这里不再重复写死，避免两个数字打架。 */
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-r2v-assets{
  flex:1 1 auto;overflow-y:auto
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-batch-prompts{
  flex:0 0 auto;min-height:140px;max-height:none;overflow:visible
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-token-wrap{
  flex:0 0 auto;min-height:120px;max-height:none;height:auto;overflow:visible
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo .bd-token-editor{
  flex:0 0 auto;min-height:120px;max-height:none;height:360px;overflow:auto;resize:vertical
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-plain,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-source,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v){
  /* Header stays compact; leftover height goes to the prompt row (not a blank gap). */
  grid-template-rows:auto minmax(0,1fr);
  align-content:stretch
}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-plain .bd-batch-head,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-source .bd-batch-head,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-head{align-self:start}
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-plain .bd-batch-prompts,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-source .bd-batch-prompts,
.bd-wrap.bd-batch-fill .bd-batch-list.bd-batch-solo>.bd-batch-card.bd-batch-refs:not(.bd-batch-r2v) .bd-batch-prompts{
  height:100%;min-height:0;max-height:100%;overflow:hidden;align-self:stretch
}
.bd-modal-overlay{position:absolute;inset:0;z-index:200;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:10px;box-sizing:border-box;border-radius:6px}
.bd-modal-overlay-fixed{position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}
.bd-modal-overlay-fixed .bd-modal{max-width:none;max-height:calc(100vh - 48px)}
.bd-modal{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:12px;width:100%;max-width:460px;max-height:calc(100% - 8px);display:flex;flex-direction:column;gap:10px;box-shadow:0 10px 28px rgba(0,0,0,.5)}
.bd-modal-title{color:#e0e0e0;font-size:12px;font-weight:600;line-height:1.35}
.bd-modal-body{color:#aaa;font-size:11px;line-height:1.5;white-space:pre-wrap}
.bd-modal-body.hidden{display:none}
.bd-modal-list{flex:1;min-height:140px;max-height:240px;overflow:auto;background:#181818;border:1px solid #333;border-radius:6px;padding:4px;display:flex;flex-direction:column;gap:2px}
.bd-modal-list.hidden{display:none}
.bd-modal-item{padding:7px 8px;border-radius:4px;cursor:pointer;color:#ccc;font-size:11px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent}
.bd-modal-item:hover{background:#252525;color:#eee}
.bd-modal-item.selected{background:#2a2a2a;border-color:#4fff8f;color:#fff}
.bd-modal-actions{display:flex;gap:8px;justify-content:flex-end;flex-shrink:0}
.bd-modal-wide{max-width:720px;min-width:420px}
.bd-modal-overlay-fixed .bd-modal-wide{width:auto;min-width:520px}
.bd-seg-export-mode{display:flex;align-items:center;gap:14px;font-size:12px;color:#ccc;flex-wrap:wrap}
.bd-seg-export-mode-label{font-weight:600;color:#e0e0e0}
.bd-seg-export-mode label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.bd-seg-export-mode input[type="radio"]{accent-color:#4fff8f;width:14px;height:14px}
.bd-seg-export-hint{color:#888;font-size:11px;line-height:1.4}
.bd-seg-export-hint.warn{color:#e0b34d}
.bd-seg-export-count{color:#4fff8f;font-size:12px;font-weight:600}
.bd-modal-overlay-fixed .bd-modal-list{flex:1;min-height:240px;max-height:min(480px,calc(100vh - 220px))}
.bd-seg-export-item{display:flex;align-items:center;gap:10px;padding:8px 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-radius:4px;color:#ccc;font-size:12px;line-height:1.4;border:1px solid transparent}
.bd-seg-export-item:hover{background:#252525;color:#eee}
.bd-seg-export-item.disabled{opacity:.42;cursor:not-allowed}
.bd-seg-export-item.disabled .bd-seg-export-cb{cursor:not-allowed}
.bd-seg-export-item .bd-seg-export-cb{accent-color:#4fff8f;width:15px;height:15px;flex-shrink:0;cursor:pointer}
.bd-seg-export-name{font-weight:600;color:#e0e0e0;flex-shrink:0;min-width:42px}
.bd-seg-export-badges{display:inline-flex;align-items:center;gap:10px;flex-wrap:nowrap;flex:1;min-width:0;overflow:hidden}
.bd-seg-dots{display:inline-flex;align-items:center;gap:3px;flex-shrink:0}
.bd-seg-dot{width:9px;height:9px;border-radius:50%;background:#3a3a3a;flex-shrink:0;box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)}
.bd-seg-dot.ready{background:#7dffa0;box-shadow:0 0 5px rgba(125,255,160,.55)}
.bd-seg-dot.partial{background:#ffd27d;box-shadow:0 0 4px rgba(255,210,125,.45)}
.bd-seg-dot.empty{background:#3a3a3a}
.bd-seg-dot-done{outline:2px solid #6ea8ff;outline-offset:1px}
.bd-seg-export-summary{margin-left:auto;color:#9aa3ad;font-size:11px;flex-shrink:0;font-variant-numeric:tabular-nums;white-space:nowrap}
.bd-seg-export-legend{display:flex;align-items:center;gap:6px;color:#888;font-size:10px;flex-wrap:wrap;line-height:1.6}
.bd-seg-export-legend .bd-seg-dot{position:relative}
.bd-seg-export-badge{font-size:10px;padding:2px 7px;border-radius:9px;line-height:1.5;white-space:nowrap;flex-shrink:0;font-weight:500;letter-spacing:.2px}
.bd-seg-export-badge.ok{color:#9dffb3;background:rgba(125,255,160,.10);border:1px solid rgba(125,255,160,.30)}
.bd-seg-export-badge.warn{color:#ffdca0;background:rgba(255,210,125,.10);border:1px solid rgba(255,210,125,.30)}
.bd-seg-export-badge.muted{color:#9aa;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.10)}
.bd-seg-export-toast{position:fixed;left:50%;bottom:56px;transform:translateX(-50%) translateY(20px);background:#1f3d2b;color:#9dffb3;border:1px solid #2f6b40;border-radius:8px;padding:10px 18px;font-size:13px;z-index:10000;opacity:0;pointer-events:none;transition:opacity .25s,transform .25s;box-shadow:0 6px 20px rgba(0,0,0,.4)}
.bd-seg-export-toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.bd-media-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
.bd-media-head .bd-modal-title{flex:1;min-width:0;padding-top:4px}
.bd-media-head-actions{display:flex;gap:8px;flex-shrink:0;flex-wrap:wrap;justify-content:flex-end}
.bd-media-status{color:#999;font-size:11px;line-height:1.4;min-height:15px}
.bd-media-modal{max-width:860px}
.bd-media-modal .bd-btn-primary{background:#2f9e44;border-color:#3cb054;color:#fff}
.bd-media-modal .bd-btn-primary:hover{background:#38b04a;border-color:#4bc45c;color:#fff}
.bd-media-body{display:grid;grid-template-columns:minmax(320px,1.2fr) minmax(240px,.8fr);gap:10px;min-height:280px}
.bd-media-left,.bd-media-right{display:flex;flex-direction:column;gap:8px;min-width:0}
.bd-media-table{width:100%;min-height:220px;max-height:320px;background:#141414;border:1px solid #333;border-radius:6px;color:#eee;box-sizing:border-box;flex:1;display:flex;flex-direction:column;overflow:hidden;outline:none}
.bd-media-thead{display:grid;grid-template-columns:minmax(0,1fr) 86px 128px;flex-shrink:0;border-bottom:1px solid #333;background:#1a1a1a}
.bd-media-th{appearance:none;background:transparent;border:none;color:#8e8e8e;font-size:11px;text-align:left;padding:8px 10px;cursor:pointer;display:flex;align-items:center;gap:5px;min-width:0;font-family:inherit;line-height:1.35}
.bd-media-th:hover{color:#ddd}
.bd-media-th.is-active{color:#d8d8d8}
.bd-media-sort{display:inline-block;width:0;height:0;border-left:4px solid transparent;border-right:4px solid transparent;opacity:0;flex-shrink:0}
.bd-media-th.is-active .bd-media-sort{opacity:1;border-top:5px solid #6ea8ff;border-bottom:0}
.bd-media-th.is-active.is-asc .bd-media-sort{border-top:0;border-bottom:5px solid #6ea8ff}
.bd-media-tbody{flex:1;overflow:auto;min-height:0}
.bd-media-tr{display:grid;grid-template-columns:minmax(0,1fr) 86px 128px;align-items:center;cursor:pointer;border-bottom:1px solid #262626;color:#ddd;font-size:11px;line-height:1.35}
.bd-media-table.bd-media-nodims .bd-media-thead,
.bd-media-table.bd-media-nodims .bd-media-tr{grid-template-columns:minmax(0,1fr) 128px}
/* 批量选择模式：表格多一列复选框，列宽固定 24px 保持与文件名/尺寸/时间对齐 */
.bd-media-table.bd-media-multi .bd-media-thead,
.bd-media-table.bd-media-multi .bd-media-tr{grid-template-columns:24px minmax(0,1fr) 86px 128px}
.bd-media-table.bd-media-multi.bd-media-nodims .bd-media-thead,
.bd-media-table.bd-media-multi.bd-media-nodims .bd-media-tr{grid-template-columns:24px minmax(0,1fr) 128px}
.bd-media-th-cb{padding:0;display:flex;align-items:center;justify-content:center;cursor:default}
.bd-media-td-cb{display:flex;align-items:center;justify-content:center;padding:7px 6px}
.bd-media-cb{cursor:pointer}
.bd-media-tr:hover{background:#222}
/* selected = 当前预览项（底色）；checked = 已勾选（左侧蓝条，批量模式） */
.bd-media-tr.selected{background:#2c2c2c}
.bd-media-tr.checked{background:#26303a;box-shadow:inset 3px 0 0 #4a9fd8}
.bd-media-td{padding:8px 10px;min-width:0}
.bd-media-td-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#eee}
.bd-media-td-dims,.bd-media-td-time{color:#9a9a9a;white-space:nowrap;font-variant-numeric:tabular-nums}
.bd-media-empty-row{padding:18px 10px;color:#666;font-size:11px;text-align:center}
.bd-media-preview{flex:1;min-height:220px;background:#111;border:1px solid #333;border-radius:6px;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
.bd-media-preview img,.bd-media-preview video{display:block;width:100%;height:100%;object-fit:contain;background:#000}
.bd-media-preview-empty{padding:18px;color:#666;font-size:11px;line-height:1.45;text-align:center}
.bd-media-meta{display:flex;flex-direction:column;gap:4px;color:#9a9a9a;font-size:10px;line-height:1.45;word-break:break-all}
.bd-toolbar-wrap{display:flex;flex-direction:column;gap:4px;width:100%}
.bd-toolbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;width:100%}
.bd-actions{display:flex;gap:6px;flex-wrap:wrap;align-items:center;flex:1;min-width:0}
.bd-smart-split-msg{width:100%;box-sizing:border-box;font-size:11px;line-height:1.4;color:#f66;padding:0 2px;min-height:0}
.bd-smart-split-msg.hidden{display:none!important}
.bd-smart-split-msg.ok{color:#8c8}
.bd-external-groups-msg{width:100%;box-sizing:border-box;font-size:11px;line-height:1.45;color:#9ad;padding:8px 10px;margin:0 0 4px;background:#152018;border:1px solid #2f4a38;border-radius:6px}
.bd-external-groups-msg.hidden{display:none!important}
.bd-wrap.bd-external-groups .bd-batch-card,.bd-wrap.bd-external-groups .bd-fl2v-shot{opacity:.48;pointer-events:none}
.bd-wrap.bd-external-groups .bd-run-select-bar,.bd-wrap.bd-external-groups .bd-batch-run-check,.bd-wrap.bd-external-groups .bd-run-select-all-wrap{pointer-events:auto;opacity:1}
.bd-wrap.bd-external-groups .bd-batch-card .bd-batch-run-check{pointer-events:auto;opacity:1}
/* External mode: duration/delete stay non-interactive; allow media preview playback. */
.bd-wrap.bd-external-groups .bd-batch-del,.bd-wrap.bd-external-groups .bd-batch-fc input{pointer-events:none!important;opacity:.55}
.bd-wrap.bd-external-groups .bd-r2v-play,.bd-wrap.bd-external-groups .bd-batch-video video,.bd-wrap.bd-external-groups .bd-batch-audio audio,.bd-wrap.bd-external-groups .bd-r2v-thumb{pointer-events:auto;opacity:1}
.bd-stage{width:100%;box-sizing:border-box;background:#0c0c0c;border:1px solid #222;border-bottom:none;border-radius:6px 6px 0 0;overflow:hidden;position:relative;min-height:120px;max-height:280px;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}
.bd-stage.hidden{display:none!important}
.bd-stage-video,.bd-stage-img{width:100%;height:100%;max-height:280px;object-fit:contain;background:#000;display:block}
.bd-stage-img.hidden,.bd-stage-video.hidden{display:none!important}
.bd-stage-empty{color:#555;font-size:11px;pointer-events:none}
.bd-stage-badge{position:absolute;left:8px;bottom:8px;padding:2px 7px;border-radius:3px;background:rgba(0,0,0,.65);color:#ccc;font-size:10px;line-height:1.4;cursor:pointer;user-select:none}
.bd-stage-badge:hover{color:#fff;background:rgba(0,0,0,.8)}
.bd-frame-jump{display:inline-flex;align-items:center;gap:4px;color:#ddd;font-size:11px;white-space:nowrap;font-variant-numeric:tabular-nums}
.bd-frame-jump .bd-frame-input{width:64px;background:#181818;border:1px solid #444;border-radius:4px;color:#eee;padding:4px 4px;font-size:11px;text-align:center;-moz-appearance:textfield}
.bd-frame-jump .bd-frame-input:focus{border-color:#4fff8f;outline:none}
.bd-frame-jump .bd-frame-input::-webkit-outer-spin-button,.bd-frame-jump .bd-frame-input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.bd-frame-jump .bd-frame-total{color:#888;min-width:2.5em}
.bd-controls{width:100%;box-sizing:border-box;background:#151515;border:1px solid #222;border-radius:0 0 6px 6px;padding:8px 10px;margin-top:0;flex-shrink:0}
.bd-stage.hidden+.bd-controls{border-radius:6px;border-color:#333;background:#1e1e1e}
.bd-viewport{width:100%;max-width:100%;min-width:0;overflow-x:hidden;overflow-y:hidden;border-radius:6px;border:1px solid #111;background:#2a2a2a;box-sizing:border-box;flex-shrink:0}
.bd-viewport.bd-zoomed{overflow-x:auto;scrollbar-width:thin;scrollbar-color:#555 #1a1a1a}
.bd-viewport.bd-zoomed::-webkit-scrollbar{height:10px}
.bd-viewport.bd-zoomed::-webkit-scrollbar-track{background:#1a1a1a;border-radius:5px}
.bd-viewport.bd-zoomed::-webkit-scrollbar-thumb{background:#555;border-radius:5px}
.bd-viewport.bd-zoomed::-webkit-scrollbar-thumb:hover{background:#777}
/* object-fit:fill + mismatched CSS/bitmap aspect stretches thumbs (esp. under graph zoom). */
.bd-canvas{display:block;width:100%;min-width:100%;height:auto;cursor:pointer;box-sizing:border-box;flex-shrink:0;object-fit:fill}
.bd-canvas.bd-grab{cursor:grab}
.bd-canvas.bd-grabbing{cursor:grabbing}
.bd-output{width:100%;box-sizing:border-box;display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:6px 8px;background:#1e1e1e;border:1px solid #333;border-radius:6px}
.bd-out-audio-wrap{display:inline-flex;align-items:center;gap:6px}
.bd-split{display:block;width:100%;box-sizing:border-box;min-width:0}
.bd-r2v-common-hint{margin:0 0 8px;font-size:11px;line-height:1.4;color:#9ab;opacity:.95}
.bd-panel.bd-r2v-common-panel{border:1px solid #3a4a5a;background:linear-gradient(180deg,#1a222c 0%,#151a20 100%)}
.bd-r2v-common-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 8px}
.bd-r2v-common-titles{display:flex;flex-direction:column;gap:2px;min-width:0;flex:1}
.bd-r2v-common-titles b{margin:0}
.bd-r2v-common-status{display:none;font-size:11px;color:#8a9}
.bd-r2v-common-status.on{color:#8fdfb0}
.bd-panel.bd-r2v-common-panel .bd-r2v-common-status{display:inline}
.bd-r2v-common-actions{display:none;align-items:center;gap:8px;flex:0 0 auto}
.bd-panel.bd-r2v-common-panel .bd-r2v-common-actions{display:flex}
.bd-btn.bd-r2v-common-toggle,.bd-btn.bd-r2v-common-fold{display:inline-block;flex:0 0 auto;padding:5px 10px;font-size:12px;border-radius:6px;border:1px solid #4a6a8a;background:#243040;color:#d8e6f5;cursor:pointer}
.bd-btn.bd-r2v-common-fold{border-color:#3a4a5a;background:#1c2430}
.bd-btn.bd-r2v-common-toggle:hover,.bd-btn.bd-r2v-common-fold:hover{border-color:#6a9aca;background:#2c3c50}
.bd-btn.bd-r2v-common-toggle.on{border-color:#7a3a3a;background:#301a1a;color:#f0c0c0}
.bd-btn.bd-r2v-common-toggle.on:hover{border-color:#a05050;background:#3a2020}
.bd-panel.bd-r2v-common-panel.bd-r2v-common-collapsed{padding-bottom:10px}
.bd-panel.bd-r2v-common-panel.bd-r2v-common-collapsed .bd-r2v-common-body{display:none!important}
.bd-panel.bd-r2v-common-panel .bd-r2v-common-body{min-width:0}
.bd-panel.bd-r2v-common-panel .bd-refs-col{height:auto;min-height:0}
.bd-panel.bd-r2v-common-panel .bd-rv2v-layout .bd-ref{min-height:72px}
.bd-panel.bd-r2v-common-panel .bd-rv2v-layout .bd-ref-audio{min-height:44px}
.bd-panel.bd-r2v-common-panel .bd-rv2v-layout .bd-ref-video{min-height:0}
.bd-player{display:flex;align-items:center;gap:10px;flex-wrap:wrap;width:100%}
.bd-btn{background:#222;color:#e0e0e0;border:1px solid #111;border-radius:4px;padding:6px 12px;font-size:11px;line-height:1.35;box-sizing:border-box;cursor:pointer;display:inline-flex;align-items:center}
.bd-actions>.bd-btn{height:29px;min-height:29px}
.bd-btn:hover{background:#333;border-color:#555}
.bd-btn-danger:hover{background:#4a1515;border-color:#c44;color:#faa}
.bd-split-edit-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;width:100%;box-sizing:border-box;padding:6px 10px;margin:0 0 4px;background:#241818;border:1px solid #633;border-radius:6px}
.bd-split-edit-bar.hidden{display:none!important}
.bd-split-edit-bar .bd-split-edit-hint{flex:1;min-width:140px;font-size:11px;line-height:1.35;color:#f88}
.bd-btn-del-split{background:#3a2020;border-color:#e66;color:#f88}
.bd-btn-del-split:hover{background:#4a1515;border-color:#f88;color:#fcc}
.bd-btn-sm{padding:3px 8px;font-size:10px}
.bd-btn-run-select.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f}
.bd-output .bd-btn-live-preview{margin-left:auto;background:#222;border-color:#333;color:#aaa;white-space:nowrap;height:29px;min-height:29px;padding:4px 12px}
.bd-output .bd-btn-live-preview:hover{background:#2a2a2a;border-color:#555;color:#ddd}
.bd-output .bd-btn-live-preview.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-live-sample{width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:8px;padding:10px 12px;background:linear-gradient(165deg,#1a1a1a 0%,#121212 100%);border:1px solid #333;border-radius:10px;flex-shrink:0}
.bd-live-sample.hidden{display:none!important}
.bd-live-sample.receiving{border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-live-sample-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}
.bd-live-sample-head b{color:#f0f0f0;font-size:12px;font-weight:650;letter-spacing:.02em}
.bd-live-sample-head .bd-meta{color:#888;font-size:11px}
.bd-live-sample-body{position:relative;width:100%;min-height:220px;max-height:360px;flex:1 1 auto;background:#0a0a0a;border:1px solid #262626;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.bd-live-sample-body img{width:100%;height:100%;max-width:100%;max-height:360px;object-fit:contain;display:block}
.bd-live-sample-body img.hidden{display:none!important}
.bd-live-sample-empty{color:#666;font-size:12px;text-align:center;padding:16px;line-height:1.45}
.bd-live-sample-empty.hidden{display:none!important}
.bd-live-sample-badge{position:absolute;left:10px;bottom:10px;padding:3px 8px;border-radius:999px;background:rgba(0,0,0,.75);color:#cfcfcf;font-size:11px;pointer-events:none}
.bd-live-sample-badge.hidden{display:none!important}
.bd-main>.bd-live-sample{margin:0 0 4px}
.bd-run-select-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:10px;color:#aaa}
.bd-run-select-all-wrap{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#aaa;cursor:pointer;user-select:none;margin-left:2px}
.bd-run-select-all-wrap.hidden{display:none!important}
.bd-run-select-all-wrap input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-run-select-bar.hidden{display:none!important}
.bd-batch-run-check{margin-right:6px;width:14px;height:14px;cursor:pointer;accent-color:#4fff8f;flex-shrink:0}
.bd-btn-primary{background:#1a3a2a;border-color:#4fff8f;color:#4fff8f}
.bd-mode{display:flex;border:1px solid #333;border-radius:4px;overflow:hidden}
.bd-mode button{border:none;background:#222;color:#aaa;padding:6px 12px;font-size:11px;cursor:pointer}
.bd-mode button.active{background:#333;color:#fff}
.bd-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex-shrink:0}
.bd-tl-zoom{display:inline-flex;align-items:center;gap:6px;flex-shrink:0}
.bd-tl-zoom.hidden{display:none!important}
.bd-btn.bd-btn-zoom.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-tl-zoom-slider{width:128px;height:18px;margin:0;accent-color:#4fff8f;cursor:pointer;flex-shrink:0;touch-action:none}
.bd-bounds,.bd-timecode{color:#aaa;font-size:11px}
.bd-timecode{color:#fff;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
.bd-player .bd-timecode{min-width:88px;font-size:11px;color:#ddd}
.bd-icon-btn{background:#2a2a2a;border:1px solid #444;color:#eee;cursor:pointer;padding:6px 10px;border-radius:4px}
.bd-icon-btn.active{background:#1a3a2a;color:#4fff8f;border-color:#4fff8f;box-shadow:0 0 0 1px rgba(79,255,143,.35)}
.bd-seek{flex:1;min-width:120px;height:6px}
.bd-panel{width:100%;box-sizing:border-box;background:#222;border:1px solid #111;border-radius:6px;padding:8px;display:flex;flex-direction:column;gap:6px}
.bd-panel.bd-rv2v-panel,.bd-panel.bd-v2v-panel{background:linear-gradient(165deg,#1c1c1c 0%,#141414 52%,#111 100%);border:1px solid #2c2c2c;border-radius:12px;padding:12px 14px;box-shadow:inset 0 1px 0 rgba(255,255,255,.035);gap:10px}
.bd-panel.bd-rv2v-panel>b,.bd-panel.bd-v2v-panel>b,.bd-seg-head>b{color:#f0f0f0;font-size:13px;font-weight:650;letter-spacing:.02em}
.bd-seg-head{display:flex;align-items:center;justify-content:flex-start;gap:10px;flex-wrap:wrap;min-width:0}
.bd-seg-head>b{flex-shrink:0;margin:0}
.bd-seg-refsize{display:inline-flex;align-items:center;gap:6px;color:#c8c8c8;font-size:11px;white-space:nowrap;margin-left:auto;flex-shrink:0}
.bd-seg-refsize select{max-width:88px}
.bd-seg-continuity{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#9ab;cursor:pointer;user-select:none;flex-shrink:0}
.bd-seg-continuity input{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#6ab0ff}
.bd-seg-head .bd-meta,.bd-panel.bd-v2v-panel .bd-seg-head .bd-meta,.bd-panel.bd-rv2v-panel .bd-seg-head .bd-meta{color:#8a8a8a;font-size:11px;line-height:1.45;padding:0;min-width:0}
.bd-prompt-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(110px,38%);gap:8px;align-items:stretch}
.bd-prompt-layout>.bd-prompt-col{order:1}
.bd-prompt-layout>.bd-refs-col{order:2}
.bd-prompt-layout.bd-rv2v-layout{grid-template-columns:minmax(240px,.85fr) minmax(0,1.4fr);gap:12px}
.bd-prompt-layout.bd-rv2v-layout>.bd-refs-col{order:1}
.bd-prompt-layout.bd-rv2v-layout>.bd-prompt-col{order:2}
/* rv2v live preview: under prompt (same stack as r2v right column) */
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live>.bd-prompt-col{gap:10px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live>.bd-prompt-col .bd-prompt{min-height:160px;flex:1 1 auto}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live>.bd-prompt-col>.bd-live-sample{margin:0;padding:8px 10px;min-height:0;flex:0 0 auto;border-radius:10px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live .bd-live-sample-body{min-height:180px;max-height:280px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live .bd-live-sample-body img{width:100%;max-height:280px;object-fit:contain}
.bd-prompt-layout.bd-v2v-layout{grid-template-columns:1fr;gap:0}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live{grid-template-columns:minmax(0,1.25fr) minmax(240px,.9fr);gap:12px;align-items:stretch}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-prompt-col{order:1;min-height:220px}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-prompt-col .bd-prompt{flex:1 1 auto;min-height:180px}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-refs-col{display:none!important}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-live-sample{order:2;margin:0;height:100%;min-height:220px;align-self:stretch}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live .bd-live-sample-body{flex:1 1 auto;min-height:180px;max-height:none}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live .bd-live-sample-body img{max-height:100%}
.bd-prompt-col{display:flex;flex-direction:column;gap:5px;min-width:0}
.bd-rv2v-layout .bd-prompt-col,.bd-v2v-layout .bd-prompt-col{background:#0c0c0c;border:1px solid #262626;border-radius:10px;padding:10px 12px;gap:6px;min-height:220px}
.bd-v2v-layout .bd-prompt-col{min-height:200px}
.bd-prompt-col .bd-label,.bd-refs-col .bd-label{color:#888;font-size:10px;line-height:1.2;flex-shrink:0}
.bd-rv2v-layout .bd-prompt-col .bd-label,.bd-v2v-layout .bd-prompt-col .bd-label{color:#eaeaea;font-size:11px;font-weight:700;letter-spacing:.02em}
.bd-wrap.locale-en .bd-rv2v-layout .bd-prompt-col .bd-label,.bd-wrap.locale-en .bd-v2v-layout .bd-prompt-col .bd-label{text-transform:uppercase;letter-spacing:.08em}
.bd-prompt{width:100%;min-height:96px;background:#181818;border:1px solid #333;border-radius:6px;color:#eee;padding:8px;resize:vertical;font-size:12px;box-sizing:border-box;font-family:inherit;line-height:1.35;flex:1}
.bd-prompt-col .bd-token-wrap{flex:1 1 auto;min-height:96px;width:100%}
.bd-ref.bd-ref-flash,.bd-batch-ref.bd-ref-flash,.bd-ref-audio.bd-ref-flash,.bd-batch-audio.bd-ref-flash,.bd-batch-video.bd-ref-flash{outline:2px solid #4fff8f;outline-offset:1px;border-color:#4fff8f!important}
.bd-rv2v-layout .bd-prompt,.bd-v2v-layout .bd-prompt{min-height:220px;background:#101010;border-color:#2e2e2e;border-radius:8px;padding:10px;font-size:12px;line-height:1.45}
.bd-v2v-layout .bd-prompt{min-height:180px}
.bd-prompt-negative{display:none!important}
.bd-refs-col{display:flex;flex-direction:column;gap:4px;min-width:0;height:100%}
.bd-rv2v-layout .bd-refs-col{gap:10px}
.bd-refs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px;width:100%;flex:1;align-content:start}
.bd-rv2v-layout .bd-refs{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;flex:0}
.bd-ref{position:relative;width:100%;aspect-ratio:1;min-width:0;max-height:64px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;font-size:9px;color:#666;transition:border-color .15s,background .15s}
.bd-rv2v-layout .bd-ref{max-height:none;min-height:0;border-radius:8px;border:1px dashed #333;background:#080808;color:#555;font-size:10px}
.bd-ref.has-img{cursor:grab;border-style:solid}
.bd-rv2v-layout .bd-ref.has-img{border-color:#3a3a3a;background:#000}
.bd-ref.has-img:active{cursor:grabbing}
.bd-ref:hover{border-color:#7a9cff;background:#1a1a1a}
.bd-rv2v-layout .bd-ref:hover{border-color:#5a5a5a;background:#101010}
.bd-ref .bd-ref-tag{position:absolute;inset:auto 0 3px 0;text-align:center;font-size:9px;color:#777;pointer-events:none;line-height:1}
.bd-ref.has-img .bd-ref-tag{display:none}
.bd-rv2v-layout .bd-ref .bd-ref-tag,.bd-rv2v-layout .bd-ref .cap{position:absolute;left:0;right:0;bottom:0;padding:14px 6px 5px;background:linear-gradient(180deg,transparent,rgba(0,0,0,.78));color:#ddd;font-size:10px;font-weight:600;text-align:center;pointer-events:none;z-index:2}
.bd-rv2v-layout .bd-ref:not(.has-img) .bd-ref-tag,.bd-rv2v-layout .bd-ref:not(.has-img) .cap{position:static;padding:0;background:none;color:#666;font-weight:500}
.bd-rv2v-layout .bd-ref.has-img .bd-ref-tag{display:block}
.bd-rv2v-layout .bd-ref img{object-fit:contain;object-position:center;background:#000}
.bd-rv2v-layout .bd-ref .dot{position:absolute;left:6px;top:6px;width:7px;height:7px;border-radius:50%;background:#4fff8f;box-shadow:0 0 0 2px rgba(0,0,0,.5);z-index:2}
.bd-rv2v-layout .bd-ref .x{top:4px;right:4px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-rv2v-layout .bd-ref:hover .x,.bd-rv2v-layout .bd-ref:focus-within .x{display:flex}
.bd-rv2v-layout .bd-ref.bd-r2v-pic-hidden{display:none!important}
.bd-rv2v-layout .bd-refs-images-wrap,.bd-rv2v-layout .bd-ref-audios-wrap,.bd-rv2v-layout .bd-ref-videos-wrap{margin-top:0}
.bd-select{background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:4px 6px;font-size:11px;max-width:240px;box-sizing:border-box}
.bd-actions>.bd-select{padding:6px 10px;font-size:11px;line-height:1.35;height:29px;min-height:29px;max-width:min(480px,55vw)}
.bd-ref img{width:100%;height:100%;object-fit:cover}
.bd-ref .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none}
.bd-ref:hover .x{display:block}
.bd-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bd-meta{color:#888;font-size:10px}
.bd-video-tag{color:#4fff8f;font-size:10px}
.bd-num{width:42px;background:#181818;border:1px solid #333;border-radius:4px;color:#eee;padding:5px 4px;font-size:11px;text-align:center;-moz-appearance:textfield}
.bd-num::-webkit-outer-spin-button,.bd-num::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.bd-output label{color:#888;font-size:10px;white-space:nowrap}
.bd-output .bd-out-fixed{display:flex;gap:4px;align-items:center}
.bd-output .bd-out-fixed.hidden{display:none}
/* Do not use margin-top:auto — with an oversized min-height it creates a huge empty gap above the status bar. */
.bd-run-status{width:100%;box-sizing:border-box;padding:8px 10px;background:#151515;border:1px solid #333;border-radius:6px;display:flex;flex-direction:column;gap:5px;margin-top:6px;margin-bottom:0;flex-shrink:0;flex:0 0 auto}
.bd-run-status.idle .bd-run-title{color:#888}
.bd-run-status.active .bd-run-title{color:#4fff8f}
.bd-run-status.done .bd-run-title{color:#7a9cff}
.bd-run-status.error .bd-run-title{color:#f88}
.bd-run-title{font-size:11px;font-weight:600;line-height:1.35}
.bd-run-detail{color:#999;font-size:10px;line-height:1.4}
.bd-run-bars{display:flex;flex-direction:column;gap:3px}
.bd-run-bar{height:5px;background:#2a2a2a;border-radius:3px;overflow:hidden}
.bd-run-bar-fill{height:100%;background:linear-gradient(90deg,#2a6b4a,#4fff8f);border-radius:3px;transition:width .15s ease}
.bd-run-bar-sub .bd-run-bar-fill{background:linear-gradient(90deg,#3a5080,#7a9cff)}
.hidden{display:none!important}
.bd-controls.hidden{display:none!important}
.bd-gen-src{width:100%;min-height:72px;max-height:100px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;align-items:center;justify-content:center;cursor:pointer;overflow:hidden;color:#666;font-size:10px;margin-top:4px;position:relative;box-sizing:border-box}
.bd-gen-src.has-img{border-style:solid;border-color:#444}
.bd-gen-src img{width:100%;height:100%;object-fit:contain;background:#000}
.bd-gen-src .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none;cursor:pointer;z-index:2}
.bd-gen-src.has-img:hover .x{display:block}
.bd-gen-src.has-video{padding:0;cursor:default;align-items:stretch;justify-content:flex-start;flex-direction:column}
.bd-gen-src.has-video .bd-ref-video-preview{width:100%;flex:1;min-height:100px;max-height:220px;object-fit:contain;background:#000;display:block;border-radius:3px}
.bd-gen-src .bd-ref-replace{position:absolute;bottom:4px;left:4px;z-index:3;background:rgba(0,0,0,.72);color:#ccc;border:1px solid #555;border-radius:3px;padding:2px 7px;font-size:9px;cursor:pointer;line-height:1.4}
.bd-gen-src .bd-ref-replace:hover{color:#fff;border-color:#888}
.bd-gen-src.has-video .x{display:block;z-index:3}
.bd-ref-video-col{display:flex;flex-direction:column;gap:4px;min-width:0;width:100%;flex:1}
.bd-ref-video-col .bd-gen-src{min-height:140px;max-height:none;flex:1}
.bd-ref-video-name{word-break:break-all;line-height:1.3}
.bd-ref-audios-wrap,.bd-ref-videos-wrap{display:flex;flex-direction:column;gap:4px;margin-top:6px;width:100%}
.bd-ref-audios,.bd-ref-videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;width:100%}
.bd-rv2v-layout .bd-ref-audios,.bd-rv2v-layout .bd-ref-videos{gap:7px}
.bd-ref-audio,.bd-ref-video{position:relative;min-height:52px;border:1px dashed #555;border-radius:4px;background:#111;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;padding:6px 4px;box-sizing:border-box;font-size:9px;color:#666;text-align:center;line-height:1.25}
.bd-rv2v-layout .bd-ref-audio,.bd-rv2v-layout .bd-ref-video{min-height:0;height:auto;align-items:stretch;justify-content:flex-start;gap:6px;padding:6px;border-radius:8px;border:1px dashed #333;background:#080808;text-align:left;font-size:11px;color:#777}
.bd-ref-audio.has-audio,.bd-ref-video.has-video{border-style:solid;border-color:#4a6a4a;color:#cfe;background:#152015}
.bd-rv2v-layout .bd-ref-audio.has-audio,.bd-rv2v-layout .bd-ref-video.has-video{border-color:#2f4a38;background:#101812;color:#d8ebe0}
.bd-ref-audio:hover,.bd-ref-video:hover{border-color:#7a9cff;background:#1a1a1a}
.bd-rv2v-layout .bd-ref-audio:hover,.bd-rv2v-layout .bd-ref-video:hover{border-color:#555;background:#101010}
.bd-ref-audio.has-audio:hover,.bd-ref-video.has-video:hover{background:#1a2a1a}
.bd-rv2v-layout .bd-ref-audio .bd-r2v-thumb{width:100%;height:44px;border-radius:6px}
.bd-rv2v-layout .bd-ref-video .bd-r2v-thumb,.bd-rv2v-layout .bd-ref-video .bd-r2v-thumb-video{width:100%;height:auto;aspect-ratio:16/9;border-radius:6px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#0c1014;border:1px solid #222;color:#6a7a8a;position:relative}
.bd-rv2v-layout .bd-ref-audio.has-audio .bd-r2v-thumb,.bd-rv2v-layout .bd-ref-video.has-video .bd-r2v-thumb{border-color:#3a5a45;color:#8fdfb0;background:#152018}
.bd-rv2v-layout .bd-ref-audio .bd-r2v-meta,.bd-rv2v-layout .bd-ref-video .bd-r2v-meta{flex-direction:row;align-items:center;justify-content:space-between;gap:4px}
.bd-rv2v-layout .bd-ref-audio audio.bd-r2v-media{position:absolute;width:0;height:0;opacity:0;pointer-events:none}
.bd-rv2v-layout .bd-ref-video video.bd-r2v-media{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none}
.bd-ref-audio .bd-ref-audio-name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9ad;font-size:9px;padding:0 2px}
.bd-rv2v-layout .bd-ref-audio .bd-ref-audio-name,.bd-rv2v-layout .bd-ref-audio .name,.bd-rv2v-layout .bd-ref-video .name{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#666;font-size:10px;padding:0}
.bd-ref-audio .x,.bd-ref-video .x{position:absolute;top:1px;right:3px;color:#f88;font-size:12px;line-height:1;display:none}
.bd-rv2v-layout .bd-ref-audio .x,.bd-rv2v-layout .bd-ref-video .x{top:8px;right:8px;width:20px;height:20px;border-radius:6px;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.72);color:#ff9a9a;font-size:14px;font-weight:700;z-index:3}
.bd-ref-audio:hover .x,.bd-ref-video:hover .x{display:block}
.bd-rv2v-layout .bd-ref-audio:hover .x,.bd-rv2v-layout .bd-ref-video:hover .x{display:flex}
.bd-rv2v-layout .bd-refs-images-wrap.bd-r2v-section,.bd-rv2v-layout .bd-ref-audios-wrap.bd-r2v-section,.bd-rv2v-layout .bd-ref-videos-wrap.bd-r2v-section{display:flex;flex-direction:column;gap:8px}
.bd-r2v-section-count:empty{display:none}
.bd-r2v-section-actions{display:flex;align-items:center;gap:8px;flex-shrink:0}
.bd-r2v-pick-existing{background:transparent;border:1px solid #3a3a3a;color:#c8c8c8;border-radius:6px;padding:2px 8px;font-size:10px;cursor:pointer;line-height:1.4;white-space:nowrap}
.bd-r2v-pick-existing:hover{border-color:#4fff8f;color:#4fff8f}
.bd-r2v-pick-existing:disabled{opacity:.4;cursor:not-allowed;border-color:#333;color:#666}
.bd-prompt-layout:not(.bd-rv2v-layout) .bd-r2v-section-head{display:contents}
.bd-prompt-layout:not(.bd-rv2v-layout) .bd-r2v-section-count,
.bd-prompt-layout:not(.bd-rv2v-layout) .bd-r2v-pick-existing{display:none}
.bd-continuous-ref{display:flex;align-items:center;gap:6px;font-size:10px;color:#aaa;user-select:none;margin-left:8px}
.bd-continuous-ref label{display:flex;align-items:center;gap:4px;cursor:pointer}
.bd-continuous-ref input[type="checkbox"]{width:14px;height:14px;margin:0;cursor:pointer;accent-color:#4fff8f}
.bd-gen-fc-row{display:flex;align-items:center;gap:6px;margin-top:6px}
${IMAGE_BATCH_STYLES}
${FL2V_STYLES}
@media(max-width:768px){
.bd-prompt-layout,.bd-prompt-layout.bd-rv2v-layout,.bd-prompt-layout.bd-v2v-layout,.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live{grid-template-columns:1fr}
.bd-prompt-layout.bd-v2v-layout.bd-v2v-with-live>.bd-live-sample{order:3;min-height:160px}
.bd-prompt-layout.bd-rv2v-layout.bd-rv2v-with-live .bd-live-sample-body{min-height:96px;max-height:140px}
.bd-ref{max-height:64px}
.bd-rv2v-layout .bd-ref{max-height:none}
.bd-v2v-layout .bd-prompt,.bd-rv2v-layout .bd-prompt{min-height:140px}
.bd-media-body{grid-template-columns:1fr}
.bd-media-preview{min-height:180px}
.bd-media-thead,.bd-media-tr{grid-template-columns:minmax(0,1fr) 72px 108px}
.bd-media-table.bd-media-nodims .bd-media-thead,
.bd-media-table.bd-media-nodims .bd-media-tr{grid-template-columns:minmax(0,1fr) 108px}
.bd-media-td,.bd-media-th{padding:7px 8px}
}
`;

let stylesInjected = false;

/** Inject the editor stylesheet once per page.
 *
 * buildDOM used to inline the whole block inside every editor root, so the same ~440
 * lines were parsed once per node on screen. ComfyUI has no CSS entry point for
 * extensions, so this module owns both the sheet and its injection -- the same
 * treatment minimax_prompt_mentions.js gives MENTION_STYLES.
 */
export function ensureEditorStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const el = document.createElement("style");
    el.textContent = STYLES;
    document.head.appendChild(el);
}
