import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { MiniMaxH3DirectorOptEditor } from "./editor/editor.js";
import { coerceTimelineFps } from "./core/dims.js";
import { bindDirectorDomWidgetSizing, destroyDirectorEditor, ensureDirectorDomWidgetWidth, finalizeDirectorWidgetOrder, getDirectorUiHeight, healOversizedDirectorNode, hookTaskTypeWidget, installDirectorClipboardGuard, parseTimeline, patchDirectorDomWidgetLayout, pruneDirectorDomWidgets, scheduleDirectorLayoutSettle, syncDirectorNodeSize } from "./core/editor_lifecycle.js";
import { logicalToSourceFrame } from "./core/frame_map.js";
import { EXTERNAL_COMBINE_NODE_TYPE, EXTERNAL_GROUP_NODE_TYPES, findDirectorNode, getStableWorkflowId, notifyDirectorsSyncExternalGroups } from "./core/graph_refs.js";
import { DIRECTOR_DOM_WIDGET_NAME } from "./core/layout_spec.js";
import { clearAllDirectorRunStatus, isDirectorNodeDef, isMiniMaxH3DirectorOptNode, normalizeDirectorOutputs, patchDirectorWidgetValueMigration, sanitizeAllWidgetValues, sanitizeWidgetValues } from "./core/node_migrations.js";
import { clamp } from "./core/utils.js";
import { DIRECTOR_GROUP_LABEL_KEYS, applyDirectorWidgetLabels } from "./core/widget_labels.js";
import { onLocaleChange, t } from "./minimax_i18n.js";
import { ensureImageBatchTimeline, renderImageBatchGroups, setImageBatchPreview } from "./minimax_image_batch.js";


function drawGroupHeader(ctx, node, widget_width, y, H, label) {
    const margin = 10;
    const barH = Math.max(18, H - 4);
    ctx.fillStyle = "#2e2e2e";
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(margin, y + 2, widget_width - margin * 2, barH, 4);
    } else {
        ctx.rect(margin, y + 2, widget_width - margin * 2, barH);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#d8dce8";
    ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, margin + 10, y + 2 + barH / 2);
}

function makeGroupHeaderWidget(inputName, inputData) {
    const opts = inputData?.[1] || {};
    const i18nKey = DIRECTOR_GROUP_LABEL_KEYS[inputName];
    const label = i18nKey ? t(i18nKey) : (opts.default || opts.label || inputName);
    const el = document.createElement("div");
    el.className = "bd-widget-group";
    el.textContent = label;
    el.style.cssText = [
        "width:100%;box-sizing:border-box;margin:8px 0 4px;padding:6px 10px",
        "border:1px solid #555;border-left:3px solid #7a9cff;border-radius:4px",
        "color:#d8dce8;font-size:11px;font-weight:600;letter-spacing:.02em",
        "background:linear-gradient(180deg,#2e2e2e 0%,#242424 100%)",
        "pointer-events:none;user-select:none",
    ].join(";");
    return {
        name: inputName,
        type: "BDGROUP",
        value: label,
        label: "",
        element: el,
        options: opts,
        _bdGroupHeader: true,
        _mmxGroupI18nKey: i18nKey || null,
        _bdGroupLabel: label,
        draw(ctx, node, widget_width, y, H) {
            const text = this._mmxGroupI18nKey ? t(this._mmxGroupI18nKey) : (this._bdGroupLabel || label);
            drawGroupHeader(ctx, node, widget_width, y, H, text);
        },
        computeSize(width) {
            return [width, 26];
        },
        mouse() {
            return false;
        },
    };
}



/**
 * Match Python ``lib.image_prep.fit_long_edge``:
 * round(dim * scale / stride) * stride — keeps aspect, long side ≤ budget.
 * Stride must be 32 for MiniMax H3 (VAE 16× then 2×2 patch). Stride 16 can
 * yield 496×864 → odd latent width → patchify_video crash under continuity/v2v.
 */


/** Upload a file to ComfyUI input/ (videos use the same endpoint as images). */




/** 0, 5, 10 … below one minute; 1:00, 1:30 … at/after 60s. */







/** Inverse of logicalToSourceFrame for sparse deletes; -1 if source is in a deleted gap. */







/**
 * Only snap *runaway* heights (old infinite-growth corruption).
 * ideal+1200 was far too aggressive: r2v users routinely drag taller, and init heal
 * wiped the workflow-saved size on every Comfy restart (#7 regression).
 */


/** After graph load / init: re-fill once LiteGraph assigns computedHeight from saved size. */

/** Grow once when content min increases (mode switch); never shrink; never use stretch. */






/** Keep one Director DOM widget; drop extras left by double-wrapped onNodeCreated. */



function initDirectorEditor(node) {
    // Must not share Bernini's `_directorDomWidget` — their loadedGraphNode mounts on that key.
    if (!isMiniMaxH3DirectorOptNode(node)) return null;
    const widget = pruneDirectorDomWidgets(node);
    const container = widget?.element;
    if (!container) return null;

    const existing = node._minimaxEditor || widget._minimaxEditor;
    if (existing?.root && existing.container === container) {
        node._minimaxEditor = existing;
        widget._minimaxEditor = existing;
        for (const wrap of [...container.querySelectorAll(":scope > .bd-wrap")]) {
            if (wrap !== existing.root) wrap.remove();
        }
        return existing;
    }
    // Constructor runs before `_minimaxEditor` is assigned; block re-entrant mounts
    // from onConfigure / loadedGraphNode / layout callbacks during buildDOM.
    if (node._minimaxEditorMounting) return existing || null;

    if (existing) destroyDirectorEditor(node);

    node._minimaxEditorMounting = true;
    try {
        for (const wrap of [...container.querySelectorAll(":scope > .bd-wrap")]) wrap.remove();
        hookTaskTypeWidget(node);
        const editor = new MiniMaxH3DirectorOptEditor(node, container, widget);
        node._minimaxEditor = editor;
        widget._minimaxEditor = editor;
        ensureDirectorDomWidgetWidth(node);
        bindDirectorDomWidgetSizing(node, widget, () => node._minimaxEditor);
        // Only clamp true runaway; never steal normal user/workflow height (#7).
        healOversizedDirectorNode(node, editor);
        syncDirectorNodeSize(node, editor);
        scheduleDirectorLayoutSettle(editor);
        return editor;
    } catch (err) {
        console.error("[MiniMax H3Director] UI init failed:", err);
        return node._minimaxEditor || null;
    } finally {
        node._minimaxEditorMounting = false;
    }
}



/** True when focus/target is a Director text field (incl. contenteditable token editor). */

/**
 * Stop Comfy graph copy/paste from firing while typing in Director prompts.
 * contenteditable chips are invisible to Comfy's INPUT/TEXTAREA checks, so Ctrl+V
 * otherwise pastes the last copied nodes beside the Director.
 */



/* ===========================================================================
 * Cross-module contract for MiniMaxH3DirectorOptEditor  (cross-module ABI)
 * ===========================================================================
 * Other feature modules address this class as ``editor.<member>`` — 74 distinct
 * members across ~540 call sites (minimax_image_batch, minimax_fl2v,
 * minimax_pack, minimax_prompt_mentions; counts below are those call sites).
 *
 * Nothing enforces this list, and most call sites use ``editor.x?.()``, so a
 * refactor that turns any of these into a module-level function, a lookup table
 * or a static silently disables the caller. Treat the names below as the ABI:
 * they must exist on the *instance*.
 *
 * See also "Feature-owned state" at the bottom: fields the feature modules
 * attach to the editor at runtime. The editor itself reads three of them, so the
 * editor half-depends on minimax_image_batch having run.
 *
 * Adding a member here without adding it to the list is how the implicit ABI
 * grew in the first place; keep both sides in step.
 * ---------------------------------------------------------------------------
 * State shared with every feature module
 *   timeline (166)            the timeline model — mutate via commit(), never in place
 *   selectedIndex (46)        selected group index
 *   root (29)                 the editor's root element
 *   node (4)                  the ComfyUI node this editor is bound to
 *   domWidget (5) / widget (1) / container (2) / mainBody (1)
 *
 * Render + sync entry points (the only supported way to publish a change)
 *   commit (28)               settle a change, then render + sync
 *   scheduleRender (15)       request a render on the next frame
 *   scheduleTimelineSync (6) / flushTimelineSync (3) / _schedulePromptRender (1)
 *   renderImageBatchGroups (25) / updateDomWidgetHeight (15)
 *   updateVideoNameLabel (12) / updateOutputPreview (3) / getDirectorUiMinHeight (2)
 *
 * Widget handles (live ComfyUI widgets the feature modules read/write)
 *   taskTypeWidget (10) / totalFramesWidget (6) / frameRateWidget (3)
 *   globalPromptWidget (3) / widthWidget (1) / heightWidget (1) / equalCountInput (11)
 *
 * Data + task queries
 *   getTaskKey (12) / getFrameRate (1) / getWorkflowId (1)
 *   buildTimelinePayload (1) / applyImportedTimeline (1)
 *
 * Mode predicates (read-only; feature modules branch on these)
 *   hasExternalI2vGroups (7) / hasExternalR2vGroups (7) / isR2vCommonEnabled (6)
 *   isRunSelectEnabled (4) / usesBatchTimeline (2) / supportsRunSelect (2)
 *   isSegmentRunEnabled (2) / isFl2vMode (2) / isImageBatch (1) / isR2vBatch (1)
 *   fl2vUi (10) / _previewSegments (3)
 *
 * Run selection / segment actions
 *   toggleRunSelectMode (1) / setRunSelectionAll (1) / toggleSegmentRun (2)
 *   _runHighlightSeg (1) / onSegmentRemoved (1) / canAlignToNext (1)
 *   _syncR2vCardSelection (2) / writeExternalGroupPrompt (1)
 *
 * Media pickers (open a file dialog and feed the editor)
 *   chooseImageInput (3) / chooseImageInputs (1) / chooseVideoInput (1)
 *   chooseVideoInputs (1) / chooseAudioInput (1) / chooseAudioInputs (1)
 *   btnVideo (3) / btnVideoExisting (3) / btnVideoAppend (3)
 *
 * Dialogs
 *   showBdDialog (2) / showBdMessage (2)
 *
 * Feature-owned DOM handles the editor keeps as a meeting point
 *   batchPanel (4) / batchList (7) / batchPicker (5) / batchI2vNotice (5)
 *   batchHint (2) / batchAddBtn (1) / globalPrompt (4) / segPrompt (1)
 *   _thumbCache (2) / _thumbPending (4) / _fl2vUploadMode (4)
 *   _fl2vSlotKind (4) / _fl2vSlotShotIndex (4)
 *
 * Feature-owned state (attached at runtime — NOT declared in this class)
 *   minimax_image_batch: r2vPage, r2vScope, r2vFold, r2vAssetPage,
 *     r2vPreviewTab, _batchRefDragMoved,
 *     batchRunSelectBtn / batchRunSelectAllWrap / batchRunSelectAllCb
 *     — the last three are read back by this class (the dependency is mutual)
 *   minimax_fl2v: _fl2vDragFrom, _fl2vShotDrag, _fl2vShotDragFrom, _fl2vSlotDrag,
 *     _fl2vIgnoreSlotClickUntil, _fl2vPromptSegIndex
 * =========================================================================== */

/** Resolve a stable, workflow-bound unique id for cache namespacing.
 *
 * Unlike the old display-name approach (which returned "" for unnamed/unsaved
 * workflows and broke whenever the user renamed the file), this persists a UUID
 * inside the graph's ``extra`` metadata. The id is generated once per workflow
 * file, is stable across reloads and machines, and never depends on the user
 * naming the workflow. Empty string is only returned if even the graph object
 * is unavailable — the server then falls back to the node-id-only directory.
 */
// Module-level fallback so a workflow id is ALWAYS non-empty. An empty id would
// make the backend fall back to the node-id-only directory (e.g. `node_5/`) and
// silently split the cache into an orphan folder that status checks / exports
// never read — which is exactly what greyed-out「对齐下段」checkboxes and
// "export produced nothing" came from.
let _fallbackWorkflowId = null;


app.registerExtension({
    name: "ComfyUI.MiniMaxH3DirectorOptPlugin",
    async setup() {
        installDirectorClipboardGuard();
        try {
            sanitizeAllWidgetValues();
        } catch (e) {
            /* best-effort */
        }
        const flushDirectors = () => {
            const graph = app.graph ?? app.canvas?.graph;
            for (const node of graph?._nodes ?? graph?.nodes ?? []) {
                // Stale widget values (new widgets in an old workflow, combos
                // saved before their model existed) fail backend validation
                // before execute() runs, so repair them before every queue.
                try {
                    sanitizeWidgetValues(node);
                } catch (e) {
                    /* best-effort */
                }
                node._minimaxEditor?.flushTimelineSync?.();
                // Keep the workflow id in the hidden `workflow_name` widget in sync
                // on every queue. The backend cache layout is keyed on this id, so
                // a stale/empty widget (e.g. after loading a saved workflow or
                // clearing the cache) would otherwise write to the bare node_<id>/
                // directory and make every「对齐下段」/「分段导出」checkbox grey out.
                const wfName = getStableWorkflowId();
                const w = (node.widgets || []).find((x) => x?.name === "workflow_name");
                if (w && w.value !== wfName) w.value = wfName;
            }
        };
        if (app.queuePrompt && !app.queuePrompt._minimaxPatched) {
            const orig = app.queuePrompt.bind(app);
            app.queuePrompt = function (...args) {
                flushDirectors();
                clearAllDirectorRunStatus();
                return orig(...args);
            };
            app.queuePrompt._minimaxPatched = true;
        }

        api.addEventListener("minimax_director_opt_progress", ({ detail }) => {
            findDirectorNode(detail?.node_id)?._minimaxEditor?.setRunProgress?.(detail);
        });

        api.addEventListener("minimax_director_opt_preview", ({ detail }) => {
            const editor = findDirectorNode(detail?.node_id)?._minimaxEditor;
            if (!editor) return;
            if (editor.isImageBatch?.()) {
                setImageBatchPreview(
                    editor,
                    detail?.segment_index ?? 0,
                    detail?.image_b64 || "",
                    {
                        frames: detail?.live ? undefined : detail?.frames,
                        fps: detail?.fps,
                        live: !!detail?.live,
                        step: detail?.step,
                        total_steps: detail?.total_steps,
                    },
                );
                return;
            }
            editor.setLiveSamplePreview?.(detail);
        });

        api.addEventListener("executing", ({ detail }) => {
            if (detail == null) return;
            const node = findDirectorNode(detail);
            const editor = node?._minimaxEditor;
            if (!editor) return;
            editor.flushTimelineSync?.();
            editor.clearLiveSamplePreview?.();
            if (editor.isImageBatch?.()) {
                for (const seg of editor.timeline.segments || []) {
                    seg.previewB64 = "";
                    seg.previewFrames = [];
                    seg.previewLive = false;
                    seg.previewStep = null;
                    seg.previewTotalSteps = null;
                }
                editor.renderImageBatchGroups?.();
            }
            const segTotal = editor.getRunProgressSegmentTotal?.() ?? (editor.timeline?.segments?.length || 1);
            const timelineTotal = editor.timeline?.segments?.length || segTotal;
            editor.setRunProgress({
                node_id: detail,
                segment: 1,
                segment_total: segTotal,
                timeline_segment: 1,
                timeline_segment_total: timelineTotal,
                partial_run: editor.isRunSelectEnabled?.() && segTotal < timelineTotal,
                phase: "plan",
                phase_label: t("executing.parseTimeline"),
                phase_value: 0,
                phase_max: 1,
                overall_value: 0,
                overall_max: Math.max(1, segTotal * 6),
                remaining_segments: Math.max(0, segTotal - 1),
            });
        });

        api.addEventListener("execution_error", ({ detail }) => {
            const node = findDirectorNode(detail?.node_id);
            if (node?._minimaxEditor) {
                node._minimaxEditor.setRunError(detail?.exception_message || t("executing.error"));
            }
        });

        patchDirectorDomWidgetLayout();
        setTimeout(patchDirectorDomWidgetLayout, 500);
    },
    async loadedGraphNode(node) {
        if (!isMiniMaxH3DirectorOptNode(node)) return;
        normalizeDirectorOutputs(node);
        pruneDirectorDomWidgets(node);
        if (!node._minimaxDomWidget) return;
        finalizeDirectorWidgetOrder(node);
        ensureDirectorDomWidgetWidth(node);
        bindDirectorDomWidgetSizing(node, node._minimaxDomWidget, () => node._minimaxEditor);
        const editor = initDirectorEditor(node);
        editor?.scheduleRender?.();
        // Workflow size is already on node.size — settle fill after widgets arrange.
        scheduleDirectorLayoutSettle(editor);
    },
    async getCustomWidgets() {
        return {
            BDGROUP(node, inputName, inputData) {
                const w = makeGroupHeaderWidget(inputName, inputData);
                if (!node.widgets) node.widgets = [];
                node.widgets.push(w);
                return w;
            },
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeType?.comfyClass || nodeData?.name || "";
        if (EXTERNAL_GROUP_NODE_TYPES.has(cls) || cls === EXTERNAL_COMBINE_NODE_TYPE) {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const out = onConnectionsChange?.apply(this, args);
                // Combine/Group wiring changes do not fire Director.onConnectionsChange.
                queueMicrotask(() => notifyDirectorsSyncExternalGroups());
                return out;
            };
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const out = onCreated?.apply(this, arguments);
                // Keep Director timeline in sync when duration/prompt widgets change.
                queueMicrotask(() => {
                    for (const w of this.widgets || []) {
                        if (w?.name !== "duration_sec" && w?.name !== "prompt") continue;
                        if (w._mmxExternalSyncPatched) continue;
                        w._mmxExternalSyncPatched = true;
                        const prev = w.callback;
                        w.callback = function (...cbArgs) {
                            const r = prev?.apply(this, cbArgs);
                            // Director→Group write-through sets this to avoid echo wipe.
                            if (!w._mmxSkipExternalSync) notifyDirectorsSyncExternalGroups();
                            return r;
                        };
                    }
                });
                return out;
            };
            return;
        }

        if (!isDirectorNodeDef(nodeType, nodeData)) return;
        if (nodeType.prototype._minimaxDirectorPatched) return;
        nodeType.prototype._minimaxDirectorPatched = true;
        // Deleting an input shifts every later entry of the positional
        // `widgets_values` array in older workflows — drop the stale slot first.
        patchDirectorWidgetValueMigration(nodeType);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            normalizeDirectorOutputs(this);
            applyDirectorWidgetLabels(this);
            // ComfyUI may attach seed's control_after_generate combo after onNodeCreated.
            queueMicrotask(() => applyDirectorWidgetLabels(this));
            setTimeout(() => applyDirectorWidgetLabels(this), 0);
            this.size = [1000, 680];

            // Backend cache layout is keyed on a stable workflow id (a UUID persisted
            // in the graph's extra metadata): the conditioning and batch caches live
            // under <workflow_id>/node_<id>. Sync that id into the hidden
            // `workflow_name` widget so the server picks the right dir.
            const syncWorkflowName = () => {
                const wfName = getStableWorkflowId();
                const w = (this.widgets || []).find((x) => x?.name === "workflow_name");
                if (w && w.value !== wfName) w.value = wfName;
            };
            syncWorkflowName();
            setTimeout(syncWorkflowName, 0);
            setTimeout(syncWorkflowName, 200);

            // 「清空缓存」 / 「清空节点所有缓存」 buttons. Unlike the old checkbox they
            // fire immediately via the HTTP route instead of riding a run's edge — so
            // clearing works even when the timeline has nothing to run.
            //   · 清空缓存           → this node's conditioning + batch scratch files, plus
            //                          any legacy *_frames_ht.pt seam window (superseded by
            //                          *_frames_ht.mp4; never touches the durable segment files).
            //   · 清空节点所有缓存   → additionally wipes every durable seg_* file in the
            //                          unified minimax_director_opt_cache dir, forcing a full
            //                          re-render of every segment.
            const runClearCache = (clearAll) => {
                const nodeId = String(this.id ?? "");
                const wfName = getStableWorkflowId();
                const scope = t(clearAll ? "cache.scopeAll" : "cache.scopeTextBatch");
                const willDelete = clearAll
                    ? [
                        t("cache.itemConditioning"),
                        t("cache.itemBatchScratch"),
                        t("cache.itemSegments"),
                    ]
                    : [
                        t("cache.itemConditioning"),
                        t("cache.itemBatchScratch"),
                        t("cache.itemLegacyHeadtail"),
                    ];
                if (!window.confirm(
                    t("cache.confirmClear", { scope }) + "\n\n" +
                    (wfName ? t("cache.labelWorkflow") + wfName + "\n" : "") +
                    t("cache.labelNodeId") + nodeId + "\n\n" +
                    t("cache.labelWillDelete") + "\n" +
                    willDelete.join("\n") +
                    (clearAll ? "\n\n" + t("cache.warnRerender") : "\n\n" + t("cache.noteKeepSegments"))
                )) {
                    return;
                }
                (async () => {
                    try {
                        const resp = await api.fetchApi("/minimax/director_opt/clear_cache", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ node_id: nodeId, workflow_name: wfName, clear_all: clearAll }),
                        });
                        const data = resp.ok ? await resp.json() : { error: (await resp.text()).slice(0, 200) };
                        if (!resp.ok) {
                            console.error("[MiniMax H3Director] clear cache failed:", data);
                            window.alert(t("cache.clearFailed", { detail: data.error || resp.status }));
                            return;
                        }
                        const cond = data.cleared?.conditioning ?? 0;
                        const batch = data.cleared?.batch ?? 0;
                        const segments = data.cleared?.segments ?? 0;
                        const headtail = data.cleared?.headtail ?? 0;
                        const msg = [
                            t("cache.clearedTitle"),
                            t("cache.labelWorkflow") + (wfName || t("cache.unnamedWorkflow")),
                            t("cache.deletedConditioning", { count: cond }),
                            t("cache.deletedBatch", { status: batch ? t("cache.removed") : t("cache.none") }),
                            ...(headtail ? [t("cache.deletedHeadtail", { count: headtail })] : []),
                            ...(clearAll ? [t("cache.deletedSegments", { status: segments ? t("cache.removed") : t("cache.none") })] : []),
                        ].join("\n");
                        console.log(
                            `[MiniMax H3Director] cache cleared: ${cond} conditioning file(s), ` +
                            `${batch ? "batch scratch removed" : "no batch scratch"}, ` +
                            `${clearAll ? (segments ? "segment cache removed" : "no segment cache") : "segments kept"} ` +
                            `(workflow '${wfName || ""}')`
                        );
                        window.alert(msg);
                    } catch (err) {
                        console.error("[MiniMax H3Director] clear cache error:", err);
                        window.alert(t("cache.clearError", { detail: err }));
                    }
                })();
            };
            // A button widget's name *is* its visible text (nothing looks these two
            // up by name), so they are built from t() and re-translated on switch.
            const clearBtn = this.addWidget("button", t("cache.buttonClear"), null, () => runClearCache(false));
            const clearAllBtn = this.addWidget("button", t("cache.buttonClearAll"), null, () => runClearCache(true));
            this._unsubCacheLocale?.();
            this._unsubCacheLocale = onLocaleChange(() => {
                const relabel = (w, key) => {
                    if (!w) return;
                    w.name = t(key);
                    w.label = t(key);
                };
                relabel(clearBtn, "cache.buttonClear");
                relabel(clearAllBtn, "cache.buttonClearAll");
                this.setDirtyCanvas?.(true, true);
            });

            const existingDom = pruneDirectorDomWidgets(this);
            // Idempotent: reuse the host if onNodeCreated / graph restore already mounted one.
            if (existingDom?.element) {
                setTimeout(() => {
                    finalizeDirectorWidgetOrder(this);
                    initDirectorEditor(this);
                }, 0);
                return r;
            }

            const container = document.createElement("div");
            container.className = "mmx-host";
            container.style.minHeight = `${getDirectorUiHeight(null)}px`;
            container.style.setProperty("--comfy-widget-min-height", `${getDirectorUiHeight(null)}px`);
            const self = this;
            const widget = this.addDOMWidget(DIRECTOR_DOM_WIDGET_NAME, "director", container, {
                getValue: () => "",
                setValue: () => {},
                getMinHeight: () => getDirectorUiHeight(self._minimaxEditor),
                hideOnZoom: false,
                onDraw() {
                    if (self._minimaxEditor?.isPlaying) return;
                    ensureDirectorDomWidgetWidth(self);
                },
                afterResize: () => {
                    if (self._minimaxEditor?.isPlaying || self._minimaxEditor?._pauseSettling) return;
                    ensureDirectorDomWidgetWidth(self);
                    self._minimaxEditor?.onNodeResize?.();
                },
            });
            bindDirectorDomWidgetSizing(self, widget, () => self._minimaxEditor);
            widget.element = container;
            ensureDirectorDomWidgetWidth(self);
            self._minimaxDomWidget = widget;
            finalizeDirectorWidgetOrder(self);

            setTimeout(() => {
                finalizeDirectorWidgetOrder(self);
                initDirectorEditor(self);
            }, 0);
            return r;
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            ensureDirectorDomWidgetWidth(this);
            const out = onResize?.apply(this, arguments);
            if (!this._minimaxEditor?.isPlaying && !this._minimaxEditor?._pauseSettling) {
                this._minimaxEditor?.onNodeResize?.(size);
            }
            return out;
        };

        const onSelected = nodeType.prototype.onSelected;
        nodeType.prototype.onSelected = function () {
            ensureDirectorDomWidgetWidth(this);
            const out = onSelected?.apply(this, arguments);
            // Reselect often lands after graph zoom/layout changes — settle redraw
            // fixes thumbs that were stretched from a mismatched canvas CSS box.
            this._minimaxEditor?.scheduleSettleRender?.();
            this._minimaxEditor?.syncExternalGroupsTimeline?.();
            return out;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const out = onConnectionsChange?.apply(this, args);
            this._minimaxEditor?.syncExternalGroupsTimeline?.();
            return out;
        };

        const onDeselected = nodeType.prototype.onDeselected;
        nodeType.prototype.onDeselected = function () {
            const out = onDeselected?.apply(this, arguments);
            if (this._minimaxEditor?.isPlaying) this._minimaxEditor._stopPlay();
            return out;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            destroyDirectorEditor(this);
            return onRemoved?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            normalizeDirectorOutputs(this);
            const out = onConfigure?.apply(this, arguments);
            setTimeout(() => {
                finalizeDirectorWidgetOrder(this);
                const ed = initDirectorEditor(this) || this._minimaxEditor;
                if (!ed) return;
                const initTotal = Math.max(0, parseInt(ed.totalFramesWidget?.value || 124, 10));
                const initFps = coerceTimelineFps(ed.frameRateWidget?.value || 24);
                ed.timeline = parseTimeline(ed.timelineWidget?.value, initTotal, initFps);
                ed.syncFrameRateUI(ed.timeline.frameRate);
                ed._directorMode = ed.getDirectorMode();
                if (ed._directorMode === "video") {
                    ed.restoreVideoFromTimeline();
                } else if (ed._directorMode === "prompt_batch" || ed._directorMode === "image_batch") {
                    ensureImageBatchTimeline(ed);
                } else {
                    ed.ensureGenTimeline();
                }
                ed.applyTaskLayout(ed._directorMode);
                ed.populateTaskSelect(ed.globalTask, ed.taskTypeWidget?.value);
                ed.setEditMode(ed.timeline.editMode || "global");
                ed.selectedIndex = 0;
                ed.updateSelectionUI();
                ed.commit(true, { syncTimeline: false });
                ed._externalGroupsSyncSig = null;
                ed.syncExternalGroupsTimeline?.();
                ed.scheduleSettleRender?.();
            }, 80);
            return out;
        };
    },
});
