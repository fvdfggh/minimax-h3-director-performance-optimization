/** Reading ComfyUI's graph and widgets around a Director node.
 *
 * The editor's panels live on a LiteGraph node, so much of what the UI needs is
 * not in the timeline but in the graph: which node is selected, which widget holds
 * a path, which upstream loader a slot is linked to, and whether that link is
 * itself a passthrough from an external group node. Everything here answers those
 * questions — no DOM and no editor instance: each helper takes a graph / node /
 * widget and returns data.
 *
 * Two behaviours are load-bearing and easy to break:
 *
 * * a widget value may be a string, an array or a wrapped ``{content: …}`` object
 *   depending on the ComfyUI version, so reading always normalises first;
 * * Director's own renders live in ``output/``, not ``input/``, so a ``/api/view``
 *   URL must echo the record's own ``type`` — see ``core/utils.js`` ``viewUrl``.
 *
 * Extracted verbatim from minimax_timeline.js — no behaviour change.
 */

import { app } from "../../scripts/app.js";

export function getStableWorkflowId() {
    try {
        const graph = app.graph ?? app.canvas?.graph;
        if (!graph) {
            if (!_fallbackWorkflowId) {
                _fallbackWorkflowId = ("wf-" + Date.now() + "-" + Math.random().toString(36).slice(2));
            }
            return _fallbackWorkflowId;
        }
        if (!graph.extra) graph.extra = {};
        let id = graph.extra.minimax_director_opt_workflow_id;
        if (!id) {
            id = (crypto?.randomUUID?.() || ("wf-" + Date.now() + "-" + Math.random().toString(36).slice(2)));
            graph.extra.minimax_director_opt_workflow_id = id;
            try { app.graph?.setDirty?.(); } catch (_) { /* ignore */ }
        }
        return String(id);
    } catch (_) { /* ignore */ }
    if (!_fallbackWorkflowId) {
        _fallbackWorkflowId = ("wf-" + Date.now() + "-" + Math.random().toString(36).slice(2));
    }
    return _fallbackWorkflowId;
}

export function findDirectorNode(nodeId) {
    const id = String(nodeId);
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        if (String(node.id) === id) return node;
    }
    return null;
}

export const EXTERNAL_GROUP_NODE_TYPES = new Set([
    "MiniMaxH3DirectorOptGroupImageToVideo",
    "MiniMaxH3DirectorOptGroupReferenceToVideo",
]);
export const EXTERNAL_COMBINE_NODE_TYPE = "MiniMaxH3DirectorOptGroupsCombine";

export function graphLinkRecord(graph, linkId) {
    if (linkId == null || !graph) return null;
    const links = graph.links;
    if (!links) return null;
    let link = links[linkId];
    if (!link && typeof links.find === "function") {
        link = links.find((l) => l && (l.id === linkId || l[0] === linkId));
    }
    if (!link) return null;
    return {
        originId: link.origin_id ?? link[1],
        originSlot: link.origin_slot ?? link[2],
    };
}

export function nodeWidgetValue(node, name) {
    const w = (node?.widgets || []).find((x) => x?.name === name);
    return w?.value;
}

/** Normalize LoadImage-style widget value → relative input path. */
export function normalizeImageWidgetPath(value) {
    if (value == null || value === "") return null;
    let val = value;
    if (Array.isArray(val)) val = val[0];
    if (typeof val === "object" && val) {
        const name = String(val.filename || val.name || "").trim();
        if (!name) return null;
        const sub = String(val.subfolder || "").replace(/\\/g, "/").replace(/\/$/, "");
        return sub ? `${sub}/${name}` : name;
    }
    if (typeof val === "string") {
        const s = val.replace(/\s*\[(input|output|temp)\]\s*$/i, "").trim();
        return s || null;
    }
    return null;
}

export function readImageWidgetPath(node) {
    for (const name of ["image", "image_path", "filename"]) {
        const path = normalizeImageWidgetPath(nodeWidgetValue(node, name));
        if (path) return path;
    }
    return null;
}

/** VHS Load Video / Load Video Path use `video`; some loaders use video_path. */
export function readVideoWidgetPath(node) {
    for (const name of ["video", "video_path", "file"]) {
        const path = normalizeImageWidgetPath(nodeWidgetValue(node, name));
        if (path) return path;
    }
    return null;
}

/** LoadAudio uses `audio`; VHS audio output often has no file — fall back to sibling video widget. */
export function readAudioWidgetPath(node) {
    for (const name of ["audio", "audio_path", "file"]) {
        const path = normalizeImageWidgetPath(nodeWidgetValue(node, name));
        if (path) return path;
    }
    return readVideoWidgetPath(node);
}

export function mediaBaseName(path) {
    const s = String(path || "").replace(/\\/g, "/");
    return s.split("/").pop() || s;
}

export function parseViewUrlToPath(url) {
    try {
        const u = new URL(String(url || ""), window.location.origin);
        const filename = u.searchParams.get("filename");
        if (!filename) return null;
        const subfolder = (u.searchParams.get("subfolder") || "").replace(/\\/g, "/").replace(/\/$/, "");
        return subfolder ? `${subfolder}/${filename}` : filename;
    } catch {
        return null;
    }
}

export function linkedSourceNode(graph, node, inputName) {
    if (!graph || !node) return null;
    const inp = (node.inputs || []).find((i) => i?.name === inputName);
    if (inp?.link == null) return null;
    const rec = graphLinkRecord(graph, inp.link);
    if (!rec) return null;
    return graph.getNodeById?.(rec.originId) || null;
}

/** Resolve an IMAGE input on `node` to a Comfy input-folder relative path. */
export function resolveLinkedImageFile(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;

    const direct = readImageWidgetPath(src);
    if (direct) return direct;

    const imgEl = (src.imageIndex != null ? src.imgs?.[src.imageIndex] : null) || src.imgs?.[0];
    const fromPreview = parseViewUrlToPath(imgEl?.src);
    if (fromPreview) return fromPreview;

    // Walk through IMAGE passthrough nodes (Resize, etc.).
    const imgInputs = (src.inputs || []).filter((i) => String(i?.type || "") === "IMAGE" && i.link != null);
    for (const next of imgInputs) {
        const path = resolveLinkedImageFile(graph, src, next.name, depth + 1);
        if (path) return path;
    }
    return null;
}

/**
 * Resolve ref_video IMAGE-batch wiring to a previewable file path.
 * VHS Load Video exposes frames on IMAGE but the path lives on the `video` widget.
 */
export function resolveLinkedVideoFile(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;

    const direct = readVideoWidgetPath(src);
    if (direct) return direct;

    // Some loaders still put the clip name on filename / image widgets.
    const fallback = readImageWidgetPath(src);
    if (fallback && /\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(fallback)) return fallback;

    const walkTypes = new Set(["IMAGE", "VIDEO"]);
    const nextInputs = (src.inputs || []).filter(
        (i) => walkTypes.has(String(i?.type || "").toUpperCase()) && i.link != null,
    );
    for (const next of nextInputs) {
        const path = resolveLinkedVideoFile(graph, src, next.name, depth + 1);
        if (path) return path;
    }
    return null;
}

/** Resolve AUDIO wiring (LoadAudio / VHS audio out) to a previewable path. */
export function resolveLinkedAudioFile(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;

    const direct = readAudioWidgetPath(src);
    if (direct) return direct;

    const nextInputs = (src.inputs || []).filter(
        (i) => String(i?.type || "").toUpperCase() === "AUDIO" && i.link != null,
    );
    for (const next of nextInputs) {
        const path = resolveLinkedAudioFile(graph, src, next.name, depth + 1);
        if (path) return path;
    }
    return null;
}

export function imageRefFromPath(path) {
    if (!path) return null;
    return { imageFile: path, width: 0, height: 0 };
}

export function videoRefFromPath(path, index) {
    if (!path) return null;
    return {
        index,
        videoFile: path,
        fileName: mediaBaseName(path),
        type: "input",
        subfolder: "",
        pairedAudioFile: "",
        previewImageFile: "",
        previewImageUrl: "",
        linked: true,
    };
}

/** First-frame poster from Load Video / IMAGE-batch upstream (when file path missing). */
export function resolveLinkedVideoPoster(graph, node, inputName, depth = 0) {
    if (!graph || !node || depth > 8) return null;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src) return null;
    const imgEl = (src.imageIndex != null ? src.imgs?.[src.imageIndex] : null) || src.imgs?.[0];
    if (imgEl?.src) {
        return {
            previewImageUrl: imgEl.src,
            previewImageFile: parseViewUrlToPath(imgEl.src) || "",
        };
    }
    const nextInputs = (src.inputs || []).filter(
        (i) => String(i?.type || "").toUpperCase() === "IMAGE" && i.link != null,
    );
    for (const next of nextInputs) {
        const poster = resolveLinkedVideoPoster(graph, src, next.name, depth + 1);
        if (poster) return poster;
    }
    return null;
}

export function collectAutogrowVideoRefs(graph, node) {
    const found = new Map();
    const re = /(?:^|\.)ref_video_(\d+)$/;
    for (const inp of node.inputs || []) {
        const m = String(inp?.name || "").match(re);
        if (!m || inp.link == null) continue;
        const idx = parseInt(m[1], 10);
        if (!Number.isFinite(idx) || found.has(idx)) continue;
        patchUpstreamVideoWidgetSync(graph, node, inp.name);
        const path = resolveLinkedVideoFile(graph, node, inp.name);
        const poster = resolveLinkedVideoPoster(graph, node, inp.name);
        if (path) {
            const ref = videoRefFromPath(path, idx);
            if (poster) {
                ref.previewImageFile = poster.previewImageFile || "";
                ref.previewImageUrl = poster.previewImageUrl || "";
            }
            found.set(idx, ref);
        } else if (poster?.previewImageUrl || poster?.previewImageFile) {
            found.set(idx, {
                index: idx,
                videoFile: "",
                fileName: poster.previewImageFile
                    ? mediaBaseName(poster.previewImageFile)
                    : `video_${idx + 1}`,
                type: "input",
                subfolder: "",
                pairedAudioFile: "",
                previewImageFile: poster.previewImageFile || "",
                previewImageUrl: poster.previewImageUrl || "",
                linked: true,
            });
        } else {
            // Linked IMAGE batch without resolvable path/poster — still mark occupied.
            found.set(idx, {
                index: idx,
                videoFile: "",
                fileName: "",
                type: "input",
                subfolder: "",
                pairedAudioFile: "",
                previewImageFile: "",
                previewImageUrl: "",
                linked: true,
            });
        }
    }
    return [...found.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

export function audioRefFromPath(path, index) {
    if (!path) return null;
    return {
        index,
        audioFile: path,
        fileName: mediaBaseName(path),
        type: "input",
        subfolder: "",
    };
}

export function patchUpstreamWidgetSync(graph, node, inputName, widgetNames) {
    if (!graph || !node || !widgetNames?.length) return;
    const src = linkedSourceNode(graph, node, inputName);
    if (!src?.widgets) return;
    const names = new Set(widgetNames);
    for (const w of src.widgets) {
        if (!w || !names.has(w.name)) continue;
        if (w._mmxExternalMediaSyncPatched) continue;
        w._mmxExternalMediaSyncPatched = true;
        const prev = w.callback;
        w.callback = function (...cbArgs) {
            const r = prev?.apply(this, cbArgs);
            queueMicrotask(() => notifyDirectorsSyncExternalGroups());
            return r;
        };
    }
}

export function patchUpstreamImageWidgetSync(graph, node, inputName) {
    patchUpstreamWidgetSync(graph, node, inputName, ["image", "image_path", "filename"]);
}

export function patchUpstreamVideoWidgetSync(graph, node, inputName) {
    patchUpstreamWidgetSync(graph, node, inputName, ["video", "video_path", "file", "filename"]);
}

export function patchUpstreamAudioWidgetSync(graph, node, inputName) {
    patchUpstreamWidgetSync(graph, node, inputName, ["audio", "audio_path", "video", "video_path", "file", "filename"]);
}

/** Collect Autogrow / legacy slots matching `(?:^|\.)prefix_(\\d+)$`. */
export function collectAutogrowSlotRefs(graph, node, prefix, resolvePath, toRef, patchSync) {
    const found = new Map();
    const re = new RegExp(`(?:^|\\.)${prefix}_(\\d+)$`);
    for (const inp of node.inputs || []) {
        const m = String(inp?.name || "").match(re);
        if (!m || inp.link == null) continue;
        const idx = parseInt(m[1], 10);
        if (!Number.isFinite(idx) || found.has(idx)) continue;
        patchSync?.(graph, node, inp.name);
        const path = resolvePath(graph, node, inp.name);
        const ref = path ? toRef(path, idx) : null;
        if (ref) found.set(idx, ref);
    }
    return [...found.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
}

export function readExternalGroupSpec(node, graph = null) {
    const g = graph || app.graph || app.canvas?.graph;
    const durRaw = Number(nodeWidgetValue(node, "duration_sec"));
    const prompt = String(nodeWidgetValue(node, "prompt") ?? "");
    const cls = node?.comfyClass || node?.type || "";
    const firstImageFile = resolveLinkedImageFile(g, node, "first_frame");
    const lastImageFile = resolveLinkedImageFile(g, node, "last_frame");
    patchUpstreamImageWidgetSync(g, node, "first_frame");
    patchUpstreamImageWidgetSync(g, node, "last_frame");

    let refImages = [];
    let refVideos = [];
    let refAudios = [];
    if (cls === "MiniMaxH3DirectorOptGroupReferenceToVideo") {
        // Autogrow: ref_images.ref_image_0 / ref_videos.ref_video_0 / …
        refImages = collectAutogrowSlotRefs(
            g, node, "ref_image", resolveLinkedImageFile,
            (path, idx) => ({ index: idx, imageFile: path, imageB64: "" }),
            patchUpstreamImageWidgetSync,
        );
        refVideos = collectAutogrowVideoRefs(g, node);
        const standaloneAudios = collectAutogrowSlotRefs(
            g, node, "ref_audio", resolveLinkedAudioFile,
            audioRefFromPath,
            patchUpstreamAudioWidgetSync,
        );
        // Paired soundtrack for the same-index reference video (ref_video_audio_N).
        const pairedAudios = collectAutogrowSlotRefs(
            g, node, "ref_video_audio", resolveLinkedAudioFile,
            audioRefFromPath,
            patchUpstreamAudioWidgetSync,
        );
        const pairedByIndex = new Map(pairedAudios.map((a) => [a.index, a]));
        for (const vid of refVideos) {
            const paired = pairedByIndex.get(vid.index);
            if (paired?.audioFile) {
                vid.pairedAudioFile = paired.audioFile;
                pairedByIndex.delete(vid.index);
            }
        }
        // Show unpaired video-audio (or standalone) in the 参考音频 strip.
        const audioMap = new Map(standaloneAudios.map((a) => [a.index, a]));
        for (const [idx, paired] of pairedByIndex) {
            if (!audioMap.has(idx)) audioMap.set(idx, paired);
        }
        refAudios = [...audioMap.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
    }

    return {
        nodeId: node?.id ?? null,
        durationSec: Number.isFinite(durRaw) && durRaw > 0 ? durRaw : null,
        prompt,
        firstImageFile,
        lastImageFile,
        refImages,
        refVideos,
        refAudios,
    };
}

/** Autogrow slots are named `groups.group_0`; legacy used `group_0` / `group_01`. */
export function combineGroupSlotIndex(name) {
    const m = String(name || "").match(/(?:^|\.)group_(\d+)$/);
    return m ? parseInt(m[1], 10) : null;
}

export function isCombineGroupSlot(input) {
    if (!input) return false;
    if (combineGroupSlotIndex(input.name) != null) return true;
    // Fallback: any MMX_DIR_GROUP input on the combine node.
    return String(input.type || "") === "MMX_DIR_GROUP";
}

/**
 * Frontend-only / wiring passthrough nodes that must be skipped when resolving
 * external groups for the Director UI.
 *
 * Official ComfyUI "reroute dots" are often just link path points (origin_id still
 * points at Combine) — so they already work. rgthree inserts a real virtual node
 * `Reroute (rgthree)` into the link chain, which used to stop expansion here.
 */
export function isExternalGroupPassthroughNode(node) {
    if (!node) return false;
    const cls = String(node.comfyClass || node.type || "");
    if (/reroute/i.test(cls)) return true;
    // Generic virtual single-input passthrough (frontend-only nodes).
    if (node.isVirtualNode) {
        const linked = (node.inputs || []).filter((i) => i?.link != null);
        if (linked.length === 1) return true;
    }
    return false;
}

export function passthroughUpstreamLinkId(node) {
    const linked = (node?.inputs || []).filter((i) => i?.link != null);
    return linked.length ? linked[0].link : null;
}

export function expandExternalGroupLink(graph, linkId, depth = 0, mode = "specs") {
    if (linkId == null || depth > 16) return [];
    const rec = graphLinkRecord(graph, linkId);
    if (!rec) return [];
    const node = graph.getNodeById?.(rec.originId);
    if (!node) return [];
    const cls = node.comfyClass || node.type || "";

    // Walk through Reroute / rgthree Reroute / other virtual passthroughs.
    if (isExternalGroupPassthroughNode(node)) {
        const upstream = passthroughUpstreamLinkId(node);
        if (upstream == null) return [];
        return expandExternalGroupLink(graph, upstream, depth + 1, mode);
    }

    if (cls === EXTERNAL_COMBINE_NODE_TYPE) {
        const out = [];
        const slots = (node.inputs || [])
            .filter(isCombineGroupSlot)
            .sort((a, b) => {
                const ai = combineGroupSlotIndex(a.name);
                const bi = combineGroupSlotIndex(b.name);
                if (ai != null && bi != null) return ai - bi;
                return 0;
            });
        for (const input of slots) {
            if (input.link == null) continue;
            out.push(...expandExternalGroupLink(graph, input.link, depth + 1, mode));
        }
        return out;
    }
    if (EXTERNAL_GROUP_NODE_TYPES.has(cls)) {
        return mode === "nodes" ? [node] : [readExternalGroupSpec(node, graph)];
    }
    // Unknown upstream packer — still reserve one slot for run-select/timeline.
    if (mode === "nodes") return [null];
    return [{
        nodeId: null,
        durationSec: null,
        prompt: "",
        firstImageFile: null,
        lastImageFile: null,
        refImages: [],
        refVideos: [],
        refAudios: [],
    }];
}

export function collectExternalGroupSpecs(editor) {
    const port = editor?.hasExternalI2vGroups?.()
        ? "i2v_groups"
        : editor?.hasExternalR2vGroups?.()
            ? "r2v_groups"
            : null;
    if (!port) return null;
    const graph = app.graph ?? app.canvas?.graph;
    const inp = editor?.node?.inputs?.find((i) => i?.name === port);
    if (!graph || inp?.link == null) return null;
    const specs = expandExternalGroupLink(graph, inp.link, 0, "specs");
    return specs.length ? specs : null;
}

export function collectExternalGroupNodes(editor) {
    const port = editor?.hasExternalI2vGroups?.()
        ? "i2v_groups"
        : editor?.hasExternalR2vGroups?.()
            ? "r2v_groups"
            : null;
    if (!port) return null;
    const graph = app.graph ?? app.canvas?.graph;
    const inp = editor?.node?.inputs?.find((i) => i?.name === port);
    if (!graph || inp?.link == null) return null;
    const nodes = expandExternalGroupLink(graph, inp.link, 0, "nodes");
    return nodes.length ? nodes : null;
}

export function notifyDirectorsSyncExternalGroups() {
    const graph = app.graph ?? app.canvas?.graph;
    for (const node of graph?._nodes ?? graph?.nodes ?? []) {
        if (!isMiniMaxH3DirectorOptNode(node)) continue;
        node._minimaxEditor?.syncExternalGroupsTimeline?.();
    }
}
