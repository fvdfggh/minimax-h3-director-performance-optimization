"""Segment-cache and segment-export routes.

The front end's cache panel and「分段导出」buttons talk to these: which segments
have a render (``/segment_export_status``, ``/second_sample_status``), whether the
next segment can be continued from this one (``/align_to_next_status``), dropping a
segment's files (``/remove_segment_slot``), clearing the cache (``/clear_cache``),
exporting (``/segment_export``) and streaming one clip (``/segment_clip``).

The「提取音频」group (``/audio_extract*``) drives the persistent audio store:
probing what can be extracted, running the extraction, listing one card's clips
and streaming one of them back so the「音频」tab can play it.

Relative imports *inside* the handlers resolve against :mod:`director` exactly as
they did in :mod:`http_routes`, because this module sits in the same package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from aiohttp import web

from ..lib.constants import ROUTE_PREFIX
from .routes_common import _plan_from_request

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.routes.segments")


def register(routes, register_route) -> None:
    """Attach this group's routes; called by :func:`http_routes.register_routes`."""
    register_route(routes, "POST", f"{ROUTE_PREFIX}/clear_cache", minimax_clear_cache)
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/segment_export_status",
        minimax_segment_export_status,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/second_sample_status",
        minimax_second_sample_status,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/align_to_next_status",
        minimax_align_to_next_status,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/remove_segment_slot",
        minimax_remove_segment_slot,
    )
    # HEAD 无需注册：RouteTableDef.get() 走 UrlDispatcher.add_get()，默认
    # allow_head=True 会自动挂上 HEAD；再显式注册一次会直接 RuntimeError
    # （"Added route will never be executed, method HEAD is already registered"）。
    register_route(routes, "GET", f"{ROUTE_PREFIX}/segment_clip", minimax_segment_clip)
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/segment_export",
        minimax_segment_export,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/audio_extract_status",
        minimax_audio_extract_status,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/audio_extract",
        minimax_audio_extract,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/audio_extract_list",
        minimax_audio_extract_list,
    )
    register_route(
        routes,
        "POST",
        f"{ROUTE_PREFIX}/audio_extract_remove",
        minimax_audio_extract_remove,
    )
    register_route(
        routes,
        "GET",
        f"{ROUTE_PREFIX}/audio_extract_file",
        minimax_audio_extract_file,
    )


async def minimax_clear_cache(request):
    """Clear the per-node cache for the given Director node.

    The Director now stores every cache kind (text/image/video conditioning,
    batch scratch intermediates, and durable segment frames / latents / audio /
    clips) in ONE flat directory: ``minimax_director_opt_cache/<workflow_slug>/node_<id>/``.

    Default clears only the transient data in that dir: text/image/video
    conditioning files and the per-run batch scratch intermediates
    (``seg_*_scratch_*.pt``), plus any ``*_frames_ht.pt`` left by runs from
    before the seam window became a clip — those are superseded dead weight,
    not part of the render. The durable rendered segments are kept.

    With ``clear_all=true`` it also wipes every durable ``seg_*`` file — every
    rendered frame / AV latent / audio / clip — **and** the「提取音频」store
    (``audio_extract/``), forcing a full re-render as well as a re-extract on the
    next run. The store sits in its own sub-directory under ``audio_*`` names, so
    the segment sweep cannot reach it: it has to be asked for by name.

    ``workflow_name`` is resolved to the same slug the cache layer uses, so the
    button always hits exactly the directory that holds this workflow's data.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    from .conditioning_cache import clear_conditioning_cache
    from . import cache_layout, segment_slots

    workflow_name = str(body.get("workflow_name") or "").strip() or None
    clear_all = bool(body.get("clear_all"))
    cleared = {
        "conditioning": 0, "batch": 0, "headtail": 0, "segments": 0, "audio_extract": 0,
    }

    cache_dir = cache_layout.node_cache_dir(node_id, workflow_name, create=False)

    # 1) text/image/video conditioning files (always cleared)
    try:
        cleared["conditioning"] = await asyncio.to_thread(
            clear_conditioning_cache,
            node_id=node_id,
            workflow_name=workflow_name,
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt clear conditioning cache failed: %s", exc)

    # 2) per-run batch scratch intermediates (always cleared via _scratch_ marker)
    if cache_dir.is_dir():
        for path in cache_layout.iter_scratch_files(cache_dir):
            try:
                path.unlink()
                cleared["batch"] += 1
            except OSError as exc:
                log.warning("MiniMax H3 Director Opt clear scratch %s failed: %s", path, exc)

    # 3) Legacy head/tail tensors (``*_frames_ht.pt``), on *every* clear.
    #    The seam window is a clip now; these are the pre-mp4 copies, tens of MB
    #    each, that no longer get rewritten because nothing re-renders their
    #    segment. Unlike the durable artefacts below they are safe to drop
    #    without forcing a re-render — see cache_layout.iter_legacy_headtail_files.
    #    Skipped under clear_all, which removes them along with everything else.
    if cache_dir.is_dir() and not clear_all:
        for path in cache_layout.iter_legacy_headtail_files(cache_dir):
            try:
                path.unlink()
                cleared["headtail"] += 1
            except OSError as exc:
                log.warning(
                    "MiniMax H3 Director Opt clear legacy head/tail %s failed: %s", path, exc
                )

    # 4) clear_all → also wipe durable segment files (both passes) so the next
    #    run must re-render. ``seg2_*`` is the「二级采样」cache family.
    if clear_all and cache_dir.is_dir():
        for glob in cache_layout.SEGMENT_GLOBS:
            for path in list(cache_dir.glob(glob)):
                if not path.is_file():
                    continue
                try:
                    path.unlink()
                    cleared["segments"] += 1
                except OSError as exc:
                    log.warning("MiniMax H3 Director Opt clear segment %s failed: %s", path, exc)
        # The slot maps name those files; drop them too so the next run rebuilds
        # the position → files mapping from scratch.
        segment_slots.clear_slots(cache_dir)
        segment_slots.clear_slots(cache_dir, variant=segment_slots.VARIANT_SECOND)

    # 5) clear_all → the「提取音频」store as well. Only here: ``audio_extract/`` holds
    #    user-made takes that survive every other clear on purpose, and the plain
    #    button's confirmation says so.
    if clear_all:
        try:
            from .audio_extract import clear_audio_store

            cleared["audio_extract"] = await asyncio.to_thread(
                clear_audio_store, node_id, workflow_name
            )
        except Exception as exc:
            log.warning("MiniMax H3 Director Opt clear audio extract failed: %s", exc)

    log.info(
        "MiniMax H3 Director Opt cleared caches for node %s (workflow '%s', clear_all=%s): %s",
        node_id, workflow_name or "", clear_all, cleared,
    )
    return web.json_response({"cleared": cleared})


async def minimax_segment_export_status(request):
    """Availability of every segment for「分段导出」(what the picker greys out)."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .plan_types import normalize_segment_export_source
        from .segment_cache import inspect_segment_export_status, sync_segment_slots
        from .segment_slots import VARIANT_SECOND

        workflow_name = str(body.get("workflow_name") or "").strip() or None
        # Which pass the picker is showing: the availability probe must answer
        # for that pass only, so switching to「二采」greys out segments that have
        # no ``seg2_*`` cache instead of reporting the first-pass render.
        source = normalize_segment_export_source(body.get("source") or body.get("cacheSource"))
        variant = VARIANT_SECOND if source == "2nd" else "1st"

        plan = _plan_from_request(body, str(timeline_data))
        # Reconcile first: the picker must not offer a render that belongs to a
        # group deleted from the middle of the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, variant=variant)
        return web.json_response(
            inspect_segment_export_status(node_id, plan, workflow_name=workflow_name, variant=variant)
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment-export status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_second_sample_status(request):
    """Availability of every segment for「二次采样」(what the picker greys out).

    A segment is second-sampleable only when its **first-pass latent** is cached
    AND the node holds a **text encoding cache** — mirroring how「分段导出」
    probes the same slot map for clip / latent, but for the second-pass source.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .segment_cache import (
            inspect_second_sample_status,
            sync_second_segment_slots,
            sync_segment_slots,
        )

        workflow_name = str(body.get("workflow_name") or "").strip() or None

        plan = _plan_from_request(body, str(timeline_data))
        # Reconcile both passes first so the picker never offers a segment whose
        # position belongs to a group deleted from the middle of the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name)
        sync_second_segment_slots(node_id, plan, workflow_name=workflow_name)
        return web.json_response(
            inspect_second_sample_status(node_id, plan, workflow_name=workflow_name)
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt second-sample status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_align_to_next_status(request):
    """Which segments may enable「对齐下段」(i.e. the next one holds an AV latent).

    The frontend cannot derive this itself: under「选择运行」``plan.index`` is
    the compact run order, so "the next segment" is not simply ``i + 1`` in the
    card list. The plan is rebuilt here (same payload as the run) so the
    backend's index semantics are the single source of truth.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .segment_cache import has_next_segment_av_latent, sync_segment_slots
        from .segment_continuity import is_continuity_active

        workflow_name = str(body.get("workflow_name") or "").strip() or None

        plan = _plan_from_request(body, str(timeline_data))
        segments = list(getattr(plan, "segments", None) or [])
        if not segments:
            # Selection-run is ON but nothing is ticked. Align-to-next availability
            # only depends on the next segment's cached AV latent, not on the
            # current run selection, so report status for *all* segments instead of
            # erroring out (which would grey every「对齐下段」control).
            _tl = json.loads(timeline_data) if timeline_data else {}
            for _k in ("runSelection", "run_selection", "runSelectEnabled", "run_select_enabled"):
                _tl.pop(_k, None)
            plan = _plan_from_request(body, json.dumps(_tl))
            segments = list(getattr(plan, "segments", None) or [])
        # The next segment's latent must be the neighbour's own render, so the
        # slot map is reconciled before anything is probed.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name)
        rows = []
        for seg in segments:
            rows.append(
                {
                    "index": int(seg.index),
                    # Master「段间引导」must be on for the pin to mean anything.
                    "continuity": bool(is_continuity_active(plan, seg)),
                    "canAlignToNext": bool(has_next_segment_av_latent(node_id, seg.index, workflow_name=workflow_name)),
                }
            )
        return web.json_response({"node_id": node_id, "segments": rows})
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt align-to-next status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_remove_segment_slot(request):
    """Drop the cached artefacts of one timeline position (UI delete).

    The UI calls this the moment a group is removed so the deleted group's
    render cannot be inherited by the group that slides into its place.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")
    try:
        index = int(body.get("index"))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid segment index.")

    workflow_name = str(body.get("workflow_name") or "").strip() or None
    try:
        from .audio_extract import sync_audio_slots
        from .segment_cache import remove_segment_slot

        removed = await asyncio.to_thread(
            remove_segment_slot,
            node_id,
            index,
            workflow_name=workflow_name,
        )
        # 删除而删除: the card is gone, so its extracted audio goes with it.
        # ``seg_ids`` is the remaining order — without it (an older front end)
        # this is a no-op and the next /audio_extract_list reconciles instead.
        seg_ids = body.get("seg_ids")
        if isinstance(seg_ids, list) and seg_ids:
            await asyncio.to_thread(
                sync_audio_slots,
                node_id,
                [str(x or "").strip() for x in seg_ids],
                workflow_name,
                _retain_ids(body),
            )
        return web.json_response({"removed": bool(removed)})
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment cache drop failed: %s", exc)
        return web.json_response({"removed": False, "error": str(exc)}, status=400)


async def minimax_segment_export(request):
    """Run a「分段导出」request against the cached segments.

    Body mirrors the timeline block: ``enabled``, ``mode``
    (``piecewise`` | ``continuous``), ``indices``. Each checked segment must have
    an exportable source — its clip cache, its frame cache, or (only when the
    caller supplies models) its latent — otherwise it is skipped and reported.
    """
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    mode = str(body.get("mode") or "").lower()
    raw_indices = body.get("indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        return web.Response(status=400, text="No segment indices selected.")

    try:
        from .plan_types import normalize_segment_export_mode, normalize_segment_export_source
        from .segment_cache import run_segment_export, sync_segment_slots
        from .segment_slots import VARIANT_SECOND

        mode = normalize_segment_export_mode(mode)
        source = normalize_segment_export_source(body.get("source") or body.get("cacheSource"))
        variant = VARIANT_SECOND if source == "2nd" else "1st"
        try:
            indices = [int(i) for i in raw_indices]
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid segment indices.")

        plan = _plan_from_request(body, str(timeline_data))
        workflow_name = str(body.get("workflow_name") or "").strip() or None
        # Reconcile before exporting: never copy out a file group that belongs
        # to a group already deleted from the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, variant=variant)
        # Disk-only export: clip cache / raw frame cache. Latent-only segments
        # cannot be decoded here (no VAE in an HTTP request), so they are skipped
        # with a hint — the full latent decode happens during node execution when
        # the VAE is loaded.
        result = await asyncio.to_thread(
            run_segment_export,
            node_id,
            plan,
            indices,
            mode=mode,
            workflow_name=workflow_name,
            variant=variant,
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment-export failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


def _audio_seg_ids(body: dict, timeline_data: str) -> list[str]:
    """Ordered segment ids for「提取音频」(the key every entry is bound to).

    The editor sends them alongside the timeline; when it does not (an older
    build), they are read straight out of the timeline payload, which carries
    ``segments[].id``.
    """
    raw = body.get("seg_ids")
    if isinstance(raw, list) and raw:
        return [str(x or "").strip() for x in raw]
    from .audio_extract import timeline_segment_ids

    return timeline_segment_ids(timeline_data)


def _retain_ids(body: dict) -> list[str]:
    """Ids pinned by「保留音频」in the timeline payload.

    Pins live in the frontend's segment objects, not in the cache manifest, so the
    backend can only avoid dropping a pinned take when they are sent along — the
    duplicate sweep otherwise keeps whichever take is newest.
    """
    raw = body.get("retainIds")
    if not isinstance(raw, list):
        return []
    return [str(x or "").strip() for x in raw]


async def minimax_audio_extract_status(request):
    """Availability of every segment for「提取音频」(what the picker greys out)."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .plan_types import normalize_segment_export_source
        from .segment_cache import inspect_audio_extract_status, sync_segment_slots
        from .segment_slots import VARIANT_SECOND

        workflow_name = str(body.get("workflow_name") or "").strip() or None
        source = normalize_segment_export_source(body.get("source") or body.get("cacheSource"))
        variant = VARIANT_SECOND if source == "2nd" else "1st"

        plan = _plan_from_request(body, str(timeline_data))
        # Reconcile first: the picker must not offer audio for a render that
        # belongs to a group deleted from the middle of the timeline.
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, variant=variant)
        return web.json_response(
            inspect_audio_extract_status(
                node_id, plan, workflow_name=workflow_name, variant=variant
            )
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt audio-extract status failed: %s", exc)
        return web.json_response({"segments": [], "error": str(exc)}, status=400)


async def minimax_audio_extract(request):
    """Run a「提取音频」request: one WAV (+ audio latent) per checked segment."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    raw_indices = body.get("indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        return web.Response(status=400, text="No segment indices selected.")

    try:
        from .plan_types import normalize_segment_export_source
        from .segment_cache import run_audio_extract, sync_segment_slots
        from .segment_slots import VARIANT_SECOND

        source = normalize_segment_export_source(body.get("source") or body.get("cacheSource"))
        variant = VARIANT_SECOND if source == "2nd" else "1st"
        try:
            indices = [int(i) for i in raw_indices]
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid segment indices.")

        plan = _plan_from_request(body, str(timeline_data))
        workflow_name = str(body.get("workflow_name") or "").strip() or None
        sync_segment_slots(node_id, plan, workflow_name=workflow_name, variant=variant)
        result = await asyncio.to_thread(
            run_audio_extract,
            node_id,
            plan,
            indices,
            workflow_name=workflow_name,
            variant=variant,
            seg_ids=_audio_seg_ids(body, str(timeline_data)),
            retain_ids=_retain_ids(body),
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt audio-extract failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=500)


async def minimax_audio_extract_list(request):
    """Entries of the「音频」tab — every clip extracted for one timeline card."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .segment_cache import list_audio_extracts

        workflow_name = str(body.get("workflow_name") or "").strip() or None
        raw_index = body.get("index")
        index = int(raw_index) if raw_index is not None and str(raw_index) != "" else None
        # Reconciling here is what actually enforces「删除而删除」: a card removed
        # while the node was closed still gets its audio dropped the moment the
        # tab asks for the list.
        rows = await asyncio.to_thread(
            list_audio_extracts,
            node_id,
            workflow_name,
            seg_ids=_audio_seg_ids(body, str(timeline_data)),
            index=index,
            retain_ids=_retain_ids(body),
        )
        return web.json_response({"node_id": node_id, "entries": rows})
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt audio-extract list failed: %s", exc)
        return web.json_response({"entries": [], "error": str(exc)}, status=400)


async def minimax_audio_extract_remove(request):
    """Delete one extracted clip (the audio tab's per-row delete)."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")
    entry_id = str(body.get("entry_id") or "").strip()
    if not entry_id:
        return web.Response(status=400, text="Missing entry id.")

    try:
        from .segment_cache import remove_audio_entry

        workflow_name = str(body.get("workflow_name") or "").strip() or None
        removed = await asyncio.to_thread(remove_audio_entry, node_id, workflow_name, entry_id)
        return web.json_response({"removed": bool(removed)})
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt audio-extract remove failed: %s", exc)
        return web.json_response({"removed": False, "error": str(exc)}, status=400)


async def minimax_audio_extract_file(request):
    """Stream one extracted WAV so the「音频」tab can play it in place.

    Read-only, and only ever serves a name registered in the entry manifest, so
    a crafted ``entry`` cannot walk out of the audio directory.
    """
    node_id = str(request.query.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")
    entry_id = str(request.query.get("entry") or "").strip()
    if not entry_id:
        return web.Response(status=400, text="Missing entry id.")
    workflow_name = str(request.query.get("workflow_name") or "").strip() or None

    try:
        from .segment_cache import resolve_audio_file
    except Exception as exc:  # pragma: no cover - import guard
        log.warning("MiniMax H3 Director Opt audio-extract import failed: %s", exc)
        return web.Response(status=500, text="Audio cache unavailable.")

    path = resolve_audio_file(node_id, workflow_name, entry_id)
    if path is None:
        return web.Response(status=404, text="Extracted audio not found.")
    return web.FileResponse(
        str(path),
        headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"},
    )


async def minimax_segment_clip(request):
    """Stream a cached segment clip so the UI「二采」tab can play it in place.

    Read-only: resolves the same slot map the export status uses, so it always
    follows the segment currently living at ``index`` (and its superseded group
    when the newest stem is still empty). Never writes anything.
    """
    node_id = str(request.query.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")
    try:
        index = int(request.query.get("index") or 0)
    except Exception:
        return web.Response(status=400, text="Invalid segment index.")
    if index < 0:
        return web.Response(status=400, text="Invalid segment index.")

    variant = str(request.query.get("variant") or "").strip().lower()
    workflow_name = str(request.query.get("workflow_name") or "").strip() or None

    try:
        from .segment_cache import clip_cache_path
        from .segment_slots import VARIANT_FIRST, VARIANT_SECOND
    except Exception as exc:  # pragma: no cover - import guard
        log.warning("MiniMax H3 Director Opt segment-clip import failed: %s", exc)
        return web.Response(status=500, text="Segment cache unavailable.")

    variant_key = VARIANT_SECOND if variant in ("2nd", "second", "2") else VARIANT_FIRST
    try:
        path = clip_cache_path(
            node_id, index, workflow_name=workflow_name,
            allow_prev=True, variant=variant_key,
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director Opt segment-clip failed: %s", exc)
        return web.Response(status=404, text="Segment clip not cached.")
    if path is None:
        return web.Response(status=404, text="Segment clip not cached.")
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return web.Response(status=404, text="Segment clip not cached.")
    except OSError:
        return web.Response(status=404, text="Segment clip not cached.")

    return web.FileResponse(
        str(path),
        headers={"Cache-Control": "no-store"},
    )
