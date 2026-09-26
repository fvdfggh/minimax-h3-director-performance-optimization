"""``/asr_check`` + ``/asr_check_status`` — the「音频有效性校验」button.

The node's ASR check used to ride the boolean ``asr_check`` widget, so re-checking
after a prompt edit meant re-generating the whole timeline. It is a button now:
the front end sends the current timeline plus the segment indices the user picked,
this route rebuilds the same plan the node would build, loads the *cached* audio
of those segments and grades it against the prompts as they are right now.

``/asr_check_status`` answers the same question *before* the user picks: which
segments have readable audio for the chosen pass (so the picker can grey out the
rest) and whether an ASR model is known yet. Both routes read the audio through
:func:`_read_audio`, so「显示可选」and「点了能跑」cannot disagree.

An ASR model needs no run at all: a wired-then-run loader hands us *your* settings,
and with nothing wired we build the pack's handle with its defaults
(:func:`asr_runtime.autoload_asr_model`). The cached audio is read off disk either
way, which is why re-checking after a prompt edit costs one transcription and no
generation.

Both routes touch disk and (for the check) the model, so the heavy calls go
through ``asyncio.to_thread`` — a long transcription must not freeze the server's
event loop.

Nothing is generated here. A segment with no cached render is reported as missing
rather than silently skipped, so「这段没跑过」never looks like「这段通过了」.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web

from ..lib.constants import ROUTE_PREFIX
from .asr_check import auto_model_available, run_asr_check_detailed
from .asr_runtime import (
    asr_model_for,
    asr_node_state,
    autoload_asr_model,
    autoload_error,
    remembered_asr_nodes,
)
from .cache_readback import load_segment_audio_by_position
from .routes_common import _plan_from_request
from .segment_slots import VARIANT_FIRST, VARIANT_SECOND

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.routes.asr")

_NO_MODEL = "音频有效性校验: 没有可用的 ASR 模型。"


def _no_model_error(node_id: str) -> str:
    """``_NO_MODEL`` plus what actually went wrong.

    Reached only when the pack's handle cannot even be built for us (no model files,
    transformers mismatch, …) — a wired handle is optional now.
    """
    detail = autoload_error()
    if detail:
        return f"{_NO_MODEL}\n（自动构造句柄失败：{detail}）"
    known = remembered_asr_nodes()
    if known:
        return (
            f"{_NO_MODEL}\n（后端已登记模型的节点：{', '.join(known)}；"
            f"本次请求 node {node_id or '空'}）"
        )
    return (
        f"{_NO_MODEL}\n（既没有接线的模型口，也没有可供自动加载的本地模型："
        "请运行 ASR 节点目录中的 scripts/download_models.py。）"
    )


def _wanted_indices(body: dict) -> list[int]:
    raw = body.get("indices")
    if raw is None:
        raw = body.get("segments")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[int] = []
    for item in raw:
        # Accept both a bare index and the {index: n} rows the picker sends.
        value = item.get("index") if isinstance(item, dict) else item
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(i for i in out if i >= 0))


def _source_variant(body: dict) -> tuple[str, str]:
    """``(label for the report, cache variant)`` for the picker's 一采 / 二采."""
    if str(body.get("source") or "").lower() == "2nd":
        return "二采", VARIANT_SECOND
    return "一采", VARIANT_FIRST


def _plan_from_body(body: dict):
    """Plan described by a picker request. Raises on a malformed timeline."""
    return _plan_from_request(body, body.get("timeline_data") or "")


def _read_audio(node_id, index, workflow_name, variant):
    """Cached audio of one timeline position, or ``None``.

    By position, not by fingerprint: the prompt is part of the fingerprint, so a
    prompt edit — the very thing this button re-checks against — must not make the
    cached audio look missing.
    """
    return load_segment_audio_by_position(
        node_id, int(index), allow_stale=True, workflow_name=workflow_name, variant=variant,
    )


def _pick_audio(node_id, plan, wanted, workflow_name, variant):
    """``(audios, indices, missing)`` for ``wanted``, in timeline order.

    Timeline order matters: the verdict maps speakers to the order the audience
    hears them in, and the audio is concatenated in exactly that order — so
    ``indices`` (the timeline position of each returned clip, index-aligned with
    ``audios``) is what lets the recognised speech be traced back to its segment.
    ``missing`` carries 1-based numbers for the report.
    """
    audios: list = []
    indices: list[int] = []
    missing: list[int] = []
    for seg in plan.segments or []:
        idx = int(getattr(seg, "index", -1))
        if idx not in wanted:
            continue
        audio = _read_audio(node_id, idx, workflow_name, variant)
        if audio is None:
            missing.append(idx + 1)
        else:
            audios.append(audio)
            indices.append(idx)
    return audios, indices, missing


def _status_rows(node_id, plan, workflow_name, variant) -> list[dict[str, Any]]:
    """Per-segment「这段有没有可读音轨」— one position-keyed cache read each."""
    rows = []
    for seg in plan.segments or []:
        idx = int(getattr(seg, "index", -1))
        rows.append({
            "index": idx,
            "hasAudio": _read_audio(node_id, idx, workflow_name, variant) is not None,
        })
    return rows


async def minimax_asr_status(request):
    """Which segments the picker may offer, and whether a model is known yet."""
    body = await request.json()
    node_id = str(body.get("node_id") or "")
    workflow_name = body.get("workflow_name")
    _, variant = _source_variant(body)

    try:
        plan = _plan_from_body(body)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("音频有效性校验: 状态探测失败 (%s)", exc, exc_info=True)
        return web.json_response(
            {"segments": [], "error": f"音频有效性校验: 时间轴解析失败 — {exc}"},
            status=400,
        )

    rows = await asyncio.to_thread(_status_rows, node_id, plan, workflow_name, variant)
    wired = asr_model_for(node_id, workflow_name) is not None
    # Nothing wired → offer the pack's own handle instead of demanding a run. The
    # probe is cheap (no hashing, no GPU), so it is safe on every picker open.
    auto = (not wired) and await asyncio.to_thread(auto_model_available)
    return web.json_response({
        "segments": rows,
        "withAudio": sum(1 for r in rows if r["hasAudio"]),
        "source": variant,
        "hasModel": wired or auto,
        # True when the check will run on the handle *we* build (default device /
        # precision) rather than on the settings the user wired up.
        "autoModel": bool(auto),
        # Which nodes currently hold a handle: lets the picker say "your model is
        # registered on another node" before the user even clicks Run.
        "knownNodes": remembered_asr_nodes(),
        # ready / ran-without / never-ran — "I ran it but nothing happened" and
        # "I never ran it" need opposite fixes, so say which one it is.
        "modelState": asr_node_state(node_id),
    })


async def minimax_asr_check(request):
    """Grade the picked segments' cached audio against their current prompts."""
    body = await request.json()
    node_id = str(body.get("node_id") or "")
    workflow_name = body.get("workflow_name")
    wanted = _wanted_indices(body)

    # Your wired settings when a run has parked them; otherwise the pack's own
    # handle, built here — building it hashes the model files, hence the thread.
    handle = asr_model_for(node_id, workflow_name)
    if handle is None:
        handle = await asyncio.to_thread(autoload_asr_model)
    if handle is None:
        return web.json_response({"success": False, "error": _no_model_error(node_id)}, status=409)
    if not wanted:
        return web.json_response(
            {"success": False, "error": "音频有效性校验: 没有选中任何片段。"},
            status=400,
        )

    try:
        plan = _plan_from_body(body)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("音频有效性校验: 时间轴解析失败 (%s)", exc, exc_info=True)
        return web.json_response(
            {"success": False, "error": f"音频有效性校验: 时间轴解析失败 — {exc}"},
            status=400,
        )

    # collect_prompt_texts() filters on plan.run_indices, so pointing it at the
    # picked segments is what makes「按当前 prompt 核对」use exactly those lines.
    plan.run_indices = frozenset(wanted)

    # Which pass's audio to grade — the picker offers 一采 / 二采 like「提取音频」.
    # Reading the cached audio is disk work; transcription below is model work.
    # Both go through a worker thread so the server keeps serving its UI.
    source_label, variant = _source_variant(body)
    audios, indices, missing = await asyncio.to_thread(
        _pick_audio, node_id, plan, wanted, workflow_name, variant
    )

    if not audios:
        return web.json_response(
            {
                "success": False,
                "error": (
                    f"音频有效性校验: 选中的片段在{source_label}缓存里都没有可识别的音轨"
                    f"（片段 {', '.join(str(m) for m in missing)}）。请先生成这些片段，"
                    "或换一个缓存来源。"
                ),
            },
            status=404,
        )

    note, details = await asyncio.to_thread(
        run_asr_check_detailed, plan, audios, handle, seg_indices=indices
    )
    return web.json_response(
        {
            "success": True,
            "report": note,
            "missing": missing,
            "checked": len(audios),
            # Per-segment「本段台词 vs 本段识别到」— the preview tab renders these,
            # one card at a time (see asr_check.js renderAsrReportInto).
            "segments": details,
        },
    )


def register(routes, register_route) -> None:
    """Attach this group's routes; called by :func:`http_routes.register_routes`."""
    register_route(routes, "POST", f"{ROUTE_PREFIX}/asr_check_status", minimax_asr_status)
    register_route(routes, "POST", f"{ROUTE_PREFIX}/asr_check", minimax_asr_check)
