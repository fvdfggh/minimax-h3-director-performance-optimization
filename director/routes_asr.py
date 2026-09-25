"""``/asr_check`` — run the「音频有效性校验」on demand.

The node's ASR check used to ride the boolean ``asr_check`` widget, so re-checking
after a prompt edit meant re-generating the whole timeline. It is a button now:
the front end sends the current timeline plus the segment indices the user picked,
this route rebuilds the same plan the node would build, loads the *cached* audio
of those segments and grades it against the prompts as they are right now.

Nothing is generated here. A segment with no cached render is reported as missing
rather than silently skipped, so「这段没跑过」never looks like「这段通过了」.
"""

from __future__ import annotations

import logging

from aiohttp import web

from ..lib.constants import ROUTE_PREFIX
from .asr_check import run_asr_check
from .asr_runtime import asr_model_for
from .cache_readback import load_segment_audio_by_position
from .routes_common import _plan_from_request

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.routes.asr")

_NO_MODEL = (
    "音频有效性校验: 还没有可用的 ASR 模型。"
    "请先接线 ASR 模型口并运行一次节点（只需一次，之后可反复点按钮校验）。"
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


async def minimax_asr_check(request):
    """Grade the picked segments' cached audio against their current prompts."""
    body = await request.json()
    node_id = str(body.get("node_id") or "")
    workflow_name = body.get("workflow_name")
    wanted = _wanted_indices(body)

    handle = asr_model_for(node_id, workflow_name)
    if handle is None:
        return web.json_response({"success": False, "error": _NO_MODEL}, status=409)
    if not wanted:
        return web.json_response(
            {"success": False, "error": "音频有效性校验: 没有选中任何片段。"},
            status=400,
        )

    try:
        plan = _plan_from_request(body, str(body.get("timeline_data") or ""))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("音频有效性校验: 时间轴解析失败 (%s)", exc, exc_info=True)
        return web.json_response(
            {"success": False, "error": f"音频有效性校验: 时间轴解析失败 — {exc}"},
            status=400,
        )

    # collect_prompt_texts() filters on plan.run_indices, so pointing it at the
    # picked segments is what makes「按当前 prompt 核对」use exactly those lines.
    plan.run_indices = frozenset(wanted)

    audios = []
    missing: list[int] = []
    for seg in plan.segments or []:
        idx = int(getattr(seg, "index", -1))
        if idx not in wanted:
            continue
        # By position, not by fingerprint: the prompt is part of the fingerprint,
        # so a prompt edit — the very thing this button re-checks against — must
        # not make the cached audio look missing.
        audio = load_segment_audio_by_position(
            node_id, idx, allow_stale=True, workflow_name=workflow_name,
        )
        if audio is None:
            missing.append(idx + 1)
        else:
            audios.append(audio)

    if not audios:
        return web.json_response(
            {
                "success": False,
                "error": (
                    "音频有效性校验: 选中的片段都还没有可识别的音轨缓存"
                    f"（片段 {', '.join(str(m) for m in missing)}）。请先生成这些片段。"
                ),
            },
            status=404,
        )

    note = run_asr_check(plan, audios, handle)
    return web.json_response(
        {"success": True, "report": note, "missing": missing, "checked": len(audios)},
    )


def register(routes, register_route) -> None:
    """Attach this group's routes; called by :func:`http_routes.register_routes`."""
    register_route(routes, "POST", f"{ROUTE_PREFIX}/asr_check", minimax_asr_check)
