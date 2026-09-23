"""The per-node segment params map: encoding key + canvas params, per segment.

「二次采样」loads conditioning straight from disk and must reproduce the *same*
``cond_text_<key>.pt``. That key depends on reference pixels which only exist
while the first pass is encoding, so it can never be recomputed later — recording
it when the first pass stores the encoding closes that gap.

Entries are keyed by the segment's **content hash**, never by timeline position: a
position-keyed map hands a segment its neighbour's prompt after an insert or
reorder and still looks valid (the referenced file exists), i.e. it fails
silently.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..lib.fs import write_json_atomic
from .conditioning_keys import _get_cache_dir

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.conditioning.params")


#: *and* still looks valid (the referenced file exists), i.e. it fails silently.
#: Per-node map remembering, for every segment, the exact text-encoding +
#: canvas/sample parameters the FIRST pass sampled it with. The second pass
#: (「二次采样」) loads conditioning straight from disk and must reproduce the
#: *same* ``cond_text_<key>.pt`` — its refs are pixel tensors that only exist
#: while encoding runs, so the key can never be recomputed afterwards. Writing
#: it when the first pass stores the encoding closes that gap.
#:
#: Entries are keyed by the segment's **content hash**, never by its timeline
#: position: the slot map already proved that a position is not an identity —
#: inserting or reordering a group shifts every later segment onto another one's
#: index, and a position-keyed map then hands a segment its neighbour's prompt
SEG_PARAMS_MAP = "segment_text_keys.json"

#: Hard cap on how many segments the map keeps. Content-hash keys accumulate one
#: entry per distinct render, so prompt edits grow the file forever; the oldest
#: entries are the ones no timeline can reference any more.
SEG_PARAMS_MAX_ENTRIES = 1024


def _seg_params_map_path(node_id: str | None, workflow_name: str | None) -> Path:
    return _get_cache_dir(node_id, workflow_name) / SEG_PARAMS_MAP


def _read_seg_params_map(node_id: str | None, workflow_name: str | None) -> dict[str, dict]:
    path = _seg_params_map_path(node_id, workflow_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segments, dict):
            return {}
        return {str(k): v for k, v in segments.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _write_seg_params_map(node_id: str | None, workflow_name: str | None, segments: dict) -> None:
    path = _seg_params_map_path(node_id, workflow_name)
    try:
        payload = {"updated": int(time.time()), "segments": segments}
        write_json_atomic(path, payload)
    except OSError as exc:
        log.warning("Segment params map write skipped (%s).", exc)


def _params_key(slot_key, segment_index) -> str:
    """Map key of one segment: its content hash, or its index as a fallback.

    ``slot_key`` is ``segment_slots``' content hash (prompt + references +
    duration + sampling). The positional fallback only exists for callers that
    have no plan at hand; it is never correct after a timeline edit, so the
    second pass must always pass the hash.
    """
    text = str(slot_key or "").strip()
    return text if text else str(int(segment_index))


def save_segment_second_params(
    node_id: str | None,
    workflow_name: str | None,
    segment_index: int,
    *,
    slot_key: str | None = None,
    text_key: str = "",
    ctx_w: int = 0,
    ctx_h: int = 0,
    sample_len: int = 0,
    num_frames: int = 0,
    frame_count: int = 0,
    context_n: int = 0,
    task_key: str = "",
    ref_image_size: str = "match",
    positive_prompt: str = "",
) -> None:
    """Remember the first-pass text/context identity of one segment.

    ``context_n`` is the number of context frames the first pass *actually*
    pinned (0 when it pinned nothing, e.g. because the previous segment had not
    been sampled in that run). The cached encoding only has room for that many
    head frames, so the second pass has to reuse this exact number instead of
    re-deriving it from the widget.
    """
    if not node_id:
        return
    segments = _read_seg_params_map(node_id, workflow_name)
    segments[_params_key(slot_key, segment_index)] = {
        "text_key": str(text_key or ""),
        "ctx_w": int(ctx_w or 0),
        "ctx_h": int(ctx_h or 0),
        "sample_len": int(sample_len or 0),
        "num_frames": int(num_frames or 0),
        "frame_count": int(frame_count or 0),
        "context_n": int(context_n or 0),
        "task_key": str(task_key or ""),
        "ref_image_size": str(ref_image_size or "match"),
        "positive_prompt": str(positive_prompt or ""),
        "ts": int(time.time()),
    }
    if len(segments) > SEG_PARAMS_MAX_ENTRIES:
        # Content-hash keys never reuse a slot, so an edited prompt leaves its
        # predecessor behind. Keep the newest ones — an entry no timeline can
        # reference is worthless, and without a cap the map grows unbounded.
        for stale in sorted(
            segments, key=lambda k: int(segments[k].get("ts") or 0)
        )[: len(segments) - SEG_PARAMS_MAX_ENTRIES]:
            segments.pop(stale, None)
    _write_seg_params_map(node_id, workflow_name, segments)


def load_segment_second_params(
    node_id: str | None,
    workflow_name: str | None,
    segment_index: int,
    *,
    slot_key: str | None = None,
) -> dict | None:
    """First-pass params a segment was sampled with, or ``None`` (no map entry).

    ``slot_key`` must be the segment's content hash (see :func:`_params_key`).
    Reading by position is only a last-resort fallback and is deliberately NOT
    applied when a hash is supplied: after a timeline reorder the entry sitting
    at that index belongs to a different segment.
    """
    if not node_id:
        return None
    segments = _read_seg_params_map(node_id, workflow_name)
    entry = segments.get(_params_key(slot_key, segment_index))
    return dict(entry) if entry else None
