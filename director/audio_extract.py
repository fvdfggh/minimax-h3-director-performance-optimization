"""「提取音频」— a persistent, per-segment audio cache kept OUTSIDE the render cache.

Why this is not part of the segment cache
----------------------------------------

The render cache is keyed by **content hash** (``seg_<hash>_*``): editing a
prompt re-hashes a timeline position, hands it a fresh stem and demotes the
rendered group to ``prev``. Anything stored under that stem therefore *churns*
with the plan — which is exactly what the user must NOT happen to an audio clip
they deliberately pulled out and want to listen to later.

This store is deliberately separate:

* files live in ``<node dir>/audio_extract/`` and are named after an **entry id**,
  never a content hash and never a timeline position;
* nothing here participates in the render fingerprint, so re-running never
  invalidates it;
* the prefix ``audio_`` matches neither ``SEGMENT_GLOBS`` (``seg_*`` / ``seg2_*``)
  nor :func:`cache_layout.stem_of_filename`, so「清空缓存」/「清空全部缓存」and the
  slot map's orphan GC cannot reach it.

Binding —「随 index 移动而移动，删除而删除」
------------------------------------------

Each entry records the **segment id** the editor stamps on every timeline card
(``timeline.segments[i].id``). The timeline position is *derived*, never stored:

* reorder / insert in the middle → :func:`sync_audio_slots` recomputes every
  entry's index from the current id order, so an extracted clip follows its card;
* delete a card → its id is no longer in the list, so the entry and both of its
  files are unlinked.

Legacy timelines whose segments carry no id cannot be reasoned about, so
reconciliation is skipped for them rather than guessed (guessing would delete
audio the user still wants).

Layering: sits next to :mod:`cache_export`. Reads the render cache through
:mod:`cache_readback` / :mod:`cache_store` and never writes to it.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from ..lib.fs import safe_unlink as _safe_unlink
from ..lib.fs import write_json_atomic
from ..lib.video_export import write_wav
from . import segment_slots
from .cache_files import clip_cache_path
from .cache_paths import _cache_root
from .cache_readback import load_segment_audio
from .cache_store import _av_latent_to_cpu, load_segment_av_latent
from .plan_types import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.audio_extract")

#: Sub-directory of the node cache dir holding every extracted audio artefact.
AUDIO_EXTRACT_DIRNAME = "audio_extract"
#: Entry manifest inside that directory.
AUDIO_INDEX_NAME = "index.json"
INDEX_VERSION = 1

#: Entry ids are ours, but file names also arrive from disk (an index written by
#: an older build). Anything but these characters is refused before it is joined
#: onto the directory, so a crafted index can never escape the folder.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
#: Fallback key used for timelines whose segments carry no id. Never reconciled.
_POSITIONAL_PREFIX = "@"


def _audio_io():
    """``lib.audio_io``, imported on demand.

    It pulls in the video-decode side of :mod:`lib` (numpy only, but heavy
    enough that this module — reachable from ``segment_cache`` — should not
    drag it in at package import time for a feature most runs never touch).
    """
    from ..lib import audio_io

    return audio_io


# --------------------------------------------------------------------------
# Directory + index I/O
# --------------------------------------------------------------------------

def audio_extract_dir(
    node_id: str | None,
    workflow_name: str | None = None,
    *,
    create: bool = True,
) -> Path | None:
    """``<node dir>/audio_extract/`` (``None`` when the cache root is unusable)."""
    if not node_id:
        return None
    root = _cache_root(str(node_id), workflow_name)
    if root is None:
        return None
    d = root / AUDIO_EXTRACT_DIRNAME
    if create:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Audio extract dir unavailable (%s).", exc)
            return None
    return d


def read_audio_index(node_id: str | None, workflow_name: str | None = None) -> dict[str, Any]:
    """Entry manifest; a missing or corrupt file reads as empty (never raises)."""
    d = audio_extract_dir(node_id, workflow_name, create=False)
    if d is None:
        return {"version": INDEX_VERSION, "entries": []}
    try:
        payload = json.loads((d / AUDIO_INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": INDEX_VERSION, "entries": []}
    entries = payload.get("entries")
    entries = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
    return {"version": INDEX_VERSION, "entries": entries}


def write_audio_index(
    node_id: str | None,
    workflow_name: str | None,
    entries: list[dict[str, Any]],
) -> None:
    d = audio_extract_dir(node_id, workflow_name)
    if d is None:
        return
    write_json_atomic(d / AUDIO_INDEX_NAME, {"version": INDEX_VERSION, "entries": list(entries)})


def _entry_files(entry: dict[str, Any]) -> list[str]:
    names = []
    for key in ("wav", "latent"):
        name = entry.get(key)
        if isinstance(name, str) and _SAFE_NAME_RE.match(name):
            names.append(name)
    return names


def _delete_entry_files(
    node_id: str | None,
    workflow_name: str | None,
    entries: list[dict[str, Any]],
) -> int:
    """Unlink both artefacts of each entry. Returns how many files went away."""
    d = audio_extract_dir(node_id, workflow_name, create=False)
    if d is None:
        return 0
    removed = 0
    for entry in entries:
        for name in _entry_files(entry):
            if _safe_unlink(d / name):
                removed += 1
    return removed


# --------------------------------------------------------------------------
# Binding — re-derive positions from the current timeline order
# --------------------------------------------------------------------------

def timeline_segment_ids(timeline_data: Any) -> list[str]:
    """Ordered segment ids from an editor timeline payload.

    A segment without an id falls back to ``"@<i>"`` so the list stays aligned
    with the timeline; :func:`sync_audio_slots` refuses to reconcile when the
    timeline has no real ids at all.
    """
    if isinstance(timeline_data, (bytes, bytearray)):
        timeline_data = timeline_data.decode("utf-8", "replace")
    if isinstance(timeline_data, str):
        try:
            timeline_data = json.loads(timeline_data)
        except ValueError:
            return []
    if not isinstance(timeline_data, dict):
        return []
    segments = timeline_data.get("segments")
    if not isinstance(segments, list):
        return []
    out: list[str] = []
    for i, seg in enumerate(segments):
        sid = str(seg.get("id") or "").strip() if isinstance(seg, dict) else ""
        out.append(sid or f"{_POSITIONAL_PREFIX}{i}")
    return out


def _has_real_ids(seg_ids: list[str]) -> bool:
    return any(s and not s.startswith(_POSITIONAL_PREFIX) for s in seg_ids)


def sync_audio_slots(
    node_id: str | None,
    seg_ids: list[str] | None,
    workflow_name: str | None = None,
) -> list[dict[str, Any]]:
    """Re-bind every entry to the current timeline and drop the ones it lost.

    Two jobs, in this order:

    1. **删除而删除** — an entry whose ``seg_id`` is gone means its card was
       deleted, so both of its files are unlinked;
    2. **随 index 移动** — ``index`` is rewritten from the current id order, so
       reordering or inserting a card in the middle carries the audio along.

    Never raises. Returns the surviving entries (with fresh ``index`` values).
    """
    ids = [str(s or "").strip() for s in (seg_ids or [])]
    entries = read_audio_index(node_id, workflow_name)["entries"]
    if not entries:
        return []
    if not _has_real_ids(ids):
        # Legacy timeline: no ids to match against. Skipping is the only safe
        # answer — positionally "reconciling" would delete audio that simply
        # belongs to a workflow saved before ids existed.
        return entries

    known = set(ids)
    position = {sid: i for i, sid in enumerate(ids)}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    changed = False
    for entry in entries:
        sid = str(entry.get("seg_id") or "")
        if sid.startswith(_POSITIONAL_PREFIX):
            # Never bound to an id (extracted on a legacy timeline): it cannot
            # follow anything, but guessing a position would be worse — keep it.
            kept.append(entry)
            continue
        if sid in known:
            idx = int(position.get(sid, -1))
            if int(entry.get("index") or -1) != idx:
                entry["index"] = idx
                changed = True
            kept.append(entry)
        else:
            dropped.append(entry)

    changed = changed or bool(dropped)

    if changed:
        _delete_entry_files(node_id, workflow_name, dropped)
        write_audio_index(node_id, workflow_name, kept)
    return kept


def resolve_audio_file(
    node_id: str | None,
    workflow_name: str | None,
    entry_id: str,
) -> Path | None:
    """Path of an entry's playable WAV, or ``None`` (never escapes its folder)."""
    d = audio_extract_dir(node_id, workflow_name, create=False)
    if d is None:
        return None
    for entry in read_audio_index(node_id, workflow_name)["entries"]:
        if str(entry.get("id") or "") != str(entry_id or ""):
            continue
        name = entry.get("wav")
        if not isinstance(name, str) or not _SAFE_NAME_RE.match(name):
            return None
        path = d / name
        try:
            return path if path.is_file() and path.stat().st_size > 0 else None
        except OSError:
            return None
    return None


def remove_audio_entry(
    node_id: str | None,
    workflow_name: str | None,
    entry_id: str,
) -> bool:
    """Drop one entry and its files (the audio tab's per-row delete)."""
    entries = read_audio_index(node_id, workflow_name)["entries"]
    target = [e for e in entries if str(e.get("id") or "") == str(entry_id or "")]
    if not target:
        return False
    _delete_entry_files(node_id, workflow_name, target)
    write_audio_index(
        node_id, workflow_name,
        [e for e in entries if str(e.get("id") or "") != str(entry_id or "")],
    )
    return True


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(entry.get("id") or ""),
        "index": int(entry.get("index") or -1),
        "variant": str(entry.get("variant") or segment_slots.VARIANT_FIRST),
        "sample_rate": int(entry.get("sample_rate") or 0),
        "channels": int(entry.get("channels") or 0),
        "duration_s": round(float(entry.get("duration_s") or 0.0), 3),
        "src_kind": str(entry.get("src_kind") or ""),
        "has_latent": bool(entry.get("has_latent")),
        "created": int(entry.get("created") or 0),
        "wav": str(entry.get("wav") or ""),
    }


def list_audio_extracts(
    node_id: str | None,
    workflow_name: str | None = None,
    *,
    seg_ids: list[str] | None = None,
    index: int | None = None,
) -> list[dict[str, Any]]:
    """Entries for the「音频」tab, newest first (optionally one card only)."""
    entries = sync_audio_slots(node_id, seg_ids or [], workflow_name)
    rows = [_public_entry(e) for e in entries]
    if index is not None:
        rows = [r for r in rows if r["index"] == int(index)]
    rows.sort(key=lambda r: (-int(r["created"] or 0), str(r["id"])))
    return rows


# --------------------------------------------------------------------------
# Availability — what the picker greys out
# --------------------------------------------------------------------------

def audio_extract_availability(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any]:
    """What「提取音频」can read for one segment, without decoding anything.

    ``extractable`` covers only the sources an HTTP request can materialise —
    the decoded waveform cache and the rendered clip's audio track. A latent-only
    segment needs ``VAEDecodeAudio``, which only exists while a run is loaded, so
    it is reported as ``needsVae`` instead of being offered.
    """
    idx = int(seg.index)
    wave = load_segment_audio(
        node_id, seg, plan, allow_stale=True,
        workflow_name=workflow_name, variant=variant,
    )
    has_wave = (
        isinstance(wave, dict)
        and isinstance(wave.get("waveform"), torch.Tensor)
        and int(wave["waveform"].numel()) > 0
    )

    clip = clip_cache_path(
        node_id, idx, workflow_name=workflow_name, allow_prev=True, variant=variant
    )
    has_clip_audio = False
    if clip is not None and clip.is_file():
        # ``None`` means ffprobe is missing — stay optimistic and let the ffmpeg
        # attempt decide rather than greying out a segment that is fine.
        has_clip_audio = _audio_io().video_has_audio(str(clip)) is not False

    latent = load_segment_av_latent(
        node_id, seg, plan, allow_stale=True,
        workflow_name=workflow_name, variant=variant,
    )
    has_latent = isinstance(latent, dict) and "samples" in latent

    return {
        "index": idx,
        "hasWave": bool(has_wave),
        "hasClipAudio": bool(has_clip_audio),
        "hasAvLatent": bool(has_latent),
        # Playable now (no VAE in an HTTP request).
        "extractable": bool(has_wave or has_clip_audio),
        # A standalone audio latent costs nothing to slice out.
        "latentAvailable": bool(has_latent),
        # Nothing playable, but a run with the VAE loaded could decode the latent.
        "needsVae": bool(has_latent and not has_wave and not has_clip_audio),
    }


def inspect_audio_extract_status(
    node_id: str | None,
    plan: DirectorPlan,
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
) -> dict[str, Any]:
    """Per-segment availability for the「提取音频」picker."""
    rows = [
        audio_extract_availability(
            node_id, seg, plan, workflow_name=workflow_name, variant=variant
        )
        for seg in (getattr(plan, "segments", None) or [])
    ]
    return {
        "node_id": str(node_id or ""),
        "segments": rows,
        "extractable_count": sum(1 for r in rows if r["extractable"]),
        "source": str(variant),
    }


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _separate_audio_latent(latent: dict[str, Any] | None) -> dict[str, Any] | None:
    """Slice the audio half out of a cached AV latent (pure tensor work).

    :class:`LTXVSeparateAVLatent` only splits the nested tensor — no VAE, no
    GPU — so this runs fine inside an HTTP request.
    """
    if not isinstance(latent, dict) or "samples" not in latent:
        return None
    try:
        from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    except ImportError:
        log.debug("LTXVSeparateAVLatent unavailable; audio latent skipped.")
        return None
    try:
        sep = LTXVSeparateAVLatent.execute(latent)
        if hasattr(sep, "args"):
            sep = sep.args
        audio_latent = sep[1] if len(sep) > 1 else None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Audio latent split failed (%s).", exc)
        return None
    if not isinstance(audio_latent, dict) or "samples" not in audio_latent:
        return None
    try:
        payload = _av_latent_to_cpu(audio_latent)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Audio latent CPU copy failed (%s).", exc)
        return None
    return payload if isinstance(payload, dict) else None


def _wav_info(path: Path) -> tuple[int, int, float]:
    """``(sample_rate, channels, duration_s)`` read straight from a WAV header.

    Preferred over trusting the source dict: for a clip-extracted track the
    container's real rate is whatever ffmpeg negotiated, and the header is the
    only place both it and the exact length are stated.
    """
    try:
        import wave

        with wave.open(str(path), "rb") as handle:
            rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            frames = int(handle.getnframes())
    except (OSError, EOFError, ValueError):
        return 0, 0, 0.0
    return rate, channels, (frames / rate) if rate > 0 else 0.0


def _extract_one(
    node_id: str | None,
    directory: Path,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    workflow_name: str | None,
    variant: str,
    seg_ids: list[str],
) -> dict[str, Any] | None:
    """Write one segment's WAV (+ audio latent when available).

    Source priority is cheapest-first: the decoded waveform cache needs nothing
    but the stdlib; the rendered clip costs one ffmpeg pass; the latent would
    need a VAE and is only sliced, never decoded here.
    """
    idx = int(seg.index)
    entry_id = uuid.uuid4().hex[:8]
    wav_name = f"audio_{entry_id}.wav"
    latent_name = f"audio_{entry_id}_latent.pt"
    wav_path = directory / wav_name
    latent_path = directory / latent_name

    src_kind = ""
    meta = (0, 0, 0.0)

    wave = load_segment_audio(
        node_id, seg, plan, allow_stale=True,
        workflow_name=workflow_name, variant=variant,
    )
    if isinstance(wave, dict) and isinstance(wave.get("waveform"), torch.Tensor):
        if write_wav(wav_path, wave):
            src_kind = "wave"

    if not src_kind:
        clip = clip_cache_path(
            node_id, idx, workflow_name=workflow_name, allow_prev=True, variant=variant
        )
        if clip is not None and clip.is_file() and _audio_io().extract_audio_track(str(clip), str(wav_path)):
            src_kind = "clip"

    # The header is authoritative for both sources — it states the rate ffmpeg
    # actually negotiated and the exact length, neither of which the source dict
    # can be trusted for.
    meta = _wav_info(wav_path) if src_kind else (0, 0, 0.0)

    has_latent = False
    if src_kind:
        latent = load_segment_av_latent(
            node_id, seg, plan, allow_stale=True,
            workflow_name=workflow_name, variant=variant,
        )
        audio_latent = _separate_audio_latent(latent)
        if audio_latent is not None:
            payload = dict(audio_latent)
            payload["sample_rate"] = int(meta[0]) or 32000
            payload["source_variant"] = str(variant)
            payload["seg_index"] = idx
            payload["created"] = int(time.time())
            try:
                torch.save(payload, latent_path)
                has_latent = True
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Audio latent write failed for segment %d (%s).", idx + 1, exc)
                _safe_unlink(latent_path)

    if not src_kind:
        # Nothing playable on disk — do not leave a half-written entry behind.
        _safe_unlink(wav_path)
        _safe_unlink(latent_path)
        return None

    seg_id = str(seg_ids[idx]) if 0 <= idx < len(seg_ids) else f"{_POSITIONAL_PREFIX}{idx}"
    return {
        "id": entry_id,
        "seg_id": seg_id,
        "index": idx,
        "variant": str(variant),
        "wav": wav_name,
        "latent": latent_name if has_latent else "",
        "has_latent": bool(has_latent),
        "src_kind": src_kind,
        "sample_rate": int(meta[0]),
        "channels": int(meta[1]),
        "duration_s": round(float(meta[2]), 3),
        "created": int(time.time()),
    }


def run_audio_extract(
    node_id: str | None,
    plan: DirectorPlan,
    indices: list[int],
    *,
    workflow_name: str | None = None,
    variant: str = segment_slots.VARIANT_FIRST,
    seg_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a「提取音频」request: write a WAV (+ audio latent) per segment.

    Entries are **appended**, never replaced — an extraction is a snapshot the
    user asked for, so re-running appends another take instead of destroying the
    one already sitting in the「音频」tab.
    """
    ids = [str(s or "").strip() for s in (seg_ids or [])]
    directory = audio_extract_dir(node_id, workflow_name)
    if directory is None:
        return {
            "entries": [],
            "skipped": [{"index": int(i), "reason": "cache dir unavailable"} for i in indices],
            "dir": "",
            "source": str(variant),
        }

    by_index = {int(s.index): s for s in (getattr(plan, "segments", None) or [])}
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw in sorted({int(i) for i in indices}):
        seg = by_index.get(raw)
        if seg is None:
            skipped.append({"index": raw, "reason": "segment not in plan"})
            continue
        try:
            entry = _extract_one(
                node_id, directory, seg, plan,
                workflow_name=workflow_name, variant=variant, seg_ids=ids,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("提取音频 segment %d failed: %s", raw + 1, exc)
            skipped.append({"index": raw, "reason": str(exc)})
            continue
        if entry is None:
            skipped.append({"index": raw, "reason": "no playable audio source"})
            continue
        created.append(entry)

    if created:
        existing = read_audio_index(node_id, workflow_name)["entries"]
        write_audio_index(node_id, workflow_name, list(existing) + created)

    # Re-bind everything (also garbage-collects entries whose card is gone).
    sync_audio_slots(node_id, ids, workflow_name)

    return {
        "entries": [_public_entry(e) for e in created],
        "skipped": skipped,
        "dir": str(directory),
        "source": str(variant),
    }
