"""Slot map: which cache file group belongs to which timeline position.

Every durable per-segment artefact used to be named ``seg_<position:04d>_*``,
i.e. the file name carried the segment's **position** on the timeline. That made
the cache position-addressed: deleting a group in the middle shifted every later
segment onto the deleted group's files, because nothing recorded which render a
file actually belonged to (the old ``prune_segment_cache`` could only drop
indices that no longer existed, never files that had been displaced).

This module keeps that mapping explicit instead::

    <node cache dir>/segment_slots.json
        {"version": 2,
         "slots": [{"hash": "1a2b3c4d5e6f", "stem": "seg_1a2b3c4d5e6f",
                    "prev": "seg_9f8e7d6c5b4a"},
                   ...]}

``slots[k]`` describes timeline position ``k``. File names are derived from the
segment's **content hash** (prompt + references + duration + sampling,
see :func:`content_hash_of_fingerprint`), never from its position, so:

* deleting a group only removes *its own* files — later groups keep theirs;
* reordering groups carries each render along instead of invalidating it;
* re-adding an identical group re-adopts its previous files;
* two groups with identical content get ``seg_<hash>`` then ``seg_<hash>_1``.

``prev`` retains a group's previous file group for one generation. The content
hash is deliberately strict, so a pipeline bump or an fps change moves a group
to a new file group; keeping the old one for one generation is what lets
「选择运行」+「全部导出」still fill unselected slots from the last render
instead of leaving them blank — the same policy the old ``allow_stale`` fill had.

The map is advisory: every helper tolerates a missing or corrupt file and falls
back to the legacy positional name, and nothing here may raise into a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import cache_layout

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.slots")

#: Fingerprint key holding the source-clip identity (mirrors segment_cache).
SOURCE_VIDEO_FP_KEY = "source_video"

#: Slot map file, stored beside the artefacts it describes.
MANIFEST_NAME = "segment_slots.json"
MANIFEST_VERSION = 2

#: Hex characters of the content hash kept in a file name.
HASH_LEN = 12

_HASHED_STEM_RE = re.compile(r"^seg_([0-9a-f]{%d})(?:_(\d+))?$" % HASH_LEN)

_LOCK = threading.RLock()
#: ``path -> ((mtime_ns, size), slots)`` so path resolution stays off the disk.
_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], list[dict[str, Any]]]] = {}


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

def legacy_stem(position: int) -> str:
    """``seg_0003`` — the pre-slot-map name, kept for migration/fallback only."""
    return f"{cache_layout.SCRATCH_PREFIX}{int(position):04d}"


def content_stem(content_hash: str, dup: int = 0) -> str:
    """``seg_<hash>``, or ``seg_<hash>_1`` / ``_2`` for repeated content."""
    stem = f"{cache_layout.SCRATCH_PREFIX}{str(content_hash)[:HASH_LEN]}"
    return stem if int(dup) <= 0 else f"{stem}_{int(dup)}"


def stem_content_hash(stem: str) -> str | None:
    """Content hash encoded in a hash-named stem, or ``None`` for a legacy one."""
    match = _HASHED_STEM_RE.match(str(stem or ""))
    return match.group(1) if match else None


def content_hash_of_fingerprint(
    fingerprint: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
) -> str:
    """Hash a segment fingerprint into the file-name hash (position-free).

    ``index`` is dropped and, for timelines without a source video, the absolute
    ``start``/``end`` collapse to their ``length``: those describe *where a
    segment sits*, not *what it renders*. With a source video the absolute range
    is real content (it selects source frames) and is kept as-is.

    ``defaults`` patches in fingerprint keys that older caches predate, the same
    way ``segment_cache._fingerprint_compatible`` does, so a hash read back from
    an old ``meta.json`` stays comparable with a freshly built one.
    """
    data: dict[str, Any] = dict(fingerprint or {})
    if defaults:
        for key, value in defaults.items():
            data.setdefault(key, value)
    data.pop("index", None)
    if data.get(SOURCE_VIDEO_FP_KEY):
        return _hash_dict(data)
    start = data.pop("start", None)
    end = data.pop("end", None)
    length = data.pop("length", None)
    if length is None and start is not None and end is not None:
        try:
            length = int(end) - int(start)
        except (TypeError, ValueError):
            length = None
    data["length"] = length
    return _hash_dict(data)


def _hash_dict(data: dict[str, Any]) -> str:
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:HASH_LEN]


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------

def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_NAME


def has_manifest(root: Path) -> bool:
    """Whether a slot map exists. An *empty* map is still a map."""
    return manifest_path(root).is_file()


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (int(getattr(st, "st_mtime_ns", 0)), int(st.st_size))


def read_slots(root: Path) -> list[dict[str, Any]]:
    """Current slot list (``[]`` when the map is missing or unreadable)."""
    path = manifest_path(root)
    key = _stat_key(path)
    if key is None:
        with _LOCK:
            _MANIFEST_CACHE.pop(str(path), None)
        return []
    with _LOCK:
        hit = _MANIFEST_CACHE.get(str(path))
        if hit and hit[0] == key:
            return [dict(item) for item in hit[1]]
    slots: list[dict[str, Any]] = []
    data: Any = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = None
    if isinstance(data, dict):
        raw = data.get("slots")
        if isinstance(raw, list):
            slots = [
                dict(item)
                for item in raw
                if isinstance(item, dict) and str(item.get("stem") or "").strip()
            ]
    with _LOCK:
        _MANIFEST_CACHE[str(path)] = (key, [dict(item) for item in slots])
    return [dict(item) for item in slots]


def write_slots(root: Path, slots: Sequence[dict[str, Any]]) -> None:
    """Persist the slot list atomically. Never raises."""
    path = manifest_path(root)
    payload = {
        "version": MANIFEST_VERSION,
        "updated": int(time.time()),
        "slots": [dict(item) for item in slots],
    }
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.is_file():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    except OSError as exc:
        log.warning("Slot map write skipped (%s); this run keeps it in memory only.", exc)
    with _LOCK:
        _MANIFEST_CACHE.pop(str(path), None)


def clear_slots(root: Path) -> None:
    """Drop the slot map (used by「清空节点所有缓存」)."""
    path = manifest_path(root)
    with _LOCK:
        _MANIFEST_CACHE.pop(str(path), None)
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def resolve_stem(root: Path, position: int) -> str | None:
    """File stem of the group currently living at timeline ``position``.

    ``None`` means "this position has no cache" — the timeline shrank, or the
    position never existed. The legacy positional name is used only while no
    slot map has been written yet, so pre-migration caches stay readable.
    """
    if not has_manifest(root):
        return legacy_stem(position)
    slots = read_slots(root)
    pos = int(position)
    if not 0 <= pos < len(slots):
        return None
    return str(slots[pos].get("stem") or "") or None


def prev_stem(root: Path, position: int) -> str | None:
    """Previous-generation stem at ``position`` (last render before a churn)."""
    slots = read_slots(root)
    pos = int(position)
    if not 0 <= pos < len(slots):
        return None
    return str(slots[pos].get("prev") or "") or None


def slot_paths(root: Path, position: int, *, stale: bool = False) -> dict[str, Path] | None:
    """Artefact paths for ``position`` (``stale=True`` → previous generation)."""
    stem = prev_stem(root, position) if stale else resolve_stem(root, position)
    if not stem:
        return None
    return cache_layout.segment_paths(Path(root), stem)


# --------------------------------------------------------------------------
# Editing the list — the shared add / remove / move entry points
# --------------------------------------------------------------------------

def insert_slot(
    root: Path,
    position: int,
    content_hash: str,
    *,
    stem: str | None = None,
) -> dict[str, Any]:
    """Add a slot at ``position``; later slots shift down, no file is touched.

    Use this when a group is inserted into the timeline: existing groups keep
    the files they already own, the new slot starts out with none.
    """
    with _LOCK:
        slots = read_slots(root)
        pos = max(0, min(int(position), len(slots)))
        entry = {
            "hash": str(content_hash),
            "stem": str(stem).strip() if stem else _allocate(root, content_hash, _used_stems(slots)),
        }
        slots.insert(pos, entry)
        write_slots(root, slots)
        return dict(entry)


def remove_slot(root: Path, position: int, *, delete_files: bool = True) -> bool:
    """Remove the slot at ``position`` and, by default, its files.

    This is "delete the cache at this position": only this slot's file group is
    unlinked, every other group keeps the files it already has.
    """
    with _LOCK:
        slots = read_slots(root)
        pos = int(position)
        if not 0 <= pos < len(slots):
            return False
        entry = slots.pop(pos)
        write_slots(root, slots)
    if delete_files:
        delete_stem(root, entry.get("stem"))
        delete_stem(root, entry.get("prev"))
    return True


def move_slot(root: Path, src: int, dst: int) -> bool:
    """Move a slot — and therefore its cache — to another position."""
    with _LOCK:
        slots = read_slots(root)
        source, dest = int(src), int(dst)
        if not 0 <= source < len(slots):
            return False
        entry = slots.pop(source)
        slots.insert(max(0, min(dest, len(slots))), entry)
        write_slots(root, slots)
        return True


def sync_slots(
    root: Path,
    hashes: Sequence[str],
    *,
    adoptable: dict[str, list[str]] | None = None,
    gc: bool = False,
) -> list[dict[str, Any]]:
    """Reconcile the slot list against the current timeline content hashes.

    Called once per run (and by every cache-reading HTTP route) with one hash
    per timeline position, in order. For each position it

    1. keeps the existing stem when the content hash is unchanged;
    2. otherwise adopts an existing file group holding the same content — this
       is what makes reordering free and lets a re-added group reclaim its old
       render;
    3. otherwise allocates a fresh ``seg_<hash>[_n]`` stem

    and remembers the superseded group as ``prev`` so a fingerprint churn still
    leaves the last render reachable for「选择运行」fills and for export.

    ``gc`` defaults to **False** on purpose. A plan edit (e.g. re-wording one
    prompt) changes that position's content hash, which allocates a fresh,
    still-empty stem and demotes the rendered group to ``prev``. Deleting files
    right there would throw away the only existing render before the user has
    run anything — and every read-only HTTP route (``cached-segments``,
    ``segment-export-status``, ``align-to-next-status``) calls this on each
    poll, so the second poll after an edit would already have destroyed it.

    The generation is therefore only advanced — i.e. unreferenced file groups
    are only deleted — by the call sites that have just produced fresh cache
    (``gc=True`` after a completed run). Until then the orphaned groups stay on
    disk and remain re-adoptable by content hash.
    """
    wanted = [str(item) for item in (hashes or [])]
    with _LOCK:
        previous = read_slots(root)
        pool: dict[str, list[str]] = {}
        for key, stems in (adoptable or {}).items():
            pool[str(key)] = [str(item) for item in stems if item]
        for stem in _disk_stems(root):
            hashed = stem_content_hash(stem)
            if hashed:
                pool.setdefault(hashed, [])
                if stem not in pool[hashed]:
                    pool[hashed].append(stem)

        used: set[str] = set()
        slots: list[dict[str, Any]] = []
        for position, content_hash in enumerate(wanted):
            old = previous[position] if position < len(previous) else None
            old_stem = str(old.get("stem") or "") if old else ""
            # ``old_stem not in used`` matters with repeated content: an earlier
            # position may already have adopted this group, and two slots must
            # never share one file group.
            keeps_old = bool(
                old and old.get("hash") == content_hash and old_stem and old_stem not in used
            )
            stem = old_stem if keeps_old else ""
            if not stem:
                stem = _take(pool.get(content_hash), used)
            if not stem:
                stem = _allocate(root, content_hash, used)
            used.add(stem)

            entry: dict[str, Any] = {"hash": content_hash, "stem": stem}
            # Keep the superseded group for one generation so a fingerprint
            # churn still leaves a render for「选择运行」fills.
            kept_prev = str(old.get("prev") or "") if old else ""
            if old_stem and old_stem != stem:
                kept_prev = old_stem
            if kept_prev and kept_prev != stem:
                entry["prev"] = kept_prev
            slots.append(entry)

        write_slots(root, slots)
        if gc:
            keep: set[str] = set()
            for entry in slots:
                keep.add(entry["stem"])
                if entry.get("prev"):
                    keep.add(str(entry["prev"]))
            gc_orphan_files(root, keep)
        return [dict(item) for item in slots]


# --------------------------------------------------------------------------
# File group helpers
# --------------------------------------------------------------------------

def delete_stem(root: Path, stem: str | None) -> int:
    """Delete every artefact of one file group. Returns the number removed."""
    stem = str(stem or "").strip()
    if not stem:
        return 0
    removed = 0
    try:
        for path in Path(root).iterdir():
            if not path.is_file() or cache_layout.stem_of_filename(path.name) != stem:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed


def gc_orphan_files(root: Path, keep_stems: Iterable[str]) -> int:
    """Delete per-segment artefacts whose stem is not in ``keep_stems``.

    Scratch files, the encoding cache and the slot map itself are never touched.
    """
    keep = {str(item) for item in keep_stems if item}
    removed = 0
    try:
        for path in Path(root).iterdir():
            if not path.is_file():
                continue
            stem = cache_layout.stem_of_filename(path.name)
            if stem is None or stem in keep:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    if removed:
        log.info("Segment cache: dropped %d orphaned file(s) in %s.", removed, Path(root).name)
    return removed


def _disk_stems(root: Path) -> list[str]:
    """Every existing per-segment file stem, hash-named ones first."""
    hashed: list[str] = []
    other: list[str] = []
    try:
        for path in Path(root).iterdir():
            if not path.is_file():
                continue
            stem = cache_layout.stem_of_filename(path.name)
            if not stem:
                continue
            (hashed if stem_content_hash(stem) else other).append(stem)
    except OSError:
        return []
    return sorted(set(hashed)) + sorted(set(other))


def _used_stems(slots: Sequence[dict[str, Any]]) -> set[str]:
    return {str(item.get("stem") or "") for item in slots if item.get("stem")}


def _take(candidates: list[str] | None, used: set[str]) -> str:
    """First candidate stem nobody has claimed yet."""
    for stem in candidates or []:
        if stem and stem not in used:
            return stem
    return ""


def _allocate(root: Path, content_hash: str, used: set[str]) -> str:
    """Fresh ``seg_<hash>[_n]`` stem that is neither claimed nor on disk."""
    for dup in range(0, 64):
        stem = content_stem(content_hash, dup)
        if stem in used:
            continue
        if not _stem_on_disk(root, stem):
            return stem
    # Pathological: 64 generations of identical content. Keep going with a
    # time-based suffix rather than overwriting a live file group.
    return content_stem(content_hash, int(time.time()) % 100000)


def _stem_on_disk(root: Path, stem: str) -> bool:
    try:
        return any(
            path.is_file() and cache_layout.stem_of_filename(path.name) == stem
            for path in Path(root).iterdir()
        )
    except OSError:
        return False
