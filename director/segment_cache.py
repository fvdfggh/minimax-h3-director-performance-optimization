"""Segment cache facade: slot reconciliation, plus every layer re-exported.

The segment cache used to be one 3000-line module. It is now six layers, each
with a single job:

* :mod:`cache_paths`    — where a segment's files live, and whether what is on
                          disk still belongs to it (content / sampling
                          fingerprint);
* :mod:`cache_codecs`   — pixel-frame and latent codecs, head/tail seam windows;
* :mod:`cache_files`    — clip artefacts, availability probes, shape probes;
* :mod:`cache_store`    — writing a segment's artefacts, second-pass writer;
* :mod:`cache_readback` — reading them back for the next segment / the export;
* :mod:`cache_export`   — the「分段导出」pipeline (decode, stitch, write mp4s).

What is left *here* is the slot-map reconciliation — the one part that ties a
timeline position to a file group — plus this facade: everything the rest of the
package used to import from ``segment_cache`` is still importable from here, so
call sites did not have to move. **New code should import from the specific
layer**, not from this module.

Cache is best-effort: write failures (cloud RO mounts, same-name overwrite
blocks, full disks) must never abort the main generation run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import cache_layout
from . import segment_slots
from .cache_export import (
    build_run_selection_clips,
    continuous_export_runs,
    merge_run_audio,
    predecode_latent_segments,
    run_segment_export,
    _expected_export_frames,
    _load_segment_export_source,
)
from .cache_files import (
    clip_cache_path,
    inspect_segment_export_status,
    inspect_second_sample_status,
    probe_segment_cache_shape,
)
from .cache_paths import _cache_root, _fingerprint_defaults, slot_content_hash
from .cache_readback import load_segment_audio, load_segment_cache
from .cache_store import (
    has_next_segment_av_latent,
    load_next_segment_av_latent,
    load_segment_av_latent,
    load_segment_handoff_meta,
    save_second_pass_cache,
    save_segment_cache,
)
from .plan_types import DirectorPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")


def sync_segment_slots(
    node_id: str | None,
    plan: DirectorPlan,
    workflow_name: str | None = None,
    *,
    gc: bool = False,
    variant: str = segment_slots.VARIANT_FIRST,
) -> None:
    """Reconcile a pass's cache files with the current timeline.

    The slot-aware replacement for the old index-based ``prune_segment_cache``:
    a group deleted in the middle of the timeline takes its own files with it,
    while every other group keeps — or re-adopts — the render matching its
    content. Never raises; a failed sync only costs cache reuse.

    ``variant="2nd"`` runs the same reconciliation on the second-pass map
    (``segment_slots_2nd.json``). Its hashes are the **first-pass** content
    hashes: the second-pass map exists to answer "which timeline position has a
    seg2 result", so it must stay position-aligned with the first pass. When a
    first-pass render churns (prompt edited), the slot churns too and the old
    ``seg2`` group drops to ``prev`` — exactly the stale marker the UI needs.

    ``gc`` controls whether the *generation is advanced*, i.e. whether file
    groups that no slot references any more are deleted from disk. It defaults
    to ``False``: editing a prompt re-hashes that position, which hands it a
    fresh empty stem and demotes the rendered group to ``prev`` — deleting
    then would discard the last render before anything has been re-generated.
    Only call sites that have just produced fresh cache pass ``gc=True``.
    """
    if not node_id:
        return
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return
    try:
        segments = list(getattr(plan, "segments", None) or [])
        hashes = [slot_content_hash(seg, plan) for seg in segments]
        adoptable = None
        is_second = str(variant) == segment_slots.VARIANT_SECOND
        if not is_second and not segment_slots.has_manifest(root):
            # First run after upgrading: adopt the positional caches by content
            # so nothing is re-rendered just because the file names changed.
            # The second pass has no legacy positional files, so no adoption.
            adoptable = _adoptable_stems(root)
        segment_slots.sync_slots(root, hashes, adoptable=adoptable, gc=gc, variant=variant)
    except Exception as exc:
        log.warning("Segment cache slot sync skipped (%s).", exc)


def sync_second_segment_slots(
    node_id: str | None,
    plan: DirectorPlan,
    workflow_name: str | None = None,
    *,
    gc: bool = False,
) -> None:
    """Sync the second-pass slot map so it stays aligned with the timeline."""
    sync_segment_slots(
        node_id,
        plan,
        workflow_name,
        gc=gc,
        variant=segment_slots.VARIANT_SECOND,
    )


def remove_segment_slot(
    node_id: str | None,
    index: int,
    workflow_name: str | None = None,
) -> bool:
    """Drop the cache of the group living at timeline ``index``.

    Called when the UI deletes a group: only that group's file group is
    unlinked, every other group keeps the files it already owns, and the slot
    list closes the gap so the following groups keep their own position → files
    mapping (the next :func:`sync_segment_slots` re-adopts them by content).
    Never raises — a failed drop only leaves files behind.
    """
    if not node_id:
        return False
    root = _cache_root(node_id, workflow_name)
    if root is None:
        return False
    try:
        return segment_slots.remove_slot(root, int(index))
    except Exception as exc:
        log.warning("Segment cache drop at position %s skipped (%s).", index, exc)
        return False


def _adoptable_stems(root: Path) -> dict[str, list[str]]:
    """``content hash -> file stems`` for caches written before the slot map.

    Those files are named after a *position*, so the only way to know which
    render one holds is to read the fingerprint it stored.
    """
    out: dict[str, list[str]] = {}
    try:
        for meta_path in sorted(root.glob("*_meta.json")):
            if meta_path.name.endswith("_pre_meta.json"):
                continue
            stem = cache_layout.stem_of_filename(meta_path.name)
            if not stem or segment_slots.stem_content_hash(stem):
                continue
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(stored, dict):
                continue
            content_hash = segment_slots.content_hash_of_fingerprint(
                stored, defaults=_fingerprint_defaults()
            )
            out.setdefault(content_hash, []).append(stem)
    except OSError:
        return out
    return out
