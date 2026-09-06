"""Single source of truth for MiniMax H3 Director on-disk cache layout.

All Director caches live under one root so a workflow's state can be inspected,
backed up or deleted as a single folder::

    output/minimax_director_cache/<workflow slug>/node_<id>/

Every artefact type shares that one directory and is told apart by file-name
prefix, so the six kinds stay recognisable in Explorer without nesting:

==========================  ======================================  ==========
Kind                        File name                               Lifetime
==========================  ======================================  ==========
text encoding               ``cond_text_<hash>.pt``                 reusable
image encoding              ``cond_image_<hash>.pt``                reusable
video encoding              ``cond_video_<hash>.pt``                reusable
sampled latent              ``seg_<hash>_latent.pt``                durable
decoded frames              ``seg_<hash>_frames.pt``                legacy/optional
head+tail frames            ``seg_<hash>_frames_ht.pt``             durable
audio latent                ``seg_<hash>_audio.pt``                 durable
rendered clip               ``seg_<hash>_clip.mp4``                 durable
segment meta / handoff      ``seg_<hash>_meta.json`` / ``_handoff`` durable
slot map (position→files)   ``segment_slots.json``                  durable
batch scratch               ``seg_XXXX_scratch_*.pt``               per-run
==========================  ======================================  ==========

``<hash>`` is the segment's **content** hash (prompt + references + duration +
sampling), not its position — see :mod:`segment_slots`, which owns the
timeline-position → file-group mapping. Repeated content appends ``_1``, ``_2``.

The split matters for clearing: scratch files are regenerate-in-place working
state, while ``seg_<hash>_*`` are what motion context and「全部导出」read back, so
``SCRATCH_PREFIX`` is the only prefix a run may delete on its own.

Historically these lived in three sibling roots (``minimax_seg_cache``,
``minimax_batch_cache``, ``minimax_conditioning_cache``). Merging them is a
path change, so previously written caches are no longer found.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache_layout")

#: Unified cache root under ComfyUI's output directory.
CACHE_ROOT = "minimax_director_cache"

#: Legacy roots, kept only so diagnostics and one-off cleanups can find them.
LEGACY_ROOTS = (
    "minimax_seg_cache",
    "minimax_batch_cache",
    "minimax_conditioning_cache",
)

# --- encoding cache prefixes (shared across segments, keyed by content hash) --
TEXT_PREFIX = "cond_text"
IMAGE_PREFIX = "cond_image"
VIDEO_PREFIX = "cond_video"
#: Glob matching every encoding cache regardless of kind.
ENC_PREFIXES = (TEXT_PREFIX, IMAGE_PREFIX, VIDEO_PREFIX)

# --- vision-tower (ViT) output cache, persisted ACROSS runs -------------------
#
# Reference images/videos enter the text encoder (Qwen3-VL) as vision entries and
# each one runs through the ViT in ``preprocess_embed`` -- that pass, not the VAE,
# dominates conditioning cost. Its output depends only on the pixels, the shape
# (which fixes ``grid``), the model variant and the image/video flag, so it is
# cached globally under its own directory rather than per workflow/node: the same
# media reused by any segment or any workflow hits the same entry.
VIT_CACHE_DIRNAME = "_vit"
VIT_PREFIX = "vit"
VIT_SUFFIX = ".pt"
#: Soft cap for the whole ViT cache in bytes. Cached ViT output is *larger* than
#: the source pixels (DeepStack carries one tensor per injected layer), so this
#: needs an explicit ceiling and LRU trimming.
VIT_CACHE_MAX_BYTES = 8 * 1024**3

# --- per-segment durable artefacts -------------------------------------------
LATENT_SUFFIX = "_latent.pt"
FRAMES_SUFFIX = "_frames.pt"
FRAMES_HT_SUFFIX = "_frames_ht.pt"
#: Number of leading/trailing frames kept in ``frames_ht`` (mirrors
#: ``segment_continuity._seam_window()`` so the seam pipeline has real pixels
#: without persisting the whole segment tensor).
FRAMES_HT_N = 16
AUDIO_SUFFIX = "_audio.pt"
CLIP_SUFFIX = "_clip.mp4"
META_SUFFIX = "_meta.json"
HANDOFF_SUFFIX = "_handoff.json"

# --- per-segment per-run working state ---------------------------------------
SCRATCH_PREFIX = "seg_"
SCRATCH_MARK = "_scratch_"
#: Glob matching any batch scratch file.
SCRATCH_GLOB = f"{SCRATCH_PREFIX}*{SCRATCH_MARK}*.pt"

# Windows-illegal path characters, Windows reserved device names.
_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


def output_root() -> Path:
    """ComfyUI's output directory."""
    import folder_paths

    return Path(folder_paths.get_output_directory())


def slugify_workflow_name(name: str | None) -> str:
    """Turn a workflow name into a single safe path component.

    CJK names are kept as-is because the directory names are user-facing —
    someone with several workflows wants to recognise them in Explorer. Only
    path separators and Windows-illegal characters are stripped, and the result
    is capped so the full path stays well inside Windows' 260-char limit.

    Returns "" when nothing usable remains, which means "no workflow namespace".
    """
    text = str(name or "").strip()
    if not text:
        return ""
    text = os.path.basename(text.replace("\\", "/"))
    text = _WIN_ILLEGAL.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    # Keep it short: this nests under output/ and above node_<id>/.
    text = text[:60]
    if not text or _WIN_RESERVED.match(text):
        return ""
    return text


def workflow_root(workflow_name: str | None = None) -> Path:
    """``<root>[/<slug>]`` — the workflow namespace layer.

    The workflow layer keeps caches from differently named workflows from
    colliding on the same ``node_<id>``: node ids are per-graph and routinely
    reused across files.
    """
    base = output_root() / CACHE_ROOT
    slug = slugify_workflow_name(workflow_name)
    if slug:
        base = base / slug
    return base


def node_cache_dir(
    node_id: str | None = None,
    workflow_name: str | None = None,
    *,
    create: bool = True,
) -> Path:
    """Directory holding every cache artefact for one Director node.

    All six kinds are written side by side here and distinguished by prefix
    (see module docstring), which is what lets one folder represent a whole
    workflow's state.
    """
    base = workflow_root(workflow_name)
    if node_id:
        base = base / f"node_{node_id}"
    if create:
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Cache dir unavailable (%s); caching disabled for this run.", exc)
    return base


def legacy_stem(seg_index: int) -> str:
    """``seg_0003`` — the pre-slot-map name, kept for migration and fallbacks."""
    return f"{SCRATCH_PREFIX}{int(seg_index):04d}"


def segment_paths(root: Path, stem: str) -> dict[str, Path]:
    """Every per-segment artefact path for one file *stem* under ``root``.

    ``stem`` comes from the slot map (:mod:`segment_slots`) and names the
    segment's **content**, never its position — so a group deleted in the middle
    of the timeline does not move anyone else's files.
    """
    return {
        "latent": root / f"{stem}{LATENT_SUFFIX}",
        "frames": root / f"{stem}{FRAMES_SUFFIX}",
        "frames_ht": root / f"{stem}{FRAMES_HT_SUFFIX}",
        "audio": root / f"{stem}{AUDIO_SUFFIX}",
        "clip": root / f"{stem}{CLIP_SUFFIX}",
        "meta": root / f"{stem}{META_SUFFIX}",
        "handoff": root / f"{stem}{HANDOFF_SUFFIX}",
    }


#: Suffixes of every durable per-segment artefact, **longest first** so
#: :func:`stem_of_filename` cannot mistake ``_pre_meta.json`` for ``_meta.json``.
#: The ``_pre_*`` entries belong to a removed feature but stay listed so files
#: left behind by older runs are still recognised and garbage-collected.
SEGMENT_SUFFIXES = (
    "_pre_handoff.json",
    "_pre_frames.pt",
    "_pre_latent.pt",
    "_pre_meta.json",
    "_frames_ht.pt",
    "_handoff.json",
    "_latent.pt",
    "_frames.pt",
    "_audio.pt",
    "_clip.mp4",
    "_meta.json",
)


def stem_of_filename(name: str) -> str | None:
    """File stem of a durable per-segment artefact, else ``None``.

    Scratch files and the encoding cache are excluded on purpose: they have
    their own lifecycle and must never be garbage-collected by the slot map.
    """
    if not name.startswith(SCRATCH_PREFIX) or SCRATCH_MARK in name:
        return None
    for suffix in SEGMENT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] or None
    return None


def scratch_path(root: Path, seg_index: int, kind: str) -> Path:
    """Per-run working file for ``seg_index`` (``cond``/``ref``/``latent``)."""
    stem = f"{SCRATCH_PREFIX}{int(seg_index):04d}{SCRATCH_MARK}{kind}"
    return root / f"{stem}.pt"


def encoding_path(root: Path, kind: str, key: str) -> Path:
    """Content-addressed encoding cache path (``text``/``image``/``video``)."""
    prefix = {
        "text": TEXT_PREFIX,
        "image": IMAGE_PREFIX,
        "video": VIDEO_PREFIX,
    }.get(str(kind), TEXT_PREFIX)
    return root / f"{prefix}_{key}.pt"


def iter_encoding_files(root: Path) -> list[Path]:
    """Every encoding cache file under ``root`` (any workflow/node depth)."""
    out: list[Path] = []
    for prefix in ENC_PREFIXES:
        out.extend(p for p in root.rglob(f"{prefix}_*.pt") if p.is_file())
    return out


def iter_scratch_files(root: Path) -> list[Path]:
    """Per-run scratch files under ``root`` — safe to delete after a run."""
    return [p for p in root.rglob(SCRATCH_GLOB) if p.is_file()]


def iter_segment_files(root: Path) -> list[Path]:
    """Durable per-segment artefacts under ``root`` (never auto-deleted)."""
    suffixes = (
        LATENT_SUFFIX,
        FRAMES_SUFFIX,
        AUDIO_SUFFIX,
        CLIP_SUFFIX,
        META_SUFFIX,
        HANDOFF_SUFFIX,
    )
    out: list[Path] = []
    for p in root.rglob(f"{SCRATCH_PREFIX}*"):
        if not p.is_file():
            continue
        name = p.name
        if SCRATCH_MARK in name:
            continue  # scratch, not durable
        if any(name.endswith(s) for s in suffixes):
            out.append(p)
    return out


def clear_state_path(node_id: str | None, workflow_name: str | None = None) -> Path:
    """Marker file for the edge-triggered clear button.

    Sits beside the per-node dirs rather than inside one, so clearing the cache
    cannot delete the record of the button already having been pressed.
    """
    base = workflow_root(workflow_name)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    slug = slugify_workflow_name(workflow_name)
    tag = f"{slug}." if slug else ""
    return base / f".clear_button_state.{tag}node_{node_id}"


def cache_stats(root: Path) -> dict[str, Any]:
    """Size/count breakdown of every cache kind under ``root``."""
    enc = iter_encoding_files(root)
    scratch = iter_scratch_files(root)
    seg = iter_segment_files(root)
    size = lambda files: sum(f.stat().st_size for f in files)  # noqa: E731
    return {
        "encoding_files": len(enc),
        "encoding_bytes": size(enc),
        "scratch_files": len(scratch),
        "scratch_bytes": size(scratch),
        "segment_files": len(seg),
        "segment_bytes": size(seg),
        "nodes": len({f.parent for f in (enc + scratch + seg)}),
    }
