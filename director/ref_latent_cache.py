"""Persisted VAE output cache for MiniMax H3 reference media.

Why this exists
---------------
Reference images / videos / soundtracks are encoded by the video and audio VAEs
into the ``minimax_refs`` latent blocks the DiT re-injects at every sampling
step. That pass is expensive — a reference clip is a whole video through the 3D
VAE — and it depends on *nothing but the canvas-fitted pixels and the VAE*: not
the prompt, not the workflow, not the segment.

``conditioning_store`` keeps the assembled conditioning (text **and** those
latents) under a key that mixes in the prompt *and* the canvas, and
``prune_unused_conditioning_cache`` deletes the file as soon as a run stops
using its key — which a canvas switch does on the very next run. Switching 768p
-> 1080p -> back therefore throws away the first encode and re-runs every
reference VAE, even though the pixels feeding them are byte-identical to the
ones already encoded.

Caching the latents here, addressed by **source media + canvas**, means the
reference media is encoded once per (media, canvas) pair and reused by every
prompt, every workflow and every later run that reaches it — so switching the
canvas back is free, and editing a prompt only costs the text encoder.

Addressing by source rather than by the fitted pixels is what also frees the text
encoding from the canvas: an encoding can name the media and leave the resolution
to whoever loads it, so one ``cond_text`` file serves 768p and 1080p alike, each
resolving to its own latents.

Why caching it is safe
----------------------
``vae.encode`` is a pure function of the pixels (the VAE weights are fixed for a
run), and the latents are consumed read-only downstream — the DiT concatenates
them — so handing out a cached tensor is bit-equivalent to re-encoding.

The key is a content hash, never ``id()``: ``id()`` misses every re-loaded
duplicate and can collide after GC reuse. The hash covers the *canvas-fitted*
pixels, so two canvases naturally land on two entries and neither can serve the
other.

Layout
------
``output/minimax_director_opt_cache/_reflat/reflat_<hash>.pt`` — global, like
:mod:`vision_cache`, because the output depends only on the media and the VAE.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import torch

from ..lib.fs import write_via_temp
from . import cache_layout

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.ref_latent_cache")

#: Set ``MINIMAX_REF_LATENT_CACHE=0`` to encode every reference afresh.
_ENABLED_ENV = "MINIMAX_REF_LATENT_CACHE"

#: Which VAE produced the entry. The two are separate models, so an audio
#: latent must never be served for a video job even if the bytes matched.
VIDEO = "video"
AUDIO = "audio"

#: How the same source media was turned into pixels. Entries are keyed by source
#: media + canvas, so anything else that changes the encode has to be named too
#: — otherwise a first frame (stretched) and a last frame (centre-cropped) of one
#: image would collide on the same address.
VARIANT_REF_IMAGE = "ref_image"
VARIANT_REF_VIDEO = "ref_video"
VARIANT_FIRST_FRAME = "first_frame"
VARIANT_LAST_FRAME = "last_frame"
VARIANT_REF_AUDIO = "ref_audio"

_puts = [0]


def enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _cache_dir() -> Path | None:
    try:
        import folder_paths

        out = folder_paths.get_output_directory()
    except Exception:
        return None
    if not out:
        return None
    return Path(out) / cache_layout.CACHE_ROOT / cache_layout.REF_LATENT_CACHE_DIRNAME


#: A reference clip is hashed once per run, not once per segment: in global edit
#: mode every segment hands over the *same* tensor, and re-hashing a whole video
#: once per segment would dwarf the lookup it is meant to speed up.
#:
#: Keyed by ``id()`` with the tensor pinned alongside it — ``id()`` alone would
#: be recycled onto another tensor after a GC. (A ``WeakKeyDictionary`` cannot be
#: used: comparing tensor keys evaluates them as booleans, which raises for
#: anything with more than one element.) Bounded, so long sessions do not pin
#: every clip a workflow ever loaded — an eviction only costs one re-hash.
_DIGEST_MEMO: dict[int, tuple[Any, str]] = {}
_DIGEST_MEMO_MAX = 32


def _digest(data: torch.Tensor) -> str:
    """Full hash of the *source* media.

    Deliberately not sampled: a collision would bind the wrong latent to a
    reference. Hashing a clip costs far less than the VAE pass it stands in for,
    and the memo keeps it to once per tensor per run.
    """
    ident = id(data)
    hit = _DIGEST_MEMO.get(ident)
    if hit is not None and hit[0] is data:
        return hit[1]
    t = data.detach().to("cpu")
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    out = hashlib.blake2b(t.numpy().tobytes(), digest_size=16).hexdigest()
    if len(_DIGEST_MEMO) >= _DIGEST_MEMO_MAX:
        _DIGEST_MEMO.pop(next(iter(_DIGEST_MEMO)), None)
    _DIGEST_MEMO[ident] = (data, out)
    return out


def address_of_seed(
    kind: str,
    seed: str,
    canvas: tuple[int, int] = (0, 0),
    *,
    variant: str = "",
    ref_image_size: str = "",
    frames: int | None = None,
) -> str:
    """Address of one VAE encode, from an already-hashed source.

    Built from the *original* media plus the parameters that shape it — never
    from the canvas-fitted pixels. Two reasons that matters:

    * the address can be derived without doing the resize, so a cache hit costs
      no tensor work at all;
    * a cached text encoding can name the media and let whoever loads it supply
      **their own** canvas, which is what lets one encoding serve several
      resolutions. The same clip at two canvases still lands on two entries, and
      neither can serve the other.
    """
    payload = "|".join(
        (
            str(kind),
            str(variant),
            f"{int(canvas[0])}x{int(canvas[1])}",
            str(ref_image_size or ""),
            "" if frames is None else str(int(frames)),
            str(seed),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def seed(raw: torch.Tensor) -> str:
    """Stable hash of a piece of source media — safe to persist.

    This is the half of the address a cached text encoding keeps, so it has to
    survive the media itself being dropped from memory.
    """
    return _digest(raw)


def address(
    kind: str,
    raw: torch.Tensor,
    canvas: tuple[int, int] = (0, 0),
    *,
    variant: str = "",
    ref_image_size: str = "",
    frames: int | None = None,
) -> str:
    """Address of the VAE encode of ``raw`` media at ``canvas``."""
    return address_of_seed(
        kind, _digest(raw), canvas,
        variant=variant, ref_image_size=ref_image_size, frames=frames,
    )


def _path_of(key: str) -> Path | None:
    d = _cache_dir()
    if d is None:
        return None
    return d / f"{cache_layout.REF_LATENT_PREFIX}_{key}{cache_layout.REF_LATENT_SUFFIX}"


def load_by_key(key: str) -> dict | None:
    """Load one entry by its address.

    Used when re-attaching latents to a cached text encoding: the encoding only
    remembers how to *derive* the address, so this is the way back to the tensor.
    """
    if not key:
        return None
    try:
        path = _path_of(key)
        if path is None or not path.is_file():
            return None
        entry = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log.debug("Reference-latent cache read skipped: %s", exc)
        return None
    if not isinstance(entry, dict) or not torch.is_tensor(entry.get("latent")):
        return None
    try:
        os.utime(path, None)  # LRU touch
    except Exception:
        pass
    return entry


def store_by_key(key: str, latent: torch.Tensor, **extra) -> bool:
    """Persist one VAE output under an already-computed address."""
    if not enabled() or not key:
        return False
    try:
        path = _path_of(key)
        if path is None:
            return False
        payload = {"latent": latent.detach().to("cpu"), **extra}
        payload["meta"] = {"ts": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_via_temp(path, lambda tmp: torch.save(payload, tmp))
    except Exception as exc:
        log.debug("Reference-latent cache write skipped: %s", exc)
        return False
    if _puts[0] >= 10:
        _puts[0] = 0
        prune()
    else:
        _puts[0] += 1
    return True


# --- swapping latents out of / back into a conditioning payload ---------------

#: Conditioning groups that carry VAE output, and the fields within them.
_LATENT_GROUPS = ("minimax_refs", "minimax_keyframes")
#: ``latent`` / ``audio_latent`` -> the field holding that tensor's *reference*
#: (source hash + variant), stamped on by :func:`encode_video_vae_batch` /
#: :func:`encode_audio_vae_batch`.
_LATENT_FIELDS = (("latent", "_reflat"), ("audio_latent", "_reflat_audio"))


def _rows(positive: Any) -> list[list]:
    return [
        r for r in (positive or [])
        if isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[1], dict)
    ]


def detach_latents(positive: Any) -> Any:
    """Swap every reference latent for its address, keeping the file text-sized.

    The conditioning cache writes one file per text key, and several prompts
    routinely reference the same clip, so without this every one of those files
    would carry its own copy of the same latent — the video one is by far the
    largest thing in the cache. Replacing the tensor with a *reference* (source
    hash + variant) leaves the file holding only the text encoding.

    The reference deliberately omits the canvas: the reader supplies its own, so
    the same file resolves to the 768p latent or the 1080p one depending on who
    loads it. That is what frees the text encoding from the canvas.

    A latent that was never cached (cache disabled, or a failed write) keeps its
    tensor: a reference that cannot be resolved later must never be written.
    """
    if not isinstance(positive, list):
        return positive
    out = []
    for row in positive:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], dict)):
            out.append(row)
            continue
        cfg = dict(row[1])
        for group in _LATENT_GROUPS:
            items = cfg.get(group)
            if not isinstance(items, (list, tuple)):
                continue
            swapped = []
            for blk in items:
                if not isinstance(blk, dict):
                    swapped.append(blk)
                    continue
                nb = dict(blk)
                for field, ref_field in _LATENT_FIELDS:
                    if not torch.is_tensor(nb.get(field)):
                        continue
                    ref = nb.pop(ref_field, None)
                    if isinstance(ref, dict) and ref.get("seed"):
                        nb[field] = dict(ref)
                swapped.append(nb)
            cfg[group] = swapped
        out.append([row[0], cfg])
    return out


def attach_latents(
    positive: Any,
    canvas: tuple[int, int] = (0, 0),
    ref_image_size: str = "",
) -> Any | None:
    """Restore the latents :func:`detach_latents` swapped out.

    ``canvas`` is the **reader's** canvas, not the one the encoding was written
    at: a reference names the source media, and this supplies the resolution to
    resolve it at. That is the whole point of keying entries by source + canvas.

    Returns ``None`` when any latent is missing — LRU trimming or「清空缓存」can
    drop an entry while its text encoding survives, and a segment first seen at a
    resolution whose latents were never encoded — so the caller treats the whole
    encoding as a miss and re-encodes. Sampling on an encoding with its
    references quietly missing would look like a working run that ignores them.
    """
    if positive is None:
        return None
    if not isinstance(positive, list):
        return positive
    out = []
    for row in positive:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], dict)):
            out.append(row)
            continue
        cfg = dict(row[1])
        for group in _LATENT_GROUPS:
            items = cfg.get(group)
            if not isinstance(items, (list, tuple)):
                continue
            restored = []
            for blk in items:
                if not isinstance(blk, dict):
                    restored.append(blk)
                    continue
                nb = dict(blk)
                for field, _ref_field in _LATENT_FIELDS:
                    ref = nb.get(field)
                    if not isinstance(ref, dict) or not ref.get("seed"):
                        continue
                    addr = address_of_seed(
                        ref.get("kind") or VIDEO, ref["seed"], canvas,
                        variant=ref.get("variant", ""),
                        ref_image_size=ref_image_size,
                        frames=ref.get("frames"),
                    )
                    entry = load_by_key(addr)
                    if entry is None:
                        log.info(
                            "Conditioning cache: %s latent not encoded at %dx%d — "
                            "re-encoding.", ref.get("variant") or "reference",
                            int(canvas[0]), int(canvas[1]),
                        )
                        return None
                    nb[field] = entry["latent"]
                restored.append(nb)
            cfg[group] = restored
        out.append([row[0], cfg])
    return out


def strip_latents(positive: Any) -> Any:
    """Drop the reference blocks, leaving the text encoding proper.

    Used when a cached encoding's latents are missing at the reader's canvas: the
    text half is still perfectly valid, and the caller is about to re-derive the
    blocks anyway, so handing back the encoding without them lets the run skip the
    Qwen prefill and pay only for the VAEs.
    """
    if not isinstance(positive, list):
        return positive
    out = []
    for row in positive:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], dict)):
            out.append(row)
            continue
        out.append([row[0], {k: v for k, v in row[1].items() if k not in _LATENT_GROUPS}])
    return out


def clear(keep_newer_than: float | None = None) -> int:
    """Drop every persisted reference latent. Returns the number deleted.

    Only rebuildable derived data lives here, so clearing is always safe — the
    worst case is one extra VAE pass. ``keep_newer_than`` spares entries written
    at or after that instant, which is how「清空缓存」avoids deleting what the run
    that triggered it just produced.
    """
    d = _cache_dir()
    if d is None or not d.is_dir():
        return 0
    deleted = 0
    try:
        files = [
            p for p in d.glob(f"{cache_layout.REF_LATENT_PREFIX}_*{cache_layout.REF_LATENT_SUFFIX}")
            if p.is_file()
        ]
    except Exception:
        return 0
    for p in files:
        if keep_newer_than is not None:
            try:
                if p.stat().st_mtime >= keep_newer_than:
                    continue
            except OSError:
                pass
        try:
            p.unlink()
            deleted += 1
        except OSError as exc:
            log.debug("Could not delete reference latent %s: %s", p.name, exc)
    if deleted:
        log.info("Reference-latent cache cleared: %d file(s)", deleted)
    _puts[0] = 0
    return deleted


def prune(max_bytes: int | None = None, cache_dir: Path | None = None) -> int:
    """Drop oldest entries until the cache fits ``max_bytes``. Returns bytes freed."""
    d = cache_dir or _cache_dir()
    if d is None or not d.is_dir():
        return 0
    limit = int(max_bytes if max_bytes is not None else cache_layout.REF_LATENT_MAX_BYTES)
    try:
        files = [
            (p, p.stat().st_size)
            for p in d.glob(f"{cache_layout.REF_LATENT_PREFIX}_*{cache_layout.REF_LATENT_SUFFIX}")
            if p.is_file()
        ]
    except Exception:
        return 0
    total = sum(sz for _, sz in files)
    if total <= limit:
        return 0
    files.sort(key=lambda it: it[0].stat().st_mtime)
    freed = 0
    for p, sz in files:
        if total <= limit:
            break
        try:
            p.unlink()
            total -= sz
            freed += sz
        except Exception:
            continue
    if freed:
        log.info("Reference-latent cache trimmed by %s (limit %s)", _human(freed), _human(limit))
    return freed


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}GB"


def stats() -> dict:
    d = _cache_dir()
    count = 0
    total = 0
    if d is not None and d.is_dir():
        try:
            for p in d.glob(f"{cache_layout.REF_LATENT_PREFIX}_*{cache_layout.REF_LATENT_SUFFIX}"):
                if p.is_file():
                    count += 1
                    total += p.stat().st_size
        except Exception:
            pass
    return {"dir": str(d) if d else None, "entries": count, "bytes": total, "human": _human(total)}
