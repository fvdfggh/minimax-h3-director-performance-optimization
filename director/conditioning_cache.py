"""Conditioning cache — save/load CLIP-encoded tensors to avoid repeated encoding."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

from . import cache_layout

log = logging.getLogger("ComfyUI-MiniMaxH3-Director-Cached.conditioning_cache")

# Cache directory relative to ComfyUI output. Encoding caches share the one
# Director cache root with segments and batch scratch, so a workflow's whole
# state lives in a single folder.
CACHE_SUBDIR = cache_layout.CACHE_ROOT

# Cache file prefix. The name carries NO segment index on purpose: two segments
# with identical text inputs must resolve to the same file so the second one is
# served from disk instead of being encoded again. The ``text``/``image``/``video``
# split is what distinguishes the three encoding kinds once they share a folder.
_TEXT_PREFIX = cache_layout.TEXT_PREFIX
_IMAGE_PREFIX = cache_layout.IMAGE_PREFIX
_VIDEO_PREFIX = cache_layout.VIDEO_PREFIX
#: Backwards-compatible alias for callers that glob every encoding cache.
_ENC_PREFIXES = cache_layout.ENC_PREFIXES


def slugify_workflow_name(name: str | None) -> str:
    """Turn a workflow name into a single safe path component.

    Thin re-export of the shared implementation in :mod:`cache_layout`, kept
    because several modules import it from here.
    """
    return cache_layout.slugify_workflow_name(name)


def _get_cache_dir(node_id: str | None = None, workflow_name: str | None = None) -> Path:
    """Get or create the cache directory shared by every artefact kind.

    Layout is ``<cache root>/[<workflow slug>/]node_<id>/``. The workflow layer
    keeps caches from differently named workflows from colliding on the same
    ``node_<id>`` — node ids are per-graph and routinely reused across files.
    """
    return cache_layout.node_cache_dir(node_id, workflow_name)


def _hash_ref_images(ref_images: Any) -> str:
    """Generate a hash for reference images to include in cache key.
    
    This ensures cache invalidation when reference images change.
    """
    if ref_images is None:
        return "none"
    
    try:
        # Handle different ref_images formats
        if isinstance(ref_images, dict):
            # Dict format: {"ref_image_0": tensor, ...}
            hash_parts = []
            for key in sorted(ref_images.keys()):
                tensor = ref_images[key]
                if isinstance(tensor, torch.Tensor):
                    # Hash the tensor data
                    tensor_bytes = tensor.cpu().numpy().tobytes()
                    hash_parts.append(f"{key}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
                else:
                    hash_parts.append(f"{key}:{hash(str(tensor))}")
            return "|".join(hash_parts) if hash_parts else "empty_dict"
        
        elif isinstance(ref_images, (list, tuple)):
            # List/tuple format
            hash_parts = []
            for i, item in enumerate(ref_images):
                if isinstance(item, torch.Tensor):
                    tensor_bytes = item.cpu().numpy().tobytes()
                    hash_parts.append(f"img_{i}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
                elif isinstance(item, dict):
                    # Nested dict
                    for key in sorted(item.keys()):
                        tensor = item[key]
                        if isinstance(tensor, torch.Tensor):
                            tensor_bytes = tensor.cpu().numpy().tobytes()
                            hash_parts.append(f"{key}:{hashlib.md5(tensor_bytes).hexdigest()[:8]}")
            return "|".join(hash_parts) if hash_parts else "empty_list"
        
        else:
            return f"other:{hash(str(ref_images))}"
    
    except Exception as exc:
        log.warning("Failed to hash ref_images: %s", exc)
        return f"error:{hash(str(ref_images))}"


def text_cache_key(
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
) -> str:
    """Public cache key for a text encoding, shared by save/load/dedupe.

    Exposed so the batch path can group segments that would produce byte-identical
    text encodings and encode each distinct key exactly once.
    """
    return _prompt_hash(prompt, width, height, length, task_key, ref_image_size, ref_images)


def _prompt_hash(prompt: str, width: int, height: int, length: int, task_key: str, 
                 ref_image_size: str = "match", ref_images: Any = None) -> str:
    """Generate a hash key for the prompt + dimensions + reference images.
    
    Including ref_images hash ensures cache invalidation when reference images change.
    """
    ref_hash = _hash_ref_images(ref_images)
    content = f"{prompt}|{width}|{height}|{length}|{task_key}|{ref_image_size}|{ref_hash}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_conditioning_cache(
    node_id: str | None,
    segment_index: int,
    positive: Any,
    negative: Any,
    latent: Any,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
    workflow_name: str | None = None,
) -> Path | None:
    """Save conditioning tensors to disk cache.
    
    Returns the cache file path, or None if saving failed.
    """
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        prompt_key = _prompt_hash(prompt, width, height, length, task_key, ref_image_size, ref_images)
        cache_file = cache_dir / f"{_TEXT_PREFIX}_{prompt_key}.pt"
        
        # Prepare conditioning for serialization
        # ComfyUI conditioning is a list of [tensor, dict] pairs
        def prepare_for_save(cond):
            if cond is None:
                return None
            if isinstance(cond, (list, tuple)):
                prepared = []
                for item in cond:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        tensor, config = item[0], item[1]
                        # Ensure tensor is on CPU for storage
                        if hasattr(tensor, 'cpu'):
                            tensor = tensor.cpu()
                        prepared.append([tensor, config])
                    else:
                        prepared.append(item)
                return prepared
            return cond
        
        cache_data = {
            "positive": prepare_for_save(positive),
            "negative": prepare_for_save(negative),
            "latent": prepare_for_save(latent) if latent is not None else None,
            "metadata": {
                "segment_index": segment_index,
                "prompt_hash": prompt_key,
                "width": width,
                "height": height,
                "length": length,
                "task_key": task_key,
                "ref_image_size": ref_image_size,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            }
        }
        
        torch.save(cache_data, cache_file)
        log.info("Conditioning cached: seg #%d → %s", segment_index + 1, cache_file.name)
        return cache_file
        
    except Exception as exc:
        log.warning("Failed to cache conditioning for seg #%d: %s", segment_index + 1, exc)
        return None


def load_conditioning_cache(
    node_id: str | None,
    segment_index: int,
    prompt: str,
    width: int,
    height: int,
    length: int,
    task_key: str,
    ref_image_size: str = "match",
    ref_images: Any = None,
    workflow_name: str | None = None,
) -> dict | None:
    """Load cached conditioning tensors from disk.

    ``segment_index`` is accepted for call-site compatibility but takes no part in
    the lookup: files are named by text hash alone, so any segment whose text
    inputs hash the same resolves to the same file.

    Returns dict with 'positive', 'negative', 'latent' keys, or None if cache miss.
    Tensors are moved to CUDA (GPU) for use by the model.
    """
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        prompt_key = _prompt_hash(prompt, width, height, length, task_key, ref_image_size, ref_images)
        cache_file = cache_dir / f"{_TEXT_PREFIX}_{prompt_key}.pt"

        if not cache_file.exists():
            log.debug("No cache hit for seg #%d (key=%s)", segment_index + 1, prompt_key)
            return None
        
        cache_data = torch.load(cache_file, map_location="cpu", weights_only=False)
        
        # Validate cache metadata
        meta = cache_data.get("metadata", {})
        if (meta.get("width") != width or 
            meta.get("height") != height or 
            meta.get("length") != length or
            meta.get("task_key") != task_key):
            log.info("Cache invalidated for seg #%d (dimensions changed)", segment_index + 1)
            return None
        
        # Move tensors to CUDA (GPU) for model use
        def move_to_device(cond):
            """Recursively move all tensors in conditioning to CUDA."""
            if cond is None:
                return None
            if isinstance(cond, torch.Tensor):
                return cond.cuda()
            if isinstance(cond, (list, tuple)):
                return type(cond)(move_to_device(item) for item in cond)
            if isinstance(cond, dict):
                return {k: move_to_device(v) for k, v in cond.items()}
            return cond
        
        positive = move_to_device(cache_data["positive"])
        negative = move_to_device(cache_data["negative"])
        latent = move_to_device(cache_data["latent"])
        
        log.info("Conditioning cache HIT: seg #%d from %s (moved to CUDA)", segment_index + 1, cache_file.name)
        return {
            "positive": positive,
            "negative": negative,
            "latent": latent,
        }
        
    except Exception as exc:
        log.warning("Failed to load conditioning cache for seg #%d: %s", segment_index + 1, exc)
        return None


def clear_conditioning_cache(
    node_id: str | None = None,
    segment_index: int | None = None,
    workflow_name: str | None = None,
) -> int:
    """Clear conditioning cache files.
    
    Returns number of files deleted.

    ``segment_index`` used to select a filename prefix; files are now named by
    text hash alone, so it no longer selects anything and is ignored. Clearing
    is per node directory.
    """
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        if not cache_dir.exists():
            return 0
        
        deleted = 0
        for f in cache_dir.glob(f"{_TEXT_PREFIX}_*.pt"):
            f.unlink()
            deleted += 1
        
        log.info("Cleared %d conditioning cache files", deleted)
        return deleted
        
    except Exception as exc:
        log.warning("Failed to clear conditioning cache: %s", exc)
        return 0


def get_cache_stats(node_id: str | None = None, workflow_name: str | None = None) -> dict:
    """Get statistics about the conditioning cache."""
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        if not cache_dir.exists():
            return {"exists": False, "files": 0, "total_size_mb": 0}
        
        files = list(cache_dir.glob(f"{_TEXT_PREFIX}_*.pt"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "exists": True,
            "files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(cache_dir),
        }
    except Exception:
        return {"exists": False, "files": 0, "total_size_mb": 0}


def clear_all_conditioning_cache(keep_newer_than: float | None = None) -> int:
    """Clear ALL conditioning cache files across all nodes.

    Returns total number of files deleted.

    ``keep_newer_than`` (epoch seconds) spares anything written at or after that
    instant. The clear now runs *after* a successful run, so without it the run
    would delete the very conditioning files it just produced — emptying the
    cache and forcing a re-encode next time, the opposite of the intent. Pass
    the run's start timestamp to mean "drop stale entries, keep what this run
    just built".
    """
    try:
        from folder_paths import output_directory

        base = Path(output_directory) / CACHE_SUBDIR
        if not base.exists():
            return 0

        deleted = 0
        # rglob, not iterdir: a workflow-name layer may sit between the cache
        # root and node_<id>/, so the files are no longer all one level deep.
        for f in base.rglob(f"{_TEXT_PREFIX}_*.pt"):
            if not f.is_file():
                continue
            if keep_newer_than is not None:
                try:
                    if f.stat().st_mtime >= keep_newer_than:
                        continue
                except OSError:
                    pass
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass

        log.info("Cleared ALL conditioning cache files: %d total", deleted)
        return deleted

    except Exception as exc:
        log.warning("Failed to clear all conditioning cache: %s", exc)
        return 0


# --- Edge-triggered clear ------------------------------------------------
# The「清除缓存」widget is a BOOLEAN (checkbox), not a real button: once ticked
# it stays True, so a naive ``if flag: clear()`` wipes the cache on *every* run
# and every run re-encodes every segment from scratch. Its tooltip promises
# "click 后会在下次运行时自动重置为 False", but the backend cannot do that —
# widget state lives in the browser, and no JS handles this widget.
#
# So remember the last-seen value on disk and fire only on a False -> True
# transition. Ticking it then behaves like pressing a button: one clear, then
# quiet until you untick and tick again.

_CLEAR_STATE_PREFIX = ".clear_button_state"


def _clear_state_path(node_id: str | None, workflow_name: str | None = None) -> Path:
    """Marker file beside the per-node cache dirs — never inside one.

    Living outside means a ``cond_*.pt`` wipe cannot remove the edge marker.
    Namespaced by workflow slug so two workflows sharing a node id do not
    fight over the same marker.
    """
    from folder_paths import output_directory

    base = Path(output_directory) / CACHE_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    slug = slugify_workflow_name(workflow_name)
    tag = f"{slug}." if slug else ""
    return base / f"{_CLEAR_STATE_PREFIX}.{tag}node_{node_id}"


def check_clear_edge(node_id: str | None, flag: bool, workflow_name: str | None = None) -> bool:
    """Read-only edge test: True only on a False -> True transition.

    Deliberately writes nothing. The caller must pair this with
    ``mark_clear_state`` **after** the run succeeds, so a run that raises leaves
    the marker at its old value and the pending clear is retried next time
    instead of having thrown the cache away for nothing.
    """
    path = _clear_state_path(node_id, workflow_name)
    previous = False
    try:
        if path.exists():
            previous = path.read_text(encoding="utf-8").strip().lower() == "true"
    except OSError:
        previous = False

    return bool(flag) and not previous


def mark_clear_state(node_id: str | None, flag: bool, workflow_name: str | None = None) -> None:
    """Record the flag value for the next run's edge test.

    Call only on a fully successful run — see ``check_clear_edge``.
    """
    path = _clear_state_path(node_id, workflow_name)
    try:
        path.write_text("true" if flag else "false", encoding="utf-8")
    except OSError as exc:
        log.warning("Failed to persist clear-button state: %s", exc)


def get_all_cache_stats() -> dict:
    """Get statistics about ALL conditioning cache files across all nodes."""
    try:
        from folder_paths import output_directory
        
        base = Path(output_directory) / CACHE_SUBDIR
        if not base.exists():
            return {"exists": False, "files": 0, "total_size_mb": 0, "nodes": 0}
        
        total_files = 0
        total_size = 0
        node_count = 0
        
        # rglob: workflow-name layer may sit between the root and node_<id>/.
        files = [f for f in base.rglob(f"{_TEXT_PREFIX}_*.pt") if f.is_file()]
        total_files = len(files)
        total_size = sum(f.stat().st_size for f in files)
        node_count = len({f.parent for f in files})
        
        return {
            "exists": True,
            "files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "nodes": node_count,
            "cache_dir": str(base),
        }
    except Exception:
        return {"exists": False, "files": 0, "total_size_mb": 0, "nodes": 0}


def prune_unused_conditioning_cache(
    node_id: str | None,
    workflow_name: str | None,
    keep_keys: set[str],
    keep_newer_than: float | None = None,
) -> int:
    """Delete cached text encodings this run did not use.

    After a run finishes encoding, every ``cond_<hash>.pt`` whose hash is absent
    from ``keep_keys`` belongs to a prompt, resolution or reference set that is no
    longer part of this timeline — leftovers from an edited prompt, or segments
    that were deleted from a longer timeline. Without this they accumulate
    forever, one ~140 MB file per prompt revision.

    ``keep_keys`` must contain every hash the run actually consumed: both the ones
    served from cache and the ones freshly encoded.

    Returns the number of files deleted. Never raises — a cleanup failure must not
    fail an expensive run.
    """
    try:
        cache_dir = _get_cache_dir(node_id, workflow_name)
        if not cache_dir.exists():
            return 0

        keep = set(keep_keys or ())
        prefix_len = len(_TEXT_PREFIX) + 1
        deleted = 0
        for f in cache_dir.glob(f"{_TEXT_PREFIX}_*.pt"):
            if not f.is_file():
                continue
            key = f.stem[prefix_len:]
            if key in keep:
                continue
            if keep_newer_than is not None:
                try:
                    if f.stat().st_mtime >= keep_newer_than:
                        continue
                except OSError:
                    pass
            try:
                f.unlink()
                deleted += 1
            except OSError as exc:
                log.warning("Could not delete stale conditioning cache %s: %s", f.name, exc)

        if deleted:
            log.info(
                "Conditioning cache: pruned %d unused file(s), kept %d", deleted, len(keep)
            )
        return deleted

    except Exception as exc:
        log.warning("Failed to prune conditioning cache: %s", exc)
        return 0
