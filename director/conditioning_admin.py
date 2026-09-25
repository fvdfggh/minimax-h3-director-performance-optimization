"""Clearing, pruning and reporting on the conditioning cache.

Two behaviours matter here:

* the「清空缓存」**edge trigger** — that widget is a BOOLEAN checkbox, not a real
  button, so a naive ``if flag: clear()`` would wipe the cache on *every* run and
  re-encode everything from scratch. :func:`check_clear_edge` compares the flag
  with the last value persisted on disk and fires only on a False -> True
  transition. The marker file lives *beside* the per-node cache dirs
  (:func:`_clear_state_path`), so wiping ``cond_*.pt`` cannot remove it;
* :func:`prune_unused_conditioning_cache` — content-hash keys accumulate one entry
  per distinct render, so prompt edits would grow the cache forever.

Directory names come from :mod:`cache_layout`; the historical aliases kept in
:mod:`conditioning_cache` are re-bound here for the moved code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .cache_layout import CACHE_ROOT as CACHE_SUBDIR
from .cache_layout import TEXT_PREFIX as _TEXT_PREFIX
from .conditioning_keys import _get_cache_dir, slugify_workflow_name
from .conditioning_params import _seg_params_map_path

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.conditioning.admin")


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

        map_path = _seg_params_map_path(node_id, workflow_name)
        try:
            if map_path.is_file():
                map_path.unlink()
        except OSError:
            pass
        
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

        # The two global derived caches live outside the per-node directories, so
        # a workflow's cache wipe misses them unless it covers them too. Both hold
        # only rebuildable model output — losing an entry costs one vision pass or
        # one VAE pass, never correctness.
        try:
            from .ref_latent_cache import clear as _clear_ref_latents

            deleted += _clear_ref_latents(keep_newer_than=keep_newer_than)
        except Exception as exc:  # pragma: no cover - cleanup, never fatal
            log.warning("Failed to clear reference-latent cache: %s", exc)

        try:
            from .vision_cache import clear as _clear_vit

            deleted += _clear_vit(keep_newer_than=keep_newer_than)
        except Exception as exc:  # pragma: no cover - cleanup, never fatal
            log.warning("Failed to clear ViT cache: %s", exc)

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
