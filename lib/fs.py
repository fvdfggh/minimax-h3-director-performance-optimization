"""Atomic file publishing shared by the cache and export layers.

The "write a unique sibling temp, then ``os.replace``" dance (plus a retry for
mounts that refuse to overwrite an existing name) used to be implemented five
times with subtly different failure handling. Every cache artefact and export
goes through here so the publish semantics stay identical.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable


def safe_unlink(path: str | os.PathLike[str]) -> bool:
    """Best-effort delete of a file/symlink; ``False`` instead of raising."""
    try:
        target = Path(path)
        if target.is_file() or target.is_symlink():
            target.unlink()
        return True
    except OSError:
        return False


def temp_path(dest: Path) -> Path:
    """A unique sibling of ``dest``, so publishing stays on the same volume."""
    return dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")


def atomic_publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` onto ``dest``, tolerating mounts that block same-name overwrite."""
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    # Some cloud mounts reject overwrite of an existing name — remove then rename.
    safe_unlink(dest)
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    tmp.rename(dest)  # last resort: raises if even this is denied


def write_via_temp(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a unique temp name in the same folder, then publish to ``dest``."""
    tmp = temp_path(dest)
    try:
        write_fn(tmp)
        atomic_publish(tmp, dest)
    finally:
        safe_unlink(tmp)


def write_text_atomic(dest: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Create ``dest``'s folder if needed, then write ``text`` atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_via_temp(dest, lambda tmp: tmp.write_text(text, encoding=encoding))


def write_json_atomic(dest: Path, payload: Any) -> None:
    """Atomically write ``payload`` as sorted, non-ASCII-preserving JSON."""
    write_text_atomic(dest, json.dumps(payload, ensure_ascii=False, sort_keys=True))
