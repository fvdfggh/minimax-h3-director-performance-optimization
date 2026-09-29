"""Reference-image slot helpers (MiniMax H3: up to 9 refs → <Picture 1>…<Picture 9>)."""

from __future__ import annotations

import torch

# Official MiniMaxH3ReferenceToVideo Autogrow max=9.
MAX_REFERENCE_IMAGES = 9
REF_IMAGE_KEY_PREFIX = "reference_image_"


def reference_image_label(index: int) -> str:
    """User-facing label for slot index (0-based) → 图片1…图片9."""
    return f"图片{int(index) + 1}"


def flatten_reference_kwargs(kwargs: dict) -> dict[str, torch.Tensor | None]:
    """Collect the ``reference_image_<N>`` slots of a kwargs mapping.

    The limit enforced here is the *count* the official node accepts, not the id:
    stored ids are renumbered to gap-free slots before the model call, so an id
    above the UI slot count is usable as long as the payload does not carry more
    than :data:`MAX_REFERENCE_IMAGES` of them.
    """
    refs: dict[str, torch.Tensor | None] = {}
    for key, value in kwargs.items():
        if not key.startswith(REF_IMAGE_KEY_PREFIX):
            continue
        index = int(key.removeprefix(REF_IMAGE_KEY_PREFIX))
        if index < 0:
            raise ValueError(f"Invalid reference image slot {key!r}; expected a non-negative id.")
        refs[key] = value
    if len(refs) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Too many reference images: {len(refs)} payload slot(s), official limit is "
            f"{MAX_REFERENCE_IMAGES}. Reference fewer materials or remove the extra slots."
        )
    return refs


def _slot_number(key: str) -> int:
    return int(key.removeprefix(REF_IMAGE_KEY_PREFIX))


def sorted_reference_items(
    extra_refs: dict[str, torch.Tensor | None],
) -> list[tuple[str, torch.Tensor]]:
    """Return connected reference images in slot order (图片1…图片9)."""
    items: list[tuple[str, torch.Tensor]] = []
    for key, value in extra_refs.items():
        if value is not None and value.shape[0] > 0:
            items.append((key, value))
    return sorted(items, key=lambda kv: _slot_number(kv[0]))

