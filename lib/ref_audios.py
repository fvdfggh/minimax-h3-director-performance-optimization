"""Standalone reference-audio helpers (MiniMax H3: up to 3 → <Audio 1>…<Audio 3>)."""

from __future__ import annotations

from typing import Any

# Official MiniMaxH3ReferenceToVideo Autogrow max=3.
MAX_REFERENCE_AUDIOS = 3
REF_AUDIO_KEY_PREFIX = "ref_audio_"

#: Sanity bound on a *stored* id, not the model slot count — see ref_videos.py.
_MAX_SLOT_ID = 64


def ref_audios_dict(items: list[tuple[int, dict[str, Any]]]) -> dict[str, dict[str, Any]] | None:
    """Build official ``ref_audio_N`` mapping from (index, AUDIO) pairs.

    Ids are kept as stored; the prepare step renumbers the referenced ones to
    gap-free slots, so an id above :data:`MAX_REFERENCE_AUDIOS` is not dropped.
    """
    out: dict[str, dict[str, Any]] = {}
    for index, audio in items:
        idx = int(index)
        if idx < 0 or idx >= _MAX_SLOT_ID or not isinstance(audio, dict):
            continue
        wave = audio.get("waveform")
        if wave is None:
            continue
        out[f"{REF_AUDIO_KEY_PREFIX}{idx}"] = audio
    return out or None
