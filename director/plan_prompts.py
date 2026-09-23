"""Prompt reinforcement for the r2v / rv2v / v2v task modes.

These rewrite a user's prompt so the model is told what its reference materials
are for: the node rewrites ``<Picture 1>`` references into a sentence naming the
shots, because the MiniMax model responds to described roles more reliably than to
bare tokens.

Pure string functions — no imports, no state — so they are trivially testable.
"""

from __future__ import annotations


def reinforce_r2v_prompt(
    prompt: str,
    *,
    ref_indices: list[int] | None = None,
    video_indices: list[int] | None = None,
    audio_indices: list[int] | None = None,
) -> str:
    """Remind <Picture N> / <Video K> / <Audio J> when tags are missing (r2v batch)."""
    text = (prompt or "").strip() or "Generate a cinematic scene."
    pic_indices = sorted({int(i) for i in (ref_indices or []) if int(i) >= 0})
    vid_indices = sorted({int(i) for i in (video_indices or []) if int(i) >= 0})
    aud_indices = sorted({int(i) for i in (audio_indices or []) if int(i) >= 0})
    prefix_parts: list[str] = []
    if pic_indices and "<Picture" not in text and "<picture" not in text:
        prefix_parts.append(" ".join(f"<Picture {i + 1}>" for i in pic_indices))
    if vid_indices and "<Video" not in text and "<video" not in text:
        prefix_parts.append(" ".join(f"<Video {i + 1}>" for i in vid_indices))
    if aud_indices and "<Audio" not in text and "<audio" not in text:
        prefix_parts.append(" ".join(f"<Audio {i + 1}>" for i in aud_indices))
    if not prefix_parts:
        return text
    return f"{' '.join(prefix_parts)} {text}"


def reinforce_v2v_prompt(prompt: str) -> str:
    """Ensure MiniMax ReferenceToVideo sees an explicit <Video 1> tag for source edit."""
    text = (prompt or "").strip()
    if not text:
        return "Edit <Video 1>."
    if "<Video" in text or "<video" in text:
        return text
    return f"<Video 1> {text}"


def reinforce_rv2v_prompt(
    prompt: str,
    *,
    ref_indices: list[int] | None = None,
    audio_indices: list[int] | None = None,
) -> str:
    """Source <Video 1> + remind <Picture N> / <Audio J> when tags are missing."""
    text = reinforce_v2v_prompt(prompt)
    pic_indices = sorted({int(i) for i in (ref_indices or []) if int(i) >= 0})
    aud_indices = sorted({int(i) for i in (audio_indices or []) if int(i) >= 0})
    prefix_parts: list[str] = []
    if pic_indices and "<Picture" not in text and "<picture" not in text:
        prefix_parts.append(" ".join(f"<Picture {i + 1}>" for i in pic_indices))
    if aud_indices and "<Audio" not in text and "<audio" not in text:
        prefix_parts.append(" ".join(f"<Audio {i + 1}>" for i in aud_indices))
    if not prefix_parts:
        return text
    return f"{' '.join(prefix_parts)} {text}"
