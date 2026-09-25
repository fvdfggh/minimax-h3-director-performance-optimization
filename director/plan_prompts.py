"""Prompt reinforcement for the r2v / rv2v / v2v task modes.

The only tag these helpers still add is the one a mode *structurally* needs: the
source ``<Video 1>`` of v2v / rv2v. Reference pictures / videos / audios are
never auto-tagged — which materials are actually sent is decided solely by the
tags the user wrote (see :func:`director.batch_prepare._filter_refs_by_prompt`),
so manufacturing tags here would pull un-referenced material back into the run.

Pure string functions — no imports, no state — so they are trivially testable.
"""

from __future__ import annotations


def reinforce_r2v_prompt(prompt: str) -> str:
    """Guarantee a non-empty prompt for an r2v batch segment.

    No ``<Picture N>`` / ``<Video K>`` / ``<Audio J>`` is added: a material the
    prompt does not name is one the user did not ask for, and the reference
    filter drops it.
    """
    return (prompt or "").strip() or "Generate a cinematic scene."


def reinforce_v2v_prompt(prompt: str) -> str:
    """Ensure MiniMax ReferenceToVideo sees an explicit <Video 1> tag for source edit."""
    text = (prompt or "").strip()
    if not text:
        return "Edit <Video 1>."
    if "<Video" in text or "<video" in text:
        return text
    return f"<Video 1> {text}"


def reinforce_rv2v_prompt(prompt: str) -> str:
    """Source <Video 1> only; pictures / audio stay exactly as the user wrote them."""
    return reinforce_v2v_prompt(prompt)
