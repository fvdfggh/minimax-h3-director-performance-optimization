"""Bridge the「音频有效性校验」button to the ASR model handle.

The check is triggered from the node's UI, i.e. from an HTTP request — which has
no access to the graph, so it cannot receive the ``asr_model`` input socket. The
node therefore parks the handle here on every run, keyed by the same
``(node_id, workflow_name)`` pair every other cache in this pack uses, and
:mod:`routes_asr` borrows it back.

This module never loads anything: it only remembers what the loader node already
handed to the graph, and the handle stays alive exactly as long as the loader's
own cache keeps it. A handle from a previous run is what the button needs — you
check audio you already generated.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.asr.runtime")

_MAX_KEYS = 32

_handles: dict[tuple[str, str], Any] = {}


def _key(node_id, workflow_name) -> tuple[str, str]:
    return (str(node_id or ""), str(workflow_name or ""))


def remember_asr_model(node_id, workflow_name, handle) -> None:
    """Park ``handle`` for later button-triggered checks."""
    if handle is None:
        return
    key = _key(node_id, workflow_name)
    if key not in _handles and len(_handles) >= _MAX_KEYS:
        # Every entry is a model another node already owns; drop the oldest rather
        # than growing without bound when many Director nodes share one workflow.
        _handles.pop(next(iter(_handles)), None)
    _handles[key] = handle
    log.debug("音频有效性校验: 已记住 ASR 模型 (node %s)", key[0])


def asr_model_for(node_id, workflow_name) -> Any | None:
    """Handle remembered for this node, or ``None``."""
    return _handles.get(_key(node_id, workflow_name))


def forget_asr_model(node_id, workflow_name) -> None:
    """Drop a remembered handle (cache clears / node removal)."""
    _handles.pop(_key(node_id, workflow_name), None)
