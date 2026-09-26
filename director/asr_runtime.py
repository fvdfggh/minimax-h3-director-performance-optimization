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

Lookup is deliberately forgiving. ``workflow_name`` is a *cache-layout* key: the
editor mints a fresh one when a workflow saved before the id existed is opened, so
a handle remembered under the old id would strand the button even though the very
same socket is still wired. A node only ever has one ASR model, so a miss on the
exact pair falls back to the node's latest handle, and — in a single-node graph —
to the only handle there is.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.asr.runtime")

_MAX_KEYS = 32

_handles: dict[tuple[str, str], Any] = {}
#: ``node_id -> key`` of that node's most recent handle (drift-tolerant lookup).
_by_node: dict[str, tuple[str, str]] = {}
#: Nodes whose *last* run delivered no handle (socket empty / loader bypassed).
#: Kept separately from :data:`_handles` so the button can tell「本次会话还没跑过」
#: apart from「跑过了，但模型口没交付模型」—— two very different fixes.
_ran_without_model: set[str] = set()
#: Nodes already told that no handle arrived (one INFO line per node, not per run).
_reported_missing: set[str] = set()


def _key(node_id, workflow_name) -> tuple[str, str]:
    return (str(node_id or ""), str(workflow_name or ""))


def remember_asr_model(node_id, workflow_name, handle) -> None:
    """Park ``handle`` for later button-triggered checks."""
    node = str(node_id or "")
    if handle is None:
        if node:
            _ran_without_model.add(node)
        # Worth exactly one INFO line per node: "I wired it, why is the button still
        # asking me to run?" is otherwise invisible — the socket may not be connected
        # at all, or the loader returned nothing.
        if node and node not in _reported_missing:
            _reported_missing.add(node)
            log.info(
                "音频有效性校验: 本次运行没有拿到 ASR 模型（node %s：asr_model 口未接线，"
                "或加载器返回空），校验按钮会提示先运行一次。",
                node,
            )
        return
    _ran_without_model.discard(node)
    key = _key(node_id, workflow_name)
    if key not in _handles and len(_handles) >= _MAX_KEYS:
        # Every entry is a model another node already owns; drop the oldest rather
        # than growing without bound when many Director nodes share one workflow.
        oldest = next(iter(_handles))
        _handles.pop(oldest, None)
        _by_node.pop(oldest[0], None)
    _handles[key] = handle
    _by_node[key[0]] = key
    log.debug("音频有效性校验: 已记住 ASR 模型 (node %s)", key[0])


def asr_model_for(node_id, workflow_name) -> Any | None:
    """Handle to use for this node, or ``None`` when nothing was ever remembered."""
    node = str(node_id or "")

    exact = _handles.get(_key(node_id, workflow_name))
    if exact is not None:
        return exact

    # Same node, another workflow id: the editor re-mints the id when a workflow
    # saved before the id existed is opened. Still the same socket, same model.
    key = _by_node.get(node)
    if key is not None:
        handle = _handles.get(key)
        if handle is not None:
            log.debug(
                "音频有效性校验: node %s 的模型登记在另一个 workflow（%s），按节点回退使用。",
                node, key[1],
            )
            return handle

    # Last resort, and only when unambiguous: one registered handle in the graph
    # means the user has exactly one ASR model — the picker's node id may simply not
    # be the execution id the backend saw (subgraph / re-imported node).
    if node not in _by_node and len(_handles) == 1:
        only_key = next(iter(_handles))
        log.warning(
            "音频有效性校验: node %s 没有登记模型，回退到唯一已登记的 node %s。",
            node or "(空)", only_key[0],
        )
        return _handles[only_key]

    return None


def remembered_asr_nodes() -> list[str]:
    """Node ids that currently hold a handle (used in the button's error text)."""
    return sorted(_by_node)


#: Sentinel key for the handle we build ourselves (never a real node id, so it is
#: kept out of the ``_by_node`` index and out of the dialog's node list).
_AUTO_KEY = ("<auto>", "")
_autoload_lock = threading.Lock()
_autoload_error = ""


def autoload_asr_model() -> Any | None:
    """Build the pack's handle with its defaults, once per process.

    The fallback for「没接线 / 重启后还没跑过 / 模型接在别的节点」: the handle is
    only a config struct (:func:`asr_check.default_model_handle`), so the check no
    longer needs a run. Cached because building it hashes the model files; guarded
    by a lock because two clicks can race and the call is made from a worker
    thread. Returns ``None`` (with :func:`autoload_error`) when it cannot be built.
    """
    global _autoload_error
    cached = _handles.get(_AUTO_KEY)
    if cached is not None:
        return cached
    with _autoload_lock:
        cached = _handles.get(_AUTO_KEY)
        if cached is not None:
            return cached
        try:
            from .asr_check import default_model_handle

            handle = default_model_handle()
        except Exception as exc:
            _autoload_error = str(exc) or type(exc).__name__
            log.info("音频有效性校验: 自动构造 ASR 模型句柄失败 (%s)", exc)
            return None
        _autoload_error = ""
        _handles[_AUTO_KEY] = handle
        return handle


def autoload_error() -> str:
    """Why :func:`autoload_asr_model` failed last time (``""`` when it worked)."""
    return _autoload_error


def asr_node_state(node_id) -> str:
    """Why the button has (or has not) got a handle — drives the dialog's hint.

    ``ready``        a handle is registered (possibly via a fallback key);
    ``ran-without``  the node *did* run, but no model came through the socket;
    ``never-ran``    the node has not executed since this server started —
                     which is what a ComfyUI restart looks like.
    """
    node = str(node_id or "")
    if not node:
        return "never-ran"
    if _by_node.get(node):
        return "ready"
    if node in _ran_without_model:
        return "ran-without"
    return "never-ran"


def forget_asr_model(node_id, workflow_name) -> None:
    """Drop a remembered handle (cache clears / node removal)."""
    key = _key(node_id, workflow_name)
    _handles.pop(key, None)
    if _by_node.get(key[0]) == key:
        _by_node.pop(key[0], None)
