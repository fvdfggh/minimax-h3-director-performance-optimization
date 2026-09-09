"""ComfyUI MiniMax H3 Director Opt — timeline plugin for MiniMax-H3 AV generation.

Fork of AIMixer/ComfyUI_MiniMaxH3_Director (base: ComfyUI official MiniMax H3
support, PR #15224 / #15228). Every node type id, display name, HTTP route,
frontend event and cache root carries the "Opt" marker so this pack can be
installed side by side with the original one without any collision.
Licensed under the Apache License, Version 2.0. See LICENSE.
"""

from .nodes.conditioning import (
    MiniMaxH3DirectorOptConditioning,
    MiniMaxH3DirectorOptPlannerConditioning,
)
from .nodes.director import MiniMaxH3DirectorOpt
from .nodes.director_groups import (
    MiniMaxH3DirectorOptGroupImageToVideo,
    MiniMaxH3DirectorOptGroupReferenceToVideo,
    MiniMaxH3DirectorOptGroupsCombine,
)
from .nodes.fast_video_vae_decode import MiniMaxH3FastVideoVAEOpt
from .nodes.latent_upscaler_3d import MiniMaxH3LatentUpscaleModelOptNode

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DirectorOpt": MiniMaxH3DirectorOpt,
    # Legacy type id kept so older workflows still load.
    "ComfyMiniMaxH3DirectorOpt": MiniMaxH3DirectorOpt,
    "MiniMaxH3DirectorOptConditioning": MiniMaxH3DirectorOptConditioning,
    "MiniMaxH3DirectorOptPlannerConditioning": MiniMaxH3DirectorOptPlannerConditioning,
    "MiniMaxH3DirectorOptGroupImageToVideo": MiniMaxH3DirectorOptGroupImageToVideo,
    "MiniMaxH3DirectorOptGroupReferenceToVideo": MiniMaxH3DirectorOptGroupReferenceToVideo,
    # Must stay in NODE_CLASS_MAPPINGS: ComfyUI skips comfy_entrypoint when
    # NODE_CLASS_MAPPINGS is present (if/elif in load_custom_node).
    "MiniMaxH3DirectorOptGroupsCombine": MiniMaxH3DirectorOptGroupsCombine,
    "MiniMaxH3FastVideoVAEOpt": MiniMaxH3FastVideoVAEOpt,
    "MiniMaxH3LatentUpscaleModelOpt": MiniMaxH3LatentUpscaleModelOptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectorOpt": "MiniMaxH3Director Opt",
    # Legacy alias kept so workflows saved with the old type id still load.
    "ComfyMiniMaxH3DirectorOpt": "MiniMaxH3Director Opt",
    "MiniMaxH3DirectorOptConditioning": "MiniMax H3 Director Opt Conditioning",
    "MiniMaxH3DirectorOptPlannerConditioning": "MiniMax H3 Director Opt Planner Conditioning",
    "MiniMaxH3DirectorOptGroupImageToVideo": "MiniMax H3 Director Opt Group (Image to Video)",
    "MiniMaxH3DirectorOptGroupReferenceToVideo": "MiniMax H3 Director Opt Group (Reference to Video)",
    "MiniMaxH3DirectorOptGroupsCombine": "MiniMax H3 Director Opt Groups Combine",
    "MiniMaxH3FastVideoVAEOpt": "MiniMax H3 Fast Video VAE Opt",
    "MiniMaxH3LatentUpscaleModelOpt": "Minimax H3 Latent Upscaler Opt (3D) [Model]",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-MiniMaxH3-Director-Opt")

# The latent upscale model node is a V3 node (io.ComfyNode) — required for its
# DynamicCombo (mode) to show/hide sub-parameters. ComfyUI auto-discovers V3
# nodes through comfy_entrypoint, which it skips for packages that expose
# NODE_CLASS_MAPPINGS, so it is registered manually here (same as nodes.py does).
if MiniMaxH3LatentUpscaleModelOptNode is None:
    NODE_CLASS_MAPPINGS.pop("MiniMaxH3LatentUpscaleModelOpt", None)
    NODE_DISPLAY_NAME_MAPPINGS.pop("MiniMaxH3LatentUpscaleModelOpt", None)
    _log.warning("MiniMax H3 latent upscale model node unavailable (comfy_api.latest missing).")
else:
    try:
        _upscale_schema = MiniMaxH3LatentUpscaleModelOptNode.GET_SCHEMA()
        NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3LatentUpscaleModelOpt"] = _upscale_schema.display_name
    except Exception as _upscale_exc:
        _log.warning("MiniMax H3 latent upscale model node failed to load: %s", _upscale_exc)
        NODE_CLASS_MAPPINGS.pop("MiniMaxH3LatentUpscaleModelOpt", None)
        NODE_DISPLAY_NAME_MAPPINGS.pop("MiniMaxH3LatentUpscaleModelOpt", None)

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "MiniMax H3 Director Opt HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /minimax/director_opt/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("MiniMax H3 Director Opt HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
