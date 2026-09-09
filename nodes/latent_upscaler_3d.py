"""MiniMax H3 latent upscale **model** node.

Ports the ``Minimax H3 Latent Upscaler (3D)`` node into a node that emits a
``LATENT_UPSCALE_MODEL`` object instead of an upscaled latent. Every parameter
(name, range, default, tooltip) is unchanged — including the ``mode``
DynamicCombo, so the sub-parameters (``scale`` / ``width``+``height`` /
``megapixels``) show and hide automatically as the mode changes.

The node is written with the V3 (``comfy_api.latest``) node API because
dynamic show/hide is only supported there; it is registered manually in the
package's ``NODE_CLASS_MAPPINGS`` (ComfyUI skips ``comfy_entrypoint`` for
packages that expose ``NODE_CLASS_MAPPINGS``).

The returned object is a real model — pass it around and call it from
anywhere:

````python
upscale_model = ...                                        # this node's output
out = upscale_model(latent)                                # settings from the node
out = upscale_model(latent, mode="scale by multiplier", scale=2.0)
````
"""

from __future__ import annotations

from enum import Enum

from ..lib.latent_upscaler_3d import (
    MODE_MEGAPIXELS,
    MODE_SCALE,
    MODE_TARGET_DIMENSIONS,
    MiniMaxH3LatentUpscaleModelOpt,
    scan_models,
)

try:
    from comfy_api.latest import io

    _HAS_V3_API = True
except Exception:  # pragma: no cover - very old ComfyUI
    io = None
    _HAS_V3_API = False


class UpscaleMode(str, Enum):
    SCALE_BY = MODE_SCALE
    TARGET_DIMENSIONS = MODE_TARGET_DIMENSIONS
    MEGAPIXELS = MODE_MEGAPIXELS


if _HAS_V3_API:

    class MiniMaxH3LatentUpscaleModelOptNode(io.ComfyNode):
        """Minimax H3 latent upscaler (3D), packaged as a reusable model object."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3LatentUpscaleModelOpt",
                display_name="Minimax H3 Latent Upscaler Opt (3D) [Model]",
                category="MiniMaxH3 Opt",
                search_aliases=["minimax", "h3", "latent", "upscale", "3d", "model"],
                description=(
                    "Minimax H3 latent upscaler (3D) with Temporal Chunking and pixel-space "
                    "alignment, packaged as a model object. Same parameters as the original node; "
                    "instead of taking a latent it returns a LATENT_UPSCALE_MODEL that other "
                    "nodes/code can call with a latent."
                ),
                inputs=[
                    io.Combo.Input(
                        "model_name",
                        options=scan_models(),
                        tooltip="Minimax H3 upscale model.",
                    ),

                    io.DynamicCombo.Input(
                        "mode",
                        tooltip="How the target size is computed.",
                        options=[
                            io.DynamicCombo.Option(UpscaleMode.SCALE_BY, [
                                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05,
                                               tooltip="Upscale factor."),
                            ]),
                            io.DynamicCombo.Option(UpscaleMode.TARGET_DIMENSIONS, [
                                io.Int.Input("width", default=1280, min=64, max=8192, step=8,
                                             tooltip="Target pixel width."),
                                io.Int.Input("height", default=704, min=64, max=8192, step=8,
                                             tooltip="Target pixel height."),
                            ]),
                            io.DynamicCombo.Option(UpscaleMode.MEGAPIXELS, [
                                io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
                                               tooltip="Target megapixels."),
                            ]),
                        ],
                    ),

                    io.Int.Input("align", default=32, min=1, max=512, step=1,
                                 tooltip="Pixel-space alignment. 32 is strictly recommended."),

                    io.Boolean.Input(
                        "enable_temporal_chunking", default=True,
                        tooltip="Enable temporal chunking to save VRAM for long videos and fix "
                                "end-frame flickering."),
                    io.Boolean.Input(
                        "force_unload", default=True,
                        tooltip="Unload model to CPU after inference to free VRAM for subsequent nodes. "
                                "Disable if you run this node repeatedly to avoid reload overhead."),

                    io.Combo.Input("device", options=["cuda", "rocm", "cpu"], default="cuda"),
                    io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="fp16"),
                ],
                outputs=[
                    io.LatentUpscaleModel.Output(
                        display_name="upscale_model",
                        tooltip="The MiniMax H3 latent upscale model with the settings above baked in. "
                                "Call it from anywhere: upscale_model(latent).",
                    ),
                ],
            )

        @classmethod
        def execute(cls, model_name: str, mode: dict, align: int,
                    enable_temporal_chunking: bool, force_unload: bool,
                    device: str, precision: str) -> io.NodeOutput:
            if not isinstance(mode, dict):
                mode = {"mode": mode}
            selected = mode.get("mode") or UpscaleMode.SCALE_BY.value

            model = MiniMaxH3LatentUpscaleModelOpt(
                model_name,
                device=device,
                precision=precision,
                align=align,
                enable_temporal_chunking=enable_temporal_chunking,
                force_unload=force_unload,
                mode=selected,
                scale=float(mode.get("scale", 2.0)),
                width=int(mode.get("width", 1280)),
                height=int(mode.get("height", 704)),
                megapixels=float(mode.get("megapixels", 1.0)),
            )
            return io.NodeOutput(model)

else:  # pragma: no cover - only on ComfyUI without comfy_api.latest
    MiniMaxH3LatentUpscaleModelOptNode = None


__all__ = [
    "MiniMaxH3LatentUpscaleModelOptNode",
    "UpscaleMode",
    "MODE_SCALE",
    "MODE_TARGET_DIMENSIONS",
    "MODE_MEGAPIXELS",
]
