"""Single-stage sampling for MiniMax H3 (SigmaShift + KSampler).

Pass ``sigmas`` to override the KSampler schedule (ManualSigmas-style).
``apply_shift=False`` skips ``MiniMaxH3SigmaShift``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def normalize_sigmas(raw):
    """Normalize a wired SIGMAS input into a 1-D float32 tensor.

    Accepts a BasicScheduler / ManualSigmas tensor, a list/tuple, or a comma
    separated string. Returns ``None`` when the input is unusable so the caller
    can silently fall back to the default steps + scheduler schedule.

    Same contract as the old Refine pass: N sigmas describe N-1 steps, and the
    schedule must end at 0 (a missing trailing 0 is appended).
    """
    if raw is None:
        return None
    import torch

    try:
        if torch.is_tensor(raw):
            values = [float(x) for x in raw.detach().float().cpu().reshape(-1).tolist()]
        elif isinstance(raw, (list, tuple)):
            values = [float(x) for x in raw]
        else:
            text = str(raw).replace(";", ",").replace("\n", ",")
            values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except (TypeError, ValueError):
        log.warning("Director: 无法解析的 SIGMAS 输入，已回退到默认采样。")
        return None
    if len(values) < 2:
        log.warning(
            "Director: SIGMAS 至少需要 2 个值（N 个 sigma = N-1 步），已回退到默认采样。"
        )
        return None
    if abs(values[-1]) > 1e-8:
        values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
    sigmas=None,
    apply_shift: bool = True,
):
    import comfy.sample
    import comfy.utils
    import latent_preview
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify(phase_name, 0)
    model_use = model
    if apply_shift:
        shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
        model_use = _unpack_node_output(shifted)[0]

    neg = negative if negative else []
    sigma_list = None
    if sigmas is not None:
        import torch

        if torch.is_tensor(sigmas):
            sigma_list = sigmas.detach().float().cpu().reshape(-1)
        else:
            sigma_list = torch.tensor([float(x) for x in sigmas], dtype=torch.float32)
        steps = max(1, int(sigma_list.numel()) - 1)
    else:
        steps = int(steps)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model_use,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    noise = comfy.sample.prepare_noise(
        latent_image,
        int(seed),
        latent.get("batch_index", None),
    )
    noise_mask = latent.get("noise_mask", None)

    base_cb = latent_preview.prepare_callback(model_use, steps)
    every = max(1, int(preview_every))

    def callback(step, x0, x, total_steps):
        if on_step_preview is not None:
            try:
                last = max(0, int(total_steps) - 1)
                # preview_every < 0 → only the last step (sigma refine starts dirty).
                if int(preview_every) < 0:
                    show = step >= last
                else:
                    show = step % every == 0 or step >= last
                if show:
                    on_step_preview(int(step), int(total_steps), x0)
            except Exception as exc:
                log.debug("Step preview callback skipped: %s", exc)
        if base_cb is not None:
            base_cb(step, x0, x, total_steps)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    # Manual sigmas still go through KSampler.sample (same path as first pass /
    # schedule=steps). sample_custom skips SigmaShift wiring and can yield
    # undecodable AV latents on H3.
    samples = comfy.sample.sample(
        model_use,
        noise,
        steps,
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        neg,
        latent_image,
        denoise=1.0 if sigma_list is not None else float(max(0.0, min(1.0, denoise))),
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
        sigmas=sigma_list,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    notify(phase_name, 1)
    return out
