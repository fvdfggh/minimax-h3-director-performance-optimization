# 保留音频：待确认项 —— 音频 latent 形状是否依赖画布

> 状态：**已核实（假设成立）**。2026-09-25 对本机 ComfyUI
> （`comfy_extras/nodes_minimax_h3.py`、`comfy/samplers.py`、`comfy/utils.py`、
> `comfy/ldm/minimax/model.py`）逐条比对后确认，本文保留作为核对记录。
>
> 文件名沿用 `OPEN_QUESTION`，仅为避免历史引用失效。

---

## 1. 结论

**AV latent 的音频流 `[1, 32, 2, T]` 的 `T` 只由帧数决定，与 width / height 无关。**
跨画布复用（二采放大、卡片换画布）安全，无需任何补偿。

## 2. 上游证据

### 2.1 音频流形状与 T 的算法

`comfy_extras/nodes_minimax_h3.py:29-49`

```python
FPS = 24
AUDIO_LATENT_FPS = 40

def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)

def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16], ...)
    audio = torch.zeros([batch_size, 32, 2, audio_t], ...)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count
```

- `T = round(frame_count / 24 * 40)` —— 参数表里没有 width / height；
  只有 `video` 用到 `height // 16`、`width // 16`。
- `AUDIO_LATENT_FPS = 40` 与本仓 `h3_motion_context.AUDIO_HZ = 40.0` 一致，
  `FRAME_RESCALE = 5/3 = 40/24`（`comfy/ldm/minimax/model.py:31`）。

### 2.2 时间轴换算（头针偏移的依据）

`MiniMaxH3AddGuide.execute`：`max_rt = floor(audio_t - FRAME_RESCALE * frame_idx)`，
注释明确写着「the streams share one time axis: FRAME_RESCALE per pixel frame, 1.0
per audio latent frame」。所以

```
tick = 帧号 × 40 / 24
```

本仓 `audio_retain.apply_retain_audio(head_seconds=...)` 里
`head_ticks = round(head_seconds × 40)` 与之完全一致。

### 2.3 音频 VAE 编码的输出形状

`comfy_extras/nodes_minimax_h3.py:73-80`

```python
def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]            # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))   # [1, 32, 2, T]
    return z, z.shape[-1]
```

- 官方确认编码结果就是 `[1, 32, 2, T]`，与音频流**同形**；
  `audio_retain.py` 里那段 `[B, 2C, T] → [B, C, 2, T]` 的 reshape 兜底
  实际不会触发（判据 `z.shape[1] == C * 2` 也不会误命中），留着只是防御。
- `getattr(audio_vae, "audio_sample_rate", 32000)` 与本仓写法一致。

### 2.4 双流 `noise_mask` 是官方支持路径

`comfy/samplers.py:1297-1314`

```python
if denoise_mask is not None:
    if denoise_mask.is_nested:
        denoise_masks = denoise_mask.unbind()          # [video_mask, audio_mask]
        denoise_masks = denoise_masks[:len(latent_shapes)]
    for i in range(len(denoise_masks)):
        denoise_masks[i] = prepare_mask(denoise_masks[i], latent_shapes[i], ...)
    if len(denoise_masks) > 1:
        denoise_mask, _ = comfy.utils.pack_latents(denoise_masks)
```

`prepare_mask` → `comfy.utils.reshape_mask`：`[1,1,1,T]` 先 bilinear 到
`[1,1,2,T]` 再通道重复到 `[B,32,2,T]`，全 0 插值后仍是全 0，锁定精确成立。
两条流随后被 `pack_latents` 拼成一条 `[B,1,video_flat+audio_flat]`。

锁定本身由 `KSamplerX0Inpaint.__call__` 完成：

```python
x   = x * denoise_mask + scale_latent_inpaint(...) * (1 - denoise_mask)
out = out * denoise_mask + latent_image * (1 - denoise_mask)
```

mask=0 的位置每步都被换回 `latent_image`（我们写进去的编码音频），
模型看得见但改不动 —— 正是「保留音频」要的语义。
`comfy/ldm/minimax/model.py:32` 的 `VISUAL_COND_TIMESTEP = 0.999` 进一步说明
H3 会把这些 token 当视觉条件处理。

## 3. 由此修正的一处实现判断

初版 `audio_retain.py` 的注释写的是「刻意不设 `PREFIX_STEPS_KEY`，否则 continue
remask 的 `apply_model` 钩子会用只含视频的 5-D mask 覆盖掉音频流」。核实后**该
判断不成立**：

- remask 拿到的是**已打包**的 3-D mask（视频+音频拼在一条），
  `_PrefixRemask.denoise_mask_function` 只改写前 `video_flat` 那一截
  （`packed[..., :elems]`），音频半截原样保留；
- `apply_model_wrapper` 确实把 5-D 视频 mask 交给了 transformer，但那个 mask
  不参与上面那两行 `latent_image` 混合，锁不由它负责。

所以「段间锥形重绘」与「保留音频」可以共存，`PREFIX_STEPS_KEY` /
`CONTINUE_SEAM_KEY` **不应**被移除（移除只会白白丢掉 per-sigma 的接缝微调）。

---

## 4. 顺带澄清：`shift video` / `shift audio` 不是画布参数

容易被误当成画布相关，实际是 **sigma 调度偏移**，按视频/音频两条流分别设置。

```python
# director/core_sampling.py:99-103
if apply_shift:
    shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
    model_use = _unpack_node_output(shifted)[0]
```

- 定义：`nodes/director.py:220-227`，默认 `shift_video=12.0`、`shift_audio=3.0`
- 语义：把噪声调度按 `σ' = shift·σ / (1 + (shift−1)·σ)` 重新分布。
  值大 → 更多步落在高噪声区（先定构图/运动）；值小 → 更多步落在低噪声区（细节）
- 为什么两个：H3 的 DiT 是双流（视频 + 音频），但 `KSampler` 只有一条 sigmas，
  官方于是按流分别 patch。视频流 token 多、需要更强的结构偏置，所以 12；音频流维度低，3 就够
- 返回的是**打过补丁的 MODEL**，不是 per-call 参数；先 shift 再 sample
- 二采透传同一对值：`nodes/director.py:503-504`

---

## 5. 相关文件速查

| 文件 | 作用 |
|---|---|
| `director/audio_retain.py` | 保留音频全部逻辑（解析开关 / 读 PCM / 注入 latent + mask） |
| `director/audio_extract.py` | 提取音频的持久缓存（音频 tab 的数据源） |
| `director/batch_phases.py` | 一采采样注入 + 出片复用 PCM |
| `director/second_sampling.py` | 二采采样注入 + 出片复用 PCM |
| `director/batch_executor.py` | 构造 retain 缓存并传给 Phase 2 / 3 |
| `comfy_extras/nodes_minimax_h3.py:29-49` | `FPS` / `AUDIO_LATENT_FPS` / `temporal_shape()` |
| `comfy_extras/nodes_minimax_h3.py:73-89` | `_encode_ref_audio()` / `_empty_av_latent()` |
| `comfy/ldm/minimax/model.py:30-33` | `FRAME_PER_TOKEN` / `FRAME_RESCALE` / `VISUAL_COND_TIMESTEP` |
| `comfy/samplers.py:1276-1315` | 双流 mask 的打包与 `KSamplerX0Inpaint` 锁定 |
| `comfy/utils.py:1348-1367` | `reshape_mask()` —— 音频 mask 的逐维对齐 |
| `director/h3_motion_context.py:28-31` | 本仓 `FPS` / `AUDIO_HZ` / `FRAME_RESCALE`（与上游一致） |

开关字段：`timeline.segments[i].retainAudioId`（存 audio_extract 的 entry id，空 = 不保留）
