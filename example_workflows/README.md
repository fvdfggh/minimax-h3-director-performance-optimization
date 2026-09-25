# 示例工作流

把 `.json` 直接拖进 ComfyUI 画布即可打开。打开后请按自己机器上的模型文件名，重新选择
`UNETLoader` / `VAELoader` / `CLIPLoader` 里的模型与 VAE。

所有工作流都基于同一条官方 MiniMax H3 管线：Director 节点出 `images` / `audio`，
再接 `VHS_VideoCombine`（或 `SaveImage` / `SaveAudio`）保存。

| 文件 | 任务 | 说明 |
|---|---|---|
| `minimax_h3_director_t2v.json` | t2v | 纯文生音视频，最小可用配置 |
| `minimax_h3_director_r2v.json` | r2v | 参考图生视频：提示词里用 `<Picture N>` 指代参考槽 |
| `minimax_h3_director_fl2v.json` | fl2v | 首尾帧镜头组：每个镜头一张首帧 + 可选尾帧 |
| `minimax_h3_director_v2v.json` | v2v | 上传源视频，按时间轴分段编辑（每段源画面作为 `<Video 1>`） |
| `minimax_h3_director_rv2v.json` | rv2v | 源视频 + 参考图 / 参考音频一起改视频 |
| `minimax_h3_director_external_groups_i2v.json` | i2v | 用 `Group (Image to Video)` + `Groups Combine` 从图上接线喂给 Director |
| `minimax_h3_director_external_groups_r2v.json` | r2v | 用 `Group (Reference to Video)` + `Groups Combine` 从图上接线喂给 Director |
| `minimax_h3_director_加速版.json` | — | 更快的参数组合（更保守的片段长度 / 采样设置），用于快速验证管线是否跑通 |

## 使用步骤

1. 拖入工作流 → 补上缺失的模型（UNET、视频 VAE、音频 VAE、CLIP type=minimax）。
2. 选中 Director 节点，在节点内的时间轴上填提示词、上传素材。
3. 需要出文件时，确认 `images` / `audio` 已接到保存节点。
4. 按 **Run** 入队；进度、采样预览与最终报告都显示在节点上。

## 提示

- 首次运行会写缓存到 `output/minimax_director_opt_cache/<工作流名>/node_<id>/`；换机器 / 换源视频后
  相关片段会自动重算。
- 只想重跑某几段，用工具栏的「选择运行」勾选后再 Run。
- 已经跑过的片段可以用「分段导出」直接输出到节点，不再采样。
- 想放大出片，接 `Minimax H3 Latent Upscaler Opt (3D) [Model]` 后用「二次采样」。
