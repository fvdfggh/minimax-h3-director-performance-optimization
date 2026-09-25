# ComfyUI MiniMax H3 Director Opt

多片段「时间轴导演」节点包，在 ComfyUI **官方 MiniMax H3 管线**（`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`，对应 ComfyUI PR #15224 / #15228）之上，提供节点内嵌的时间轴编辑器、分段缓存、段间连贯、二次采样与音视频导出。

- 上游项目：`AIMixer/ComfyUI_MiniMaxH3_Director`（作者：AI搅拌手）
- 本 Opt 分支作者：**fvdfggh**
- 仓库：<https://github.com/fvdfggh/minimax-h3-director-performance-optimization>
- 许可：Apache-2.0（见 `LICENSE`）

> 本分支的所有节点类型 id、显示名、HTTP 路由、前端事件与缓存根目录都带 `Opt` 标记，因此**可以和原版同时安装**，互不冲突。

---

## 目录

1. [特性概览](#特性概览)
2. [安装](#安装)
3. [模型与前置条件](#模型与前置条件)
4. [快速开始](#快速开始)
5. [节点清单](#节点清单)
6. [Director 主节点参数](#director-主节点参数)
7. [任务类型 t2v / i2v / fl2v / r2v / v2v / rv2v](#任务类型-t2v--i2v--fl2v--r2v--v2v--rv2v)
8. [时间轴编辑器（前端）](#时间轴编辑器前端)
9. [段间引导与锥形重绘](#段间引导与锥形重绘)
10. [二级采样（二采）](#二级采样二采)
11. [分段导出](#分段导出)
12. [音频：生成 / 原声 / 静音、提取与保留](#音频生成--原声--静音提取与保留)
13. [音频有效性校验（ASR）](#音频有效性校验asr)
14. [缓存与「清空缓存」](#缓存与清空缓存)
15. [导演包导入 / 导出](#导演包导入--导出)
16. [HTTP 接口与 WebSocket 事件](#http-接口与-websocket-事件)
17. [示例工作流](#示例工作流)
18. [常见限制与排错](#常见限制与排错)

---

## 特性概览

- **节点内时间轴**：上传源视频 → 分割 / 均分 / 智能分割 → 逐段写提示词、挂参考图 / 参考音频 / 参考视频 → 队列运行。
- **官方 H3 管线**：conditioning 直接走 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`，采样为单段 KSampler + `MiniMaxH3SigmaShift`，解码走 `LTXVSeparateAVLatent`，音画同出。
- **六种任务**：`t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v`。
- **分段缓存**：每段按「内容指纹」落盘，改一段只重算一段；缓存位于 `output/minimax_director_opt_cache/<工作流名>/node_<id>/`。
- **段间引导**：把上一段尾帧写进本段 body 前缀再重绘（continue 模式），接缝按 5 / 22 / 39 / 56 帧上下文窗口对齐。
- **二级采样**：对已缓存段做「放大 + 重采样」，独立 `seg2_*` 缓存，不覆盖一采结果。
- **分段导出**：勾选片段直接输出到节点 `images` / `audio`（分片或连续两种模式，可选一采 / 二采来源）。
- **音频三态**：`generate`（模型生成）/ `source`（用源视频原声）/ `mute`（静音）；另有「提取音频」「保留音频」。
- **音频有效性校验**：接 MOSS 转写模型后，用 ASR 把生成音轨与提示词里的台词逐说话人比对。
- **导演包（.mmxpack.zip）**：时间轴 + 素材一键导出 / 导入，与原版包格式兼容。
- **外部 Group 节点**：可从图上接线 `MMX_DIR_GROUP`，优先级高于节点内 UI 卡片。

---

## 安装

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/fvdfggh/minimax-h3-director-performance-optimization ComfyUI_MiniMaxH3_Director_Opt
pip install -r ComfyUI_MiniMaxH3_Director_Opt/requirements.txt
```

依赖（`requirements.txt`）：

| 包 | 用途 |
|---|---|
| `av>=13.0` | v2v / 时间轴的源视频解码（ComfyUI 便携版自带；刻意不用 OpenCV） |
| `imageio-ffmpeg>=0.4` | 源音频抽取、片段 mp4 增量编码 |
| `scenedetect>=0.6.4,<0.8` | 「智能分割」镜头检测 |

重启 ComfyUI 后，前端会自动加载 `web/js`（扩展名 `ComfyUI.MiniMaxH3DirectorOptPlugin`）。

---

## 模型与前置条件

| 口 | 类型 | 说明 |
|---|---|---|
| `model`（`model_b` / `model_c`） | MODEL | MiniMax H3 UNET（`UNETLoader`）。`model` 必接，也是备用口未接线时的兜底 |
| `video_vae` | VAE | MiniMax H3 视频 VAE（`minimax_h3_video_vae`） |
| `audio_vae` | VAE | MiniMax H3 音频 VAE（`minimax_h3_audio_vae`），`r2v` / `v2v` / `rv2v` 必接 |
| `clip` | CLIP | `CLIPLoader` type=minimax（qwen3vl） |
| `upscale_model` | LATENT_UPSCALE_MODEL | 二采专用，接 `Minimax H3 Latent Upscaler Opt (3D) [Model]` |
| `asr_model` | T8_MOSS_TRANSCRIBE_MODEL | 可选，来自 `Comfyui-MOSS-Transcribe-Diarize-T8` |

还要求 ComfyUI 自带 `comfy_extras.nodes_minimax_h3`（官方 MiniMax H3 节点）。缺失时节点会在运行时报「Upgrade to ComfyUI with PR #15224 merged」。

二采节点还需要 `comfy_api.latest`（V3 节点 API）：不可用时 `Minimax H3 Latent Upscaler Opt (3D) [Model]` 与两个 Autogrow Group 节点会降级 / 不注册。

---

## 快速开始

1. 添加节点 **MiniMaxH3Director Opt**（分类 `MiniMaxH3 Opt`）。
2. 接好 `model` / `video_vae` / `audio_vae` / `clip`。
3. 在节点内的时间轴上选任务类型、写全局提示词（或分段提示词）、上传素材。
4. 把 `images` / `audio` 接到 `VHS_VideoCombine`（或 `SaveImage` / `SaveAudio`）出片。
5. 按 ComfyUI 的 **Run** 入队；进度与采样预览会实时显示在节点上。

默认画布：864×480（0.4 MP 16:9）、124 帧 @ 24 fps、cfg 1.0、steps 25、`res_multistep` + `simple`、`shift_video=12` / `shift_audio=3`（对齐官方模板）。

---

## 节点清单

| 类型 id | 显示名 | 分类 | 输出 |
|---|---|---|---|
| `MiniMaxH3DirectorOpt`（旧 id `ComfyMiniMaxH3DirectorOpt` 仍可加载） | MiniMaxH3Director Opt | `MiniMaxH3 Opt` | `images`, `audio`, `fps`, `frame_count`, `source_images`, `report` |
| `MiniMaxH3DirectorOptConditioning` | MiniMax H3 Director Opt Conditioning | `MiniMaxH3 Opt` | `positive`, `latent` |
| `MiniMaxH3DirectorOptPlannerConditioning` | MiniMax H3 Director Opt Planner Conditioning | `MiniMaxH3 Opt` | `positive`, `latent`, `task_mode` |
| `MiniMaxH3DirectorOptGroupImageToVideo` | MiniMax H3 Director Opt Group (Image to Video) | `MiniMaxH3 Opt/Director Groups` | `group` |
| `MiniMaxH3DirectorOptGroupReferenceToVideo` | MiniMax H3 Director Opt Group (Reference to Video) | `MiniMaxH3 Opt/Director Groups` | `group` |
| `MiniMaxH3DirectorOptGroupsCombine` | MiniMax H3 Director Opt Groups Combine | `MiniMaxH3 Opt/Director Groups` | `groups` |
| `MiniMaxH3FastVideoVAEOpt` | MiniMax H3 Fast Video VAE Opt | `MiniMaxH3 Opt` | `vae` |
| `MiniMaxH3LatentUpscaleModelOpt` | Minimax H3 Latent Upscaler Opt (3D) [Model] | `MiniMaxH3 Opt` | `upscale_model` |

主节点输出中 `images` / `audio` / `source_images` 是**列表**（`OUTPUT_IS_LIST`）：导出模式为「分段」时每段一条。

### Group 节点（外部接线）

- **Group (Image to Video)**：`prompt` + `duration_sec` + 可选 `first_frame` / `last_frame`。无帧 = t2v，仅首帧 = i2v，有尾帧 = fl2v。
- **Group (Reference to Video)**：`prompt` + `duration_sec` + Autogrow 槽位 —— 参考图 ≤ 9、参考视频 ≤ 3、参考视频音轨 ≤ 3、独立参考音频 ≤ 3（与官方 R2V 一致的自动增长交互）。
- **Groups Combine**：把多个 Group 合成一个列表；**不允许** i2v 组与 r2v 组混在一个列表里。

接上 `i2v_groups` / `r2v_groups` 时，执行以外部组为准（覆盖节点内 UI 卡片）。输出尺寸始终由 Director 节点决定，Group 节点不管尺寸。

---

## Director 主节点参数

### 必填

| 控件 | 默认 | 说明 |
|---|---|---|
| `model` / `video_vae` / `audio_vae` / `clip` | — | 见「模型与前置条件」 |
| `task_type` | `t2v — 文生视频(Text to Video)` | 六选一，见下节 |
| `global_prompt` | `A cinematic scene with natural motion and synchronized ambience` | 全局提示词，直接送入 H3 节点 |
| `cfg` | 1.0 | KSampler cfg |
| `seed` | 0 | 随机种子（支持 control_after_generate） |
| `frame_rate` | 24.0 | 时间轴 / 输出帧率 |
| `width` / `height` | 864 / 480 | 画布，step 32 |
| `ref_max_size` | 864 | 长边缩放模式下的最长边 |
| `total_frames` | 124 | 时间轴总帧数（fl2v = 各镜头之和） |
| `timeline_data` | — | 内部字段，由前端时间轴写入（UI 中隐藏） |

### 可选

| 控件 | 默认 | 说明 |
|---|---|---|
| `run_model` | 主模型 (model) | 本次用哪个 MODEL 口：`model` / `model_b` / `model_c`；选中的口没接线则回退主模型 |
| `model_b` / `model_c` | — | 备用 UNET 1 / 2 |
| `i2v_groups` / `r2v_groups` | — | 外部 Group 包，优先级高于 UI 卡片 |
| `steps` | 25 | 采样步数 |
| `sampler` | `res_multistep` | KSampler 采样器 |
| `scheduler` | `simple` | 调度器 |
| `shift_video` / `shift_audio` | 12.0 / 3.0 | `MiniMaxH3SigmaShift` |
| `sigmas` + `use_sigmas` | 关闭 | 自定义噪声调度（接 `BasicScheduler` / `ManualSigmas`）。开启后步数 = `len(sigmas)-1`、denoise 固定 1.0，`steps` / `scheduler` 不再参与调度；未接线或解析失败自动回退 |
| `conn_noise` | 关闭 | 段间锥形重绘开关（见「段间引导」） |
| `upscale_model` | — | 二采**硬性要求**，未接则二采直接报错 |
| `second_sigmas` | 未接线 | 二采专用调度，未接线用默认海螺二采 `(0.85, 0.7250, 0.4219, 0.0)`（euler 3 步） |
| `second_run_model` | 主模型 | 二采用哪个 MODEL 口 |
| `second_seed` | 20240 | 二采固定种子，与一采 seed 相互独立 |
| `asr_model` | 可选 | 音频有效性校验：接上后运行一次，r2v 模式下节点上出现「音频有效性校验」按钮 |
| `workflow_name` | 隐藏 | 前端自动写入当前工作流名，用于缓存分目录 |

节点上的分组标题（`采样设置` / `高级采样` / `二级采样`）是前端自定义控件 `BDGROUP`，只用于折叠，不参与计算。

### 固定行为（不再是控件）

以下开关已固化为常量，写在 `nodes/director_common.py` 顶部，改完需重启 ComfyUI：

| 常量 | 值 | 含义 |
|---|---|---|
| `USE_CONDITIONING_CACHE` | `True` | 跨运行复用磁盘上的 CLIP / VAE conditioning（键是提示词 + 画布 + 模型的指纹） |
| `CLEAR_VRAM_BETWEEN_SEGMENTS` | `True` | 每段结束后卸载模型并清 CUDA 缓存 |
| `EXPORT_SOURCE_IMAGES` | `False` | 是否把源时间轴画面解码到 `source_images` 输出 |

---

## 任务类型 t2v / i2v / fl2v / r2v / v2v / rv2v

| key | 名称 | 说明 |
|---|---|---|
| `t2v` | 文生视频 | 无首帧、无参考 |
| `i2v` | 图生视频 | 首帧约束（`ImageToVideo` + `first_frame`） |
| `fl2v` | 首尾帧生视频 | 首帧 + 尾帧（可不传图，等同文生） |
| `r2v` | 参考主体生视频 | 参考图 / 视频 / 音频，提示词用 `<Picture N>` / `<Video K>` / `<Audio J>` |
| `v2v` | 视频转视频 | 上传源视频后按时间轴分段编辑，每段源画面作为 `<Video 1>` 送入 `ReferenceToVideo` |
| `rv2v` | 参考素材改视频 | 源视频 `<Video 1>` + 参考图 `<Picture N>` + 参考音频 `<Audio J>`；无参考素材时等同 v2v |

前端按任务分三种模式：

- `fl2v` → 镜头（首帧 / 尾帧）面板；
- `t2v` / `i2v` / `r2v` → 提示词组（批量）面板，且不显示源视频上传；
- `v2v` / `rv2v` → 源视频时间轴面板。

---

## 时间轴编辑器（前端）

入口 `web/js/minimax_timeline.js`，DOM 控件名 `minimax_director_ui`，编辑器类 `MiniMaxH3DirectorOptEditor`（由 25 个 mixin 组装）。节点默认尺寸 `[1000, 680]`，最小宽度 900。

### 工具栏

| 按钮 | 作用 |
|---|---|
| 上传视频 / 选已有视频 / 追加视频 | 载入源视频（追加会拼到时间轴末尾并新建片段） |
| + 分割 | 在当前帧切一刀 |
| 均分（段数 2–64） | 等分时间轴，并保留 clip 边界作为强制分割点 |
| 智能分割 | 调后端 `detect_shots`（scenedetect）自动切镜头；需总帧 ≥ 8 |
| 选择运行 / 全选 | 只重跑勾选的片段 |
| 分段导出 | 勾选片段 → 直接输出到节点（分片 / 连续，可选一采 / 二采缓存） |
| 二次采样 | 勾选片段 → 放大 + 重采样 |
| 提取音频 | 从源视频抽音，作为可复用的音频条目 |
| 删除片段 | 会**真的裁掉源帧**并重排后续片段 |
| 全局模式 / 分段模式 | 切换提示词编辑范围 |
| 导入导演包 / 导出导演包 | `.mmxpack.zip` |
| 清空缓存 / 清空节点所有缓存 | 见「缓存与清空缓存」 |
| EN | 中 / 英界面切换（存于 `mmx_director_ui_locale`） |

### 输出条

分辨率预设（默认 16:9 宽屏）、百万像素（0.1–16）、最长边、宽 / 高（step 32）、缩放模式（`long_edge` / `fixed`）、帧率（1–240）、声音（`generate` / `source` / `mute`）、导出方式（`all` 全部导出 / `segments` 分段导出）、段间引导开关 + 上下文帧数（5 / 22 / 39 / 56，默认 22）、实时预览开关。

### 播放器与预览

`▶` 播放 / `⟳` 循环 / `‹` `›` 前后一帧、帧号输入（1-based）、时间码、进度条；采样过程中会实时回传预览图（`minimax_director_opt_preview`）。

### 快捷键（鼠标在编辑器上才生效）

| 键 | 作用 |
|---|---|
| `Delete` / `Backspace` | 删除选中片段 |
| `Space` | 播放 / 暂停 |
| `←` / `→` | ±1 帧；`Shift` + 方向键 ±10 帧 |
| `Esc` | 关闭弹窗 / 还原帧号输入 |
| 画布右键 | 在该位置加分割点（fl2v 模式无效） |

焦点在 `INPUT` / `TEXTAREA` / `SELECT` / 可编辑区域时快捷键全部失效。**分割点不能用 Delete 删除**，要点「删除分割点」。

### 拖拽

拖片段主体排序；拖左右边缘改起止（视频模式会同步相邻段，fl2v 会连带后移）；文件可直接拖入（视频 → 载入源视频；图片 → 落到参考槽）。

---

## 段间引导与锥形重绘

「段间引导」是**按时间轴开关**的（默认关闭，写在 `timeline.output.continuityEnabled`）：

- 需 ≥ 2 个片段；仅 `t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v` 生效；第 1 段永不引用。
- 上下文帧数只能取 **5 / 22 / 39 / 56**（默认 22，对齐 VAE 的 17 帧周期）。
- 每段可单独关「引用上段」（默认开）；「对齐下段」默认关，只有下一段已有缓存 AV latent 时才可用。
- `conn_noise`（段间锥形重绘）开启时，把上一段尾写入本段 body 前缀并重绘（continue 模式），重绘幅度由时间轴面板的「重绘幅度」控制，默认 0.10（0 = 接缝硬锁，0.95 = 几乎不重绘）。
- 关闭 `conn_noise` 时只做参考帧引导（guide），不重绘。
- **不要和独立的 H3 Motion Context 节点同时启用。**

换源视频后旧缓存失效：若某段需要上一段结果而上一段没跑过 / 未纳入「选择运行」，会直接报错提示，而不是静默出坏片。

---

## 二级采样（二采）

对**已缓存**的片段做「放大 → 重采样 → 合并出片」：

1. 读出一采的 AV latent；
2. 用 `upscale_model`（`Minimax H3 Latent Upscaler Opt (3D) [Model]`）放大；
3. 按一采实际 pin 的上下文帧数重新接缝；
4. 用二采自己的噪声调度采样（默认 euler 3 步；接线时用连入的 SIGMAS + 高级采样里的采样器）；
5. 结果写进独立的 `seg2_*` 文件组与 `segment_slots_2nd.json`，**不会覆盖一采**；
6. 出片按「相邻段」分批，避免所有段的解码帧同时驻留内存。

硬性要求：必须接 `upscale_model`，否则直接报错（不再静默回退到原分辨率）。

二采触发方式：节点上点「二次采样」勾选片段 → 前端往 `timeline.output.secondSample` 写一次性标志并自动入队。它与「运行」互斥：携带该标志的那次运行只做二采，不跑一采。

---

## 分段导出

节点上点「分段导出」勾选片段（无缓存的片段灰掉）→ 一次性标志写进 `timeline.output.segmentExport` 并自动入队。

- **导出模式**：`piecewise`（每个勾选片段一个视频）/ `continuous`（时间轴上相邻的勾选片段拼成一个视频）。
- **缓存来源**：`1st`（`seg_*`，一采）/ `2nd`（`seg2_*`，二采）。
- 结果**直接输出到节点的 `images` / `audio`**，不再另外写磁盘；`images` 每个 clip 一条。
- 二采时，相邻且「引用上段」的片段会被合并成一个不可拆分的运行区间。

---

## 音频：生成 / 原声 / 静音、提取与保留

声音三态写在 `timeline.output.audioMode`：

| 模式 | 行为 |
|---|---|
| `generate`（默认） | 用模型生成的音轨 |
| `source` | 用源视频原声（仅 `v2v` / `rv2v` 有效，其他任务自动回退 `generate`） |
| `mute` | 静音输出（44.1 kHz 空音轨） |

全部导出合并画面时，各组音频会按时间轴拼接，不会出现后半段无声。

**提取音频**：从源视频抽出一段 PCM，存在 `<节点缓存目录>/audio_extract/`，按 **entry id** 命名（不是内容哈希、不是时间轴位置），因此改提示词 / 重跑都不会失效，「清空缓存」也不会误删。片段重排时条目跟着卡片走，卡片删除时条目一并删除。

**保留音频**：把某条提取音频标记为该片段的「保留音频」。运行 Phase 2 会把它 VAE 编码进 AV latent 的音频流并把该流的 `noise_mask` 置零 —— UNet 看得到它、以它为条件，但永远不会重绘它；Phase 3 跳过音频 VAE 解码，直接把同一份 PCM 混流回去，所以听到的就是你选的那一段，不经过音频 VAE 往返。

---

## 音频有效性校验（ASR）

接上 `asr_model`（来自 `Comfyui-MOSS-Transcribe-Diarize-T8` 的 `T8_MOSS_ModelLoader`，类型 `T8_MOSS_TRANSCRIBE_MODEL`）并**运行一次**节点后，r2v 模式的节点上会出现「**音频有效性校验**」按钮：

1. 点按钮 → 弹出片段列表，勾选要核对的片段；
2. 确认后，后端读取这些片段**已缓存**的音轨，与本段**当前**提示词里的台词块逐说话人比对（错误率 + 说话人是否对得上）；
3. 结果以弹窗展示，不写入 `report`，也**不会重新生成任何片段**。

没有音轨缓存的片段会被明确列为「已跳过」，不会被当成通过。所以用法是：先生成，再改提示词、随时点按钮复核。

台词必须严格写成：

```
<Subject 1> (S1) says: <d>[Chinese] 师尊，你一直说你是毒修。</d>
```

- `<Subject N>`：哪个参考主体在说；
- `(SN)`：说话人编号；
- `<d>[lang]…</d>`：台词内容与语言标签。

格式不对就当普通文本（前端不渲染成说话人卡片，也不参与校验）。长音频超过 30 秒会自动改走分块长音频识别，报告里会写明走的是哪条路径。

详见 [`docs/asr_check.md`](docs/asr_check.md)。

---

## 缓存与「清空缓存」

统一缓存根目录：

```
output/minimax_director_opt_cache/<工作流名 slug>/node_<节点 id>/
```

同一目录内靠文件名前缀区分用途（`<hash>` 是片段**内容**指纹，不是位置）：

| 前缀 / 文件 | 内容 |
|---|---|
| `cond_text_<hash>.pt` | 文本编码缓存 |
| `seg_<hash>_latent.pt` | 一采采样 latent |
| `seg_<hash>_clip.mp4` | 一采成片 |
| `seg_<hash>_frames_ht.mp4` (+`.json`) | 头尾接缝窗口（crf 12） |
| `seg_<hash>_audio.pt` | 音频 latent |
| `seg_<hash>_meta.json` / `_handoff.json` | 指纹与交接信息 |
| `segment_slots.json` | 位置 → 文件组映射（一采） |
| `seg2_<hash>_*` / `segment_slots_2nd.json` | 二采的独立一套 |
| `seg_XXXX_scratch_*.pt` | 单次运行的临时文件 |
| `audio_extract/` | 提取音频（独立生命周期） |
| `_vit/` | 全局 ViT（视觉塔）输出缓存，跨工作流复用，上限 8 GB |

指纹包含提示词、任务、画布、帧率、参考素材文件、源视频身份（相对路径 + 大小 + mtime）、连续性开关与管线版本。**改了这些中的任何一项，该段就会重算。**

节点上两个按钮：

- **清空缓存**：删文本编码缓存 + batch 临时文件 + 旧版 `*_frames_ht.pt`，保留已渲染片段。
- **清空节点所有缓存**：额外删除所有 `seg_*`（一采）文件，强制全部重算。

两者都会先弹确认框（显示 workflow id / node id / 待删项），再 POST `/minimax/director_opt/clear_cache`（`clear_all: true/false`）。

> 缓存按工作流名分目录。隐藏控件 `workflow_name` 为空时缓存会落到裸 `node_<id>/` 目录，可能导致「对齐下段」变灰、导出为空 —— 该字段由前端自动填写，不要手改。

---

## 导演包导入 / 导出

`.mmxpack.zip`：把时间轴 JSON + 参考图 / 参考视频 / 参考音频 / 源视频打包，磁盘格式与原版 `ComfyUI_MiniMaxH3_Director` 一致，两边可互相导入。

- 导出：`POST /minimax/director_opt/export_pack` → `GET .../download_pack?filename=...`（文件名形如 `MiniMaxH3DirectorOpt-<task>-<时间戳>.mmxpack.zip`；超过 500 MB 会二次确认）。
- 导入：`POST /minimax/director_opt/import_pack`（大文件走 8 MiB 分块上传）。

---

## HTTP 接口与 WebSocket 事件

路由前缀：`/minimax/director_opt`（`lib/constants.py`）。

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/upload_chunk` | 大文件分块上传（8 MiB） |
| POST | `/probe_video` | 探测源视频 fps / 帧数 / 宽高 / 时长 |
| GET | `/list_input_media` | 列 `input` 目录素材（可 `includeCache=1` 带出已生成片段） |
| POST | `/detect_shots` | 智能分割 |
| POST | `/prepare_reference_audio_chunk` / `/extract_reference_audio` | 参考音频分块上传与抽音 |
| POST | `/audio_extract_status` / `/audio_extract` / `/audio_extract_list` / `/audio_extract_remove` | 提取音频增删改查 |
| GET | `/audio_extract_file` | 下载提取出的音频 |
| POST | `/segment_export_status` | 分段导出可用性 |
| POST | `/segment_export` | 分段导出 |
| GET | `/segment_clip` | 流式返回某段成片 |
| POST | `/second_sample_status` | 二采可用性 |
| POST | `/align_to_next_status` | 「对齐下段」可用性 |
| POST | `/remove_segment_slot` | 丢弃某段的缓存文件 |
| POST | `/clear_cache` | 清空缓存 |
| POST | `/export_pack` / `/import_pack`，GET `/download_pack` | 导演包 |

WebSocket 事件：

| 事件 | 载荷要点 |
|---|---|
| `minimax_director_opt_progress` | `node_id`, `segment`, `segment_total`, `phase`（`prepare` / `context_encode` / `sample` / `decode`）, `phase_value`, `overall_value` … |
| `minimax_director_opt_preview` | `node_id`, `segment_index`, `image_b64`, `width`, `height`, 可选 `step` / `total_steps` |

---

## 示例工作流

`example_workflows/` 下按任务提供了可直接打开的工作流，详见 [`example_workflows/README.md`](example_workflows/README.md)。

---

## 常见限制与排错

| 现象 / 限制 | 说明 |
|---|---|
| 单个扩散片段最多 512 帧 | 帧数必须落在 `17k+5` 网格上，最小 5（124 ≈ 5 s @ 24 fps） |
| 视频时间轴最短片段 4 帧，批量 / fl2v 最短 5 帧 | 前端常量 |
| 参考图 ≤ 9、参考视频 ≤ 3、参考音频 ≤ 3 | MiniMax H3 `ReferenceToVideo` 上限 |
| 均分段数 2–64；画布对齐 32；帧率 1–240 | 前端常量 |
| 上传 ≤ 95 MiB 走 `/upload/image`，更大按 8 MiB 分块 | `core/upload.js` |
| 段间引导只支持 5 / 22 / 39 / 56 帧上下文，且需 ≥ 2 段 | `h3_motion_context.py` |
| `use_sigmas` 开了但没接线 | 日志提示并回退默认采样 |
| 二采没接 `upscale_model` | 直接返回错误报告，不会静默出原分辨率 |
| 旧工作流打开后控件值错位 | 前端会剥除废弃输出 `segment_images`、修正失效控件值；时间轴宽高 / 总帧数异常时后端还会夹回安全默认值并告警，重新保存一次即可固化 |
| `/minimax/director_opt/*` 返回 404 | 说明 PromptServer 当时未就绪，重启 ComfyUI |
| 依赖 `import cv2` 失败 | 刻意不依赖 OpenCV；源视频解码走 PyAV |
| 画面偏色 / 反色 | 不要把 `MiniMax H3 Fast Video VAE Opt` 的输出再套非 H3 的 VAE 处理；该节点刻意不做 ComfyUI 的 `*2-1` 变换 |

---

## 许可与署名

Apache License 2.0，详见 `LICENSE`。

- 上游项目：`AIMixer/ComfyUI_MiniMaxH3_Director`（作者：AI搅拌手），基于 ComfyUI 官方 MiniMax H3 支持（PR #15224 / #15228）。
- 本 Opt 分支：**fvdfggh** —— <https://github.com/fvdfggh/minimax-h3-director-performance-optimization>
