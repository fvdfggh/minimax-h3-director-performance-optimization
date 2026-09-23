# MiniMax H3 Director — 二级采样（二采）改造计划

> 基于 git 历史中被删除的原二采子系统（`2acda19` / `9f445ab`，共删 3451 行）重新设计，适配当前版本架构。

---

## 一、需求与已确认决策

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 执行入口 | 按钮只负责**选段**；只有点击弹窗内的**「执行」按钮**才真正执行二采，**不与节点外部的运行混用** |
| 2 | 缓存 | 二采缓存与一采**并行独立**存放（`seg2_*`），互不覆盖、可独立清理 |
| 3 | 参数 | 除二级多选外，**再增加一个种子（seed）参数**；种子位于**节点外面板**的二级采样模块内（**不在弹窗里**） |
| 4 | 出片 | 二采结果**直接出片**；出片逻辑与分段导出的**连续导出**一致，**只有相邻的段才合并** |

---

## 二、现状关键代码（改造锚点）

| 模块 | 位置 | 说明 |
|---|---|---|
| 主节点定义 | `nodes/director.py:120-252` | `INPUT_TYPES`；`bd_grp_advanced`（BDGROUP「高级采样」）在 190 行 |
| widget 顺序约束 | `nodes/director.py:223-225` | **新 widget 必须追加到 optional 末尾**，否则旧工作流 `widgets_values` 错位 |
| run_model | `nodes/director.py:28-47` | `RUN_MODEL_CHOICES` = 主模型 / model_b / model_c；`resolve_run_model()` |
| 放大模型节点 | `nodes/latent_upscaler_3d.py` | `MiniMaxH3LatentUpscaleModel`，输出 `LATENT_UPSCALE_MODEL`，**可调用对象** |
| 分段导出可用性 | `director/segment_cache.py:1047-1114` | `segment_export_availability()` / `inspect_segment_export_status()` |
| 分段导出执行 | `director/segment_cache.py:1751+` | `run_segment_export(mode="continuous"\|"piecewise", vae=...)` |
| 连续导出的相邻合并 | `director/segment_cache.py:1764-1767` | 相邻勾选段合并为 `seg_AA-BB`；孤立段单独导出 |
| 导出目录 | `director/segment_mp4_export.py:29+` | `new_segment_mp4_run_dir()` |
| 路由注册 | `director/http_routes.py:851-868` | `segment_export_status` / `segment_export` |
| 缓存布局 | `director/cache_layout.py:11-23` | 一采文件前缀表；slot map `segment_slots.json` |
| 采样核心 | `director/core_sampling.py:63+` | `sample_single_stage(..., sigmas=, denoise=, apply_shift=)` |
| 前端工具栏 | `web/js/minimax_timeline.js:2378-2384` | `data-a="seg-export"` 分段导出按钮 |
| 弹窗样式 | `web/js/minimax_timeline.js:612-631` | `.bd-seg-export-*`（item / cb / badge / disabled / toast） |
| 引用上段 | `web/js/minimax_timeline.js:2218-2219` | `continuityFromPrev`（默认 true），即"引用上段"连接状态 |
| 弹窗配置读写 | `web/js/minimax_timeline.js:3808-3850` | `_segmentExportConfig()` / `_segmentExportPayload()` / status 拉取 |

### 原二采（已删除，供逻辑参考）

| 文件 | 行数 | 作用 |
|---|---|---|
| `director/refine_sampling.py` | 629 | `apply_segment_refine()`：放大 → `sample_single_stage(sigmas=..., denoise=1.0, apply_shift=True)`；接缝 re-pin |
| `director/refine_pack.py` | 515 | `pack_refine` / `refine_passes_for` / `refine_seed_for` / 画布解析 |
| `nodes/director_refine.py` | 303 | `MiniMaxH3DirectorRefine` 节点 |
| `director/h3_latent_upscale.py` | 414 | latent 放大桥接 |

原二采 AV 拆分/合并做法（`refine_sampling.py:67-104`）：
```python
def _split_av(samples):      # LTXVSeparateAVLatent → (video_latent, audio_latent)
def _join_av(v, a, t):       # LTXVConcatAVLatent
```

---

## 三、总体架构

### 3.1 执行时序（关键：模型只在节点执行期存在）

HTTP 请求中**没有** UNET / VAE / CLIP / 放大模型（`http_routes.py:792-795` 注释已明确）。因此二采**不能**像分段导出那样纯磁盘执行。

方案：弹窗「执行」按钮内部触发一次 **queue prompt**，节点 `execute()` 检测到二采模式后**独占分支**运行——既不跑一采，也不走正常生成，满足"不与外面混用"。

```
[分段导出按钮] [二次采样按钮]            ← 工具栏
        │
        └─ 点击 → 拉取 /second_sample_status
                  ↓
            弹出二级多选弹窗
              ├─ 一级：按「引用上段」聚合的组（可整组全选）
              ├─ 二级：组内段 checkbox（badge 显示缓存可用性，无缓存则禁用）
              └─ [取消] [执行]
                       │
                       └─ 写入 timeline.output.secondSample = {enabled, indices}
                          （种子不在此写入，直接读节点面板的 second_seed widget）
                          → app.queuePrompt()
                               ↓
                          节点 execute()
                               ├─ secondSample.enabled? ── 否 → 正常流程（不变）
                               └─ 是 → 【二采独占分支】
                                    逐段：读文本缓存 + latent 缓存
                                          → 拆 AV → 放大 → 合 AV
                                          → 采样（强制 sigmas）
                                          → VAE 解码
                                          → 写 seg2_* 新缓存
                                    → 连续导出出片（仅相邻合并）
                                    → 直接出片返回
```

### 3.2 二采分支内部流程（单段）

```
load_conditioning_cache(...)        # 复用一采文本编码，不重编码
        ↓
load_segment_av_latent(...)         # 按 slot map 定位一采 latent
        ↓
_split_av()                         # LTXVSeparateAVLatent → video + audio
        ↓
upscale_model(video_latent)         # 接入的 MiniMaxH3LatentUpscaleModel
        ↓
_join_av()                          # LTXVConcatAVLatent
        ↓
sample_single_stage(
    model  = resolve_run_model(second_run_model),
    seed   = second_seed,          # 来自节点面板 widget
    sigmas = <强制，未接线直接报错>,
    denoise = 1.0, apply_shift = True,
    sampler = 高级采样里的 sampler,
)
        ↓
接缝处理（re-pin 前缀帧 / fl2v keep_existing_keyframes）
        ↓
VAE 解码 → 写 seg2_* 新缓存
```

---

## 四、详细改造方案

### 4.1 节点输入端 — `nodes/director.py`

**① 新增放大模型连接口**（非 widget 类型，**不占 `widgets_values` 下标，安全**）：

```python
"upscale_model": ("LATENT_UPSCALE_MODEL", {
    "tooltip": "接 MiniMax H3 Latent Upscaler (3D) [Model] 节点，用于二级采样前的 latent 放大。",
}),
```

放在 optional 中**非 widget 区域**（紧邻 `model_b` / `model_c` 之后），不影响任何 widget 下标。

**② optional 最末尾追加二级采样模块**（必须末尾，见 223-225 行约束）：

```python
"bd_grp_second": ("BDGROUP", {"default": "二级采样"}),
# 种子就在这个「二级采样」模块里（节点外面板 widget），弹窗内不再提供种子输入
"second_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                        "tooltip": "二级采样随机数种子。"}),
"second_run_model": (list(RUN_MODEL_CHOICES), {
    "default": RUN_MODEL_MAIN,
    "tooltip": "二级采样用哪个 MODEL 口（主模型 / model_b / model_c）；未接线回退主模型。",
}),
```

- `second_run_model` 复用 `RUN_MODEL_CHOICES` + `resolve_run_model()`。
- `second_seed` 就是**种子输入本体**，位于节点外面板的「二级采样」模块内；加 `control_after_generate: True` 以获得 ComfyUI 标准的种子行为（固定 / 随机 / 递增 / 递减）。运行时直接读该 widget 值，弹窗不参与。
- 弹窗只负责**选段 + 执行**，不承载任何采样参数。

### 4.2 缓存 — 并行新一套（**沿用哈希命名 + 纳入顺序管理**）

> **硬性约定**：二采缓存**不改命名规则**——文件名仍由**内容哈希**决定，与时间线位置解耦；位置→文件的对应关系**一律走 slot map（顺序管理）**。不得退化成按 index 拼名，否则时间线增删/重排后必然对不上。

**① 命名：内容哈希（与一采同规则）**

`<hash>` = 段内容指纹，复用 `segment_cache_fingerprint(seg, plan)`（prompt + 参考 + 时长 + 采样），并**追加二采专属参数**（seed / 放大模型名 / sigmas / run_model）参与哈希，避免不同二采配置互相命中。内容重复时按现规则追加 `_1`、`_2` 后缀（`cache_layout.py:25-27`）。

**② 前缀：`director/cache_layout.py` 新增常量**

| 类型 | 一采 | 二采（新增） |
|---|---|---|
| latent | `seg_<hash>_latent.pt` | `seg2_<hash>_latent.pt` |
| 解码帧 pt | `seg_<hash>_frames.pt` | `seg2_<hash>_frames.pt` |
| 头尾帧 | `seg_<hash>_frames_ht.pt` | `seg2_<hash>_frames_ht.pt` |
| 音频 latent | `seg_<hash>_audio.pt` | `seg2_<hash>_audio.pt` |
| mp4 | `seg_<hash>_clip.mp4` | `seg2_<hash>_clip.mp4` |
| meta | `seg_<hash>_meta.json` | `seg2_<hash>_meta.json` |
| slot map | `segment_slots.json` | `segment_slots_2nd.json` |

**③ 顺序管理：`director/segment_slots.py`**

给 `slot_paths()` / slot map 路径增加 `variant` 参数（默认 `"1st"`），二采用 `variant="2nd"` → 落到 `segment_slots_2nd.json`。要点：

- 二采 slot map 与一采**共用同一套 position 语义**（同一时间线槽位），因此某段的二采结果始终能在**同一 position** 上被查到。
- 二采写缓存时**必须同时把该文件组登记进二采 slot map 的对应 position**，不能只落盘不登记。
- **`sync_segment_slots()` 需同时同步两个 variant**（或在其中传 `variant` 遍历），保证时间线编辑（删中间组、重排）后一采/二采的槽位**同步收敛**，不出现一边新一边旧。
- 读取一律经 `_slot_paths(variant="2nd")` 解析，并复用现有的 **stale / prev 上一代回落**（`segment_slots.py:252-257`），与 `segment_export_availability()` 的容错一致。

**④ `director/segment_cache.py` 新增**

- `save_second_pass_cache()` / `load_second_pass_cache()` / `probe_second_cache_shape()`
- 二采 meta 中**记录所依赖的一采指纹**，用于校验"二采结果是否仍对应当前一采"，失配则标记 `stale` 并在 UI 上提示重跑

**⑤ 防对不上的关键校验**

| 风险 | 处置 |
|---|---|
| 时间线重排/插入/删除后位置漂移 | 全程走 slot map，不以 index 拼名；`sync_segment_slots` 双 variant 同步 |
| 一采重跑后二采结果过期 | 二采 meta 存一采指纹，比对失配即 `stale` |
| 二采参数变了却命中旧缓存 | seed / 放大模型 / sigmas / run_model 全部纳入 hash |
| 一采与二采 slot map 版本不一致 | 两者在同一次 `sync_segment_slots` 中收敛，禁止各自独立更新 |

### 4.3 后端路由与可用性判断

**新增路由**（`director/http_routes.py`，照抄 `minimax_segment_export_status` 骨架，851-868 行附近注册）：

```
POST /minimax/director/second_sample_status
```

处理函数流程：`build_director_plan(...)` → `sync_segment_slots(...)` → `inspect_second_sample_status(...)`。

**新增 `inspect_second_sample_status()` / `second_sample_availability()`**（`segment_cache.py`，照抄 `segment_export_availability()` 1047-1098 行结构），判定改为：

| 字段 | 判定 |
|---|---|
| `hasLatent` | `_slot_paths()["latent"].is_file()`（含 stale/prev 上一代回落） |
| `hasTextCond` | `cond_text_<hash>.pt` 存在（`conditioning_cache.text_cache_key` + `cache_layout` 路径） |
| **`canSecondSample`** | **`hasLatent AND hasTextCond`**（不含 clip，与分段导出不同） |
| `continuityFromPrev` | 透传给前端用于分组（或前端直接用 timeline 数据） |

> 注：`/second_sample` 执行路由**不需要**——执行由弹窗「执行」按钮走 queue prompt 触发。

### 4.4 执行链路 — 新增 `director/second_sampling.py`

新增主函数 `run_second_sample(...)`，在节点 `execute()` 中被二采分支调用：

```python
def run_second_sample(
    *, plan, node_id, workflow_name,
    indices: list[int],
    seed: int,
    model, video_vae, audio_vae,      # 节点输入，仅执行期可得
    upscale_model,                     # LATENT_UPSCALE_MODEL 可调用对象
    sigmas,                            # 强制，None 则抛错
    sampler, shift_video, shift_audio,
    ...
) -> dict
```

内部逐段执行第三章 3.2 流程，并：
- **强制 sigmas**：`sigmas` 为空 → 直接 `raise`，**绝不回退**到 steps/scheduler（与一采的自动回退行为相反）
- **接缝**：复用原版 `apply_segment_refine` 的 re-pin 前缀帧逻辑（`pin_frames`）与 `keep_existing_keyframes`（fl2v 专用）
- 单段失败不中断整体，记入 `skipped`（与 `run_segment_export` 一致的容错策略）

**新文件**：`director/second_sampling.py`

### 4.5 出片 — 复用连续导出（仅相邻合并）

二采结果**直接出片**，复用 `run_segment_export(..., mode="continuous")` 的相邻合并语义（`segment_cache.py:1764-1767`）：
- 时间线**相邻**的勾选段 → 合并为单个 `seg2_AA-BB.mp4`
- 无相邻勾选邻居的孤立段 → 单独导出
- 稀疏选择 → 混合输出

实施上新增 `run_second_sample_export()`（或给 `run_segment_export` 增加 `variant="2nd"` 参数），使其：
- 数据源指向 `seg2_*` 缓存
- 传入 `vae=video_vae`（`run_segment_export` 已支持 `vae` 参数，可解码 latent）
- 输出目录用 `new_segment_mp4_run_dir()` 的二采变体

### 4.6 前端 — `web/js/minimax_timeline.js`

**① 工具栏按钮**（2378-2384 行区域）：在 `data-a="seg-export"` 右侧插入

```html
<button type="button" class="bd-btn" data-a="second-sample"
        data-i18n="toolbar.secondSample"
        data-i18n-title="tooltip.secondSample">二次采样</button>
```

**② 弹窗**（复用 `.bd-seg-export-*` 样式，612-631 行）：

| 区域 | 内容 |
|---|---|
| 一级（组） | 按 `continuityFromPrev`（2218-2219 行，默认 true）把存在引用关系的**连续片段归为一组**；组头 checkbox 可整组全选/全不选 |
| 二级（段） | 组内每段一个 checkbox + badge（`latent缓存` / `文本缓存` / `可二采` / `不可用`） |
| 禁用 | `canSecondSample=false` → `.disabled` 且不可勾选 |
| 底部 | `[取消] [执行]` |

> **弹窗内不含种子输入** —— 种子是节点外面板「二级采样」模块里的 `second_seed` widget，运行时直接读取该 widget 值。

**③ 执行**：点击「执行」→
1. 写入 `timeline.output.secondSample = { enabled: true, indices }`
2. 调用 `app.queuePrompt()` 触发运行
3. 运行结束 toast 提示（复用 `.bd-seg-export-toast`）

**④ 序列化**：新增 `_secondSamplePayload()`，仿照 `_segmentExportPayload()` 随 timeline 一起提交（在 `buildPayload` 各分支中挂载，参考 2163 / 2226 / 2257 / 2318 行）。

**⑤ i18n**：`minimax_i18n.js` 新增 `toolbar.secondSample` / `tooltip.secondSample` 等键（现已有完善的多语言机制）。

---

## 五、文件改动清单

| 文件 | 改动类型 | 内容 | 状态 |
|---|---|---|---|
| `nodes/director.py` | 修改 | 增 `upscale_model` 输入；末尾增 `bd_grp_second` / `second_seed` / `second_run_model`；`execute()` 增二采分支 | 🔶（输入/参数完成，`execute()` 分支随第 6 步接线） |
| `director/second_sampling.py` | **新增** | `run_second_sampling()` + `export_second_pass()` 主执行链路（放大 + 采样 + 接缝 + 连续出片） | ✅ |
| `director/segment_cache.py` | 修改 | 增 `second_sample_availability()` / `inspect_second_sample_status()` / `save_second_pass_cache()` / `load_second_pass_av_latent()` / `resolve_second_stem()` / `sync_second_segment_slots()` | ✅ |
| `director/cache_layout.py` | 修改 | 增 `seg2_*` 前缀常量（`SECOND_PREFIX` / `SEGMENT_GLOBS`）+ `segment_slots_2nd.json` | ✅ |
| `director/segment_slots.py` | 修改 | 全模块 `variant` 机制（`VARIANT_SECOND` / `manifest_path` / `content_stem` / `sync_slots` / `resolve_stem` / `slot_paths` / `gc_orphan_files` 双前缀隔离等） | ✅ |
| `director/http_routes.py` | 修改 | 增 `minimax_second_sample_status` 及路由注册；clear_all 同时清 `seg2_*` + 二采 map | ✅ |
| `director/conditioning_cache.py` | 修改 | 段→text_key/上下文参数映射（`save_segment_second_params` / `load_segment_second_params`），二采免重算 ref 哈希 | ✅ |
| `director/batch_executor.py` | 修改 | 一采跑每段时持久化 second-params 映射 | ✅ |
| `nodes/director.py` | 修改 | 增 `upscale_model` 输入；末尾增 `bd_grp_second` / `second_seed` / `second_run_model`；`execute()` 增二采独占分支 | ✅ |
| `nodes/director_common.py` | 修改 | 增 `_parse_second_sample` / `_attach_second_sample`，在 plan 构建处挂载 | ✅ |
| `director/plan.py` | 修改 | 增 `SegmentSecondSampleRequest` dataclass + `second_sample` 字段 + `_parse_second_sample` | ✅ |
| `web/js/minimax_timeline.js` | 修改 | 增「二次采样」按钮、弹窗（二级多选 + 执行，无种子输入）、status 拉取、queuePrompt | ✅ |
| `web/js/minimax_i18n.js` | 修改 | 新增 i18n 键 | ⬜（未改，弹窗用中文硬编码） |
| `director/segment_mp4_export.py` | 修改 | 二采导出目录 / variant 支持 | ⬜（未改，复用现有 `_write_export_mp4`） |

---

## 六、风险与注意事项

1. **widget 下标（最高优先级）**：`second_seed` / `second_run_model` **必须**追加在 optional 最末尾；`upscale_model` 是非 widget 类型，不占下标但仍建议放在非 widget 区域。
2. **强制 sigmas**：二采不得回退。若 `sigmas` 未接线或解析失败，直接抛错并给出明确提示。
3. **旧工作流兼容**：新增 widget 后旧工作流的 `widgets_values` 长度不足，ComfyUI 会用默认值补齐——追加在末尾是安全方向，切勿前插。
4. **显存**：放大 + 重采样 + VAE 解码三重叠加，建议沿用 `vram_cleanup` / 模型卸载策略；放大模型已有 `force_unload` 开关。
5. **缓存指纹**：二采缓存的 hash 必须包含 seed、放大模型名、sigmas、run_model，否则不同配置会错误命中同一份缓存。
6. **缓存顺序管理（"对不上"的主要来源）**：文件名一律用内容哈希，**位置映射一律走 slot map**，严禁按 index 拼名；写缓存必须同时登记进 `segment_slots_2nd.json`；`sync_segment_slots()` 必须**同时同步一采/二采两个 variant**，否则时间线编辑后两边槽位会漂移。
7. **二采结果过期**：二采 meta 记录一采指纹，一采重跑后比对失配即标记 `stale` 并提示重跑，避免拿旧二采结果拼接。
8. **AV 拆分/合并**：必须走 `LTXVSeparateAVLatent` / `LTXVConcatAVLatent`（原版做法），否则音频通道会被破坏。
9. **执行独占性**：二采模式下 `execute()` 不得同时跑一采，避免缓存与输出混淆。

---

## 七、建议实施顺序（完成进度）

- [x] 1. `cache_layout.py` + `segment_slots.py` — 并行缓存骨架（variant 机制）**已完成**
- [x] 2. `nodes/director.py` — 输入端与参数（widget 顺序验证）**已完成**（`execute()` 分支已接线）
- [x] 3. `director/segment_cache.py` — 可用性判断 + 二采缓存读写 **已完成**
- [x] 4. `director/http_routes.py` — status 路由 **已完成**（清理也覆盖 seg2）
- [x] 5. `web/js/minimax_timeline.js` — 工具栏「二次采样」按钮（分段导出右侧）+ 二级多选弹窗（按「引用上段」分两组）+「执行」走 queuePrompt **已完成**（弹窗文案用中文硬编码，未新增 i18n 键）
- [x] 6. `director/second_sampling.py` — 执行链路（拆/放大/合 AV + `sample_single_stage` + 接缝复用 `apply_motion_context`）+ `nodes/director.py` `execute()` 二采独占分支 **已完成**
- [x] 7. 出片 — `export_second_pass()` 复用 `continuous_export_runs` + `merge_run_audio` + `_write_export_mp4`，仅相邻合并 **已完成**

> 实施备注：
> - **sigmas**：二采没有独立 sigmas 输入（模块只有 seed + run_model），故复用一采的 `use_sigmas` + `sigmas`（即节点面板的 SIGMAS）；未接线时 `normalize_sigmas` 返回 `None`，执行器仅 `warning` 并回退 steps/scheduler，不会抛错中断。这与原计划 4.4「为空即 raise」有偏差——改为容错回退，避免用户未开 use_sigmas 时完全无法二采。
> - **出片目录**：直接复用现有 `_write_export_mp4`（落 `minimax_second_pass_export/<node_id>/`），未改动 `segment_mp4_export.py`，故该文件清单项标记为未改。
> - **前端 i18n**：`minimax_i18n.js` 未改动，弹窗标题/提示用中文硬编码。
