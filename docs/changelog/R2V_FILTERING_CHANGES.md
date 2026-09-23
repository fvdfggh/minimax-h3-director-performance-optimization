# ComfyUI MiniMax H3 Director - R2V 素材过滤改动

## 改动概述

针对 `ref2v` 模式（Reference-to-Video）进行了优化：**只有在prompt中明确引用到的素材才会参与编码**，其他素材不进行处理。

这样可以显著减少显存占用和编码时间，特别是当一个素材组中有很多图片，但只使用其中少数几张时。

## 改动范围

修改了两个关键文件：
1. `director/batch_executor.py` - 批处理执行器
2. `director/executor_core.py` - 核心执行器

## 实现方式

### 1. 新增辅助函数

在两个文件中都添加了两个新的辅助函数：

#### `_extract_referenced_picture_indices(prompt: str) -> set[int]`
- **功能**: 从prompt中提取所有被引用的图片索引
- **支持的格式**:
  - 英文：`<Picture 1>`, `<Picture 2>` 等
  - 中文：`图片1`, `图片2` 等（支持空格变体 `图片 1`）
- **返回**: 0-based索引集合（`<Picture 1>` → `{0}`）

#### `_filter_ref_images_by_prompt(ref_images, prompt) -> dict | None`
- **功能**: 过滤ref_images字典，仅保留在prompt中引用的素材
- **输入**: `{"ref_image_0": tensor, "ref_image_1": tensor, ...}`
- **输出**: 过滤后的字典，或如果没有引用则返回None
- **特性**: 
  - 如果prompt中没有找到任何引用，出于向后兼容性保留所有素材
  - 支持部分引用（如只在prompt中引用了图片1和图片3）

### 2. 应用过滤

#### batch_executor.py
在 `prepare_segment_materials()` 函数中：
- 对R2V、V2V、RV2V任务类型应用过滤
- 在处理`ref_images`后立即过滤
- 记录过滤前后的数量用于调试

#### executor_core.py
在 `_build_minimax_inputs()` 函数中：
- 对R2V任务：在构建`ref_images`后过滤
- 对RV2V任务：在构建`ref_images`后过滤
- 每次过滤时记录日志

## 向后兼容性

- **默认行为**: 如果prompt中没有明确的图片引用标签（`<Picture N>` 或 `图片N`），则保留所有素材（不过滤）
- **现有工作流**: 不需要任何修改就能继续工作
- **新工作流**: 可以通过在prompt中添加`<Picture>`标签来利用这个优化

## 使用示例

### 场景1：有5张图，只用第1张和第3张

**Prompt:**
```
In the style of <Picture 1>, show a person transforming. 
The final result should match <Picture 3>.
```

**结果**:
- 只有 `ref_image_0` 和 `ref_image_2` 会被编码
- `ref_image_1`, `ref_image_3`, `ref_image_4` 跳过
- 节省约60%的编码时间和显存

### 场景2：中文prompt

**Prompt:**
```
参考图片1中的服装风格，将视频中的人物替换为图片2中的形象。
```

**结果**:
- 只有 `ref_image_0` 和 `ref_image_1` 会被编码
- 其他素材不处理

### 场景3：没有明确引用（向后兼容）

**Prompt:**
```
A person walking in the forest with beautiful lighting
```

**结果**:
- 所有连接的ref_images都会被编码（保持原有行为）

## 性能影响

根据素材组大小和实际引用数量：
- **最佳情况**: 大素材组（9张图）但只引用2张 → **编码时间减少70-80%**
- **典型情况**: 中等素材组（5张图）引用3张 → **编码时间减少40-50%**
- **无优化场景**: 所有图都有引用 → **无性能变化**

## 测试覆盖

已验证的场景：
- ✓ 单个英文引用 (`<Picture 1>`)
- ✓ 多个英文引用 (`<Picture 1>` and `<Picture 3>`)
- ✓ 单个中文引用 (`图片1`)
- ✓ 多个中文引用 (`图片1和图片3`)
- ✓ 混合引用 (`<Picture 2>` 和 `图片4`)
- ✓ 中文空格变体 (`图片 1 和 图片 3`)
- ✓ 无引用（保持原素材）
- ✓ 超出范围的引用（返回None）

## 日志输出

在调试日志中可以看到过滤过程：
```
DEBUG: R2V ref_images filtered by prompt: 5 → 2 items
DEBUG: Seg #1 R2V ref_images filtered by prompt: 5 → 2 items
```

## 注意事项

1. **Prompt中的拼写**: 确保使用正确的标签格式：
   - 英文：`<Picture 1>` （注意大小写和空格）
   - 中文：`图片1` （支持 `图片 1` 的空格变体）

2. **索引映射**: 
   - `<Picture 1>` 对应 `ref_image_0`
   - `<Picture 2>` 对应 `ref_image_1`
   - 以此类推（1-based在prompt，0-based在code）

3. **向后兼容**: 旧的工作流和没有明确引用的prompt会保持原有行为，无需修改

## 如何禁用此功能

如果需要禁用自动过滤（回到原有行为），可以：
1. 注释掉 `prepare_segment_materials()` 中的过滤调用
2. 或始终在prompt中不使用 `<Picture N>` 标签

---

**修改时间**: 2024年
**兼容性**: 完全向后兼容，无破坏性改动
