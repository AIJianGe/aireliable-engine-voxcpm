# VoxCPM TTS 引擎使用指南

> 引擎 ID：`voxcpm_tts_c4a8f2e7`
> 引擎名称：VoxCPM-TTS
> 引擎类型：voice / text-to-speech
> 范式：integrated（全自定义模式，继承 `CustomEngine`）
> 基于：[OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 官方 Python 库
> 端口：7003（L1 层 voice 系列首位）

---

## 一、简介

VoxCPM 是 OpenBMB 开源的多语言文本转语音（TTS）模型，本引擎将其封装为 AIReliable 兼容的 `integrated` 范式引擎，提供四种合成模式：

| 模式 | 说明 | 适用场景 |
|---|---|---|
| **TTS** | 输入文本 + 说话人音色描述 → 生成语音 | 常规 TTS，无需参考音频 |
| **Voice Design** | 输入文本 + 创造性音色参数 → 生成语音 | 创造性音色生成 |
| **Controllable Cloning** | 上传参考音频 → 克隆音色合成 | 已有目标音色，快速克隆 |
| **Ultimate Cloning** | 上传参考音频 + 转写文本 → 最高保真克隆 | 专业级音色克隆 |

---

## 二、模型放哪里？

### 2.1 放置目录

所有 VoxCPM 模型文件统一放到项目根目录下：

```
models/voice/tts/voxcpm/
├── voxcpm2/              ← VoxCPM2 主模型（48kHz，多语言，约 8GB）
│   ├── model_index.json
│   ├── loc_encoder/
│   ├── tslm/
│   ├── loc_dit/
│   └── audio_vae_v2/
└── voxcpm-0.5b/          ← VoxCPM-0.5B 轻量模型（16kHz，中英，约 1GB）
    ├── model_index.json
    └── ...
```

**Windows 完整路径示例**：
```
D:\myProInstallDir\AIReliable\models\voice\tts\voxcpm\voxcpm2\model_index.json
```

**Linux 完整路径示例**：
```
/opt/aireliable/models/voice/tts/voxcpm/voxcpm2/model_index.json
```

> ⚠️ 不要放到其他目录（如 `~/.cache/huggingface/`）。引擎只扫描 `models/voice/tts/voxcpm/`，放错位置会导致启动按钮变灰。

### 2.2 模型下载

VoxCPM 模型为 HuggingFace diffusers 目录格式（含 `model_index.json` + 多个 `*.safetensors` 子文件），需用 `snapshot_download()` 下载整个 repo。

**方式 1：通过 AIReliable UI 下载**

引擎启动时检测到模型缺失，会显示下载指引（URL / 镜像 / 目标路径），点击「下载」按钮自动拉取。

**方式 2：手动下载（推荐国内用户）**

```bash
# 设置 HF 镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com

# 下载 VoxCPM2 主模型（约 8GB）
huggingface-cli download OpenBMB/VoxCPM2 \
    --local-dir models/voice/tts/voxcpm/voxcpm2 \
    --local-dir-use-symlinks False

# 下载 VoxCPM-0.5B 轻量模型（约 1GB）
huggingface-cli download OpenBMB/VoxCPM-0.5B \
    --local-dir models/voice/tts/voxcpm/voxcpm-0.5b \
    --local-dir-use-symlinks False
```

**Windows PowerShell**：
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"

huggingface-cli download OpenBMB/VoxCPM2 `
    --local-dir models\voice\tts\voxcpm\voxcpm2 `
    --local-dir-use-symlinks False
```

### 2.3 支持的模型变体

| variant_id | alias | 采样率 | 支持语言 | 显存需求 | 来源 |
|---|---|---|---|---|---|
| `voxcpm2` | VoxCPM2 | 48 kHz | 中/英/日/韩/法/德等 | 8GB | [OpenBMB/VoxCPM2](https://huggingface.co/OpenBMB/VoxCPM2) |
| `voxcpm-0.5b` | VoxCPM-0.5B | 16 kHz | 中/英 | 5GB | [OpenBMB/VoxCPM-0.5B](https://huggingface.co/OpenBMB/VoxCPM-0.5B) |

---

## 三、运行

### 3.1 通过 AIReliable 主壳启动（推荐）

1. 打开 AIReliable 桌面客户端
2. 进入「引擎管理」页面
3. 找到 VoxCPM-TTS 引擎卡片
4. 点击「启动」按钮
5. 状态变为 Ready 后，点击「打开」按钮通过 iframe 加载 Gradio UI

### 3.2 命令行启动（开发调试）

```bash
# Windows / Linux / macOS
uv run --no-sync --project envs/voice-voxcpm python -m engines.voxcpm_tts_c4a8f2e7.server --port 7003
```

启动后访问：
- Gradio UI：http://127.0.0.1:7003/
- API 端点：http://127.0.0.1:7003/health、/capabilities、/infer 等

### 3.3 环境准备

首次运行需创建 uv 环境：

```bash
cd envs/voice-voxcpm
uv sync
```

> SDK 采用 editable 模式，修改 `sdk/aireliable-engine-sdk/` 源码后无需重新安装，重启引擎即可生效。

---

## 四、四种合成模式使用说明

### 4.1 TTS 模式（默认）

输入文本 + 说话人音色描述，生成语音。

**UI 操作**：
1. 切换到「TTS」Tab
2. 输入文本（如 "你好，欢迎使用 VoxCPM 语音合成引擎"）
3. 输入音色描述（如 "A female speaker with a warm and gentle voice"）
4. 调整高级参数（可选）：
   - `cfg_value`：CFG 引导强度（默认 2.0）
   - `inference_timesteps`：推理步数（默认 10）
   - `seed`：随机种子（留空则随机）
5. 点击「生成」按钮
6. 等待合成完成，播放/下载结果

**API 调用**：
```bash
curl -X POST http://127.0.0.1:7003/infer \
  -H "Content-Type: application/json" \
  -d '{
    "variant": "voxcpm2",
    "inputs": {
      "text": "你好，欢迎使用 VoxCPM",
      "mode": "tts",
      "voice_description": "A female speaker with a warm and gentle voice"
    },
    "task_id": "test-tts-001"
  }'
```

### 4.2 Voice Design 模式

输入文本 + 创造性音色参数，生成具有设计感的语音。

**UI 操作**：
1. 切换到「Voice Design」Tab
2. 输入文本
3. 输入创造性音色描述（如 "A young girl with energetic and cheerful voice"）
4. 点击「生成」按钮

### 4.3 Controllable Cloning 模式

上传参考音频，克隆音色合成语音。

**UI 操作**：
1. 切换到「Controllable Cloning」Tab
2. 输入要合成的文本
3. 上传参考音频（拖拽或点击上传，支持 wav/mp3/flac）
4. 点击「生成」按钮

**API 调用**：
```bash
curl -X POST http://127.0.0.1:7003/infer \
  -H "Content-Type: application/json" \
  -d '{
    "variant": "voxcpm2",
    "inputs": {
      "text": "你好，这是克隆的语音",
      "mode": "clone",
      "reference_audio_path": "/path/to/reference.wav"
    },
    "task_id": "test-clone-001"
  }'
```

### 4.4 Ultimate Cloning 模式

上传参考音频 + 转写文本，最高保真克隆。

**UI 操作**：
1. 切换到「Ultimate Cloning」Tab
2. 输入要合成的文本
3. 上传参考音频
4. 输入参考音频的转写文本（prompt_text）
5. 点击「生成」按钮

**API 调用**：
```bash
curl -X POST http://127.0.0.1:7003/infer \
  -H "Content-Type: application/json" \
  -d '{
    "variant": "voxcpm2",
    "inputs": {
      "text": "你好，这是极致克隆的语音",
      "mode": "ultimate",
      "reference_audio_path": "/path/to/reference.wav",
      "prompt_text": "参考音频对应的转写文本"
    },
    "task_id": "test-ultimate-001"
  }'
```

---

## 五、API 端点

引擎实现 8 个标准 HTTP 端点：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/capabilities` | GET | 能力声明（变体列表 / 支持的模式） |
| `/load` | POST | 加载模型（指定 variant） |
| `/unload` | POST | 卸载模型 |
| `/infer` | POST | 推理（返回 SSE 流） |
| `/status` | GET | 状态查询（模型/显存/变体） |
| `/lora/load` | POST | LoRA 加载（VoxCPM 不支持，返回 NotImplementedError） |
| `/lora/unload` | POST | LoRA 卸载（VoxCPM 不支持） |

**SSE 流式进度**：

`/infer` 端点返回 `text/event-stream` 响应，包含以下阶段帧：

| stage | progress | message | 说明 |
|---|---|---|---|
| `start` | 0.0 | 推理开始 | 初始帧 |
| `model_loading` | 0.1 | 加载模型... | 模型加载中 |
| `text_processing` | 0.3 | 处理文本... | 文本归一化 |
| `generating` | 0.5 | 合成音频... | 模型推理中 |
| `saving` | 0.9 | 保存音频... | 写入文件 |
| `completed` | 1.0 | 合成完成 | 最终帧（含 audio_url） |
| `error` | - | 错误信息 | 错误帧 |

**音频输出**：
- 输出目录：`data/files/output/audio/`
- 文件命名：`{task_id}.wav`
- 访问路径：`/output/audio/{task_id}.wav`
- 采样率：48kHz（VoxCPM2）/ 16kHz（VoxCPM-0.5B）
- 格式：wav（PCM 16-bit）

---

## 六、新增 Variant 步骤

### 6.1 下载模型文件

将新的 VoxCPM 模型下载到 `models/voice/tts/voxcpm/<new_variant>/`。

### 6.2 在 manifest.yaml 声明

打开 `engines/voxcpm_tts_c4a8f2e7/manifest.yaml`，在 `variants` 数组追加：

```yaml
models:
  variants:
    - name: <new_variant>
      alias: <显示名>
      base_model: <new_variant>
      precision: fp16
      loader: diffusers
      vram_required: 8GB
      path: voice/tts/voxcpm/<new_variant>
      model_file_ids:
        - <new_variant>-model-index
      uv_env: envs/voice-voxcpm
      default_steps: 10
      default_guidance: 2.0
```

在 `model_files` 数组追加对应文件清单：

```yaml
  model_files:
    - id: <new_variant>-model-index
      role: main
      filename: model_index.json
      target_dir: voice/tts/voxcpm/<new_variant>
      format: json
      source:
          type: huggingface
          repo_id: OpenBMB/<new_repo>
          filename: model_index.json
          url: https://huggingface.co/OpenBMB/<new_repo>/resolve/main/model_index.json
          mirror_url: https://hf-mirror.com/OpenBMB/<new_repo>/resolve/main/model_index.json
      description: "<new_variant> 模型目录标志文件"
```

### 6.3 重启引擎

重启 AIReliable 主壳，引擎会自动扫描新 variant。

---

## 七、依赖环境

### 7.1 uv 环境

- 环境目录：`envs/voice-voxcpm/`
- 依赖文件：`envs/voice-voxcpm/pyproject.toml`
- Python 版本：>=3.10, <3.13

### 7.2 关键依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| `torch` | >=2.5.0 | PyTorch 推理框架（VoxCPM 要求；**默认 CPU 版**，GPU 推理需切换到 CUDA 版，见 §8） |
| `torchaudio` | >=2.5.0 | 音频处理 |
| `torchcodec` | - | VoxCPM 音频编解码 |
| `transformers` | >=4.36.2 | MiniCPM-4 骨干模型 |
| `gradio` | >=4.44.0 | Web UI |
| `fastapi` | >=0.115.0 | API 框架 |
| `soundfile` | - | 音频读写 |
| `librosa` | - | 音频分析 |
| `inflect` | - | 英文数字转文字 |
| `wetext` | - | 中文文本归一化 |
| `funasr` | - | 语音识别（转写/对齐） |

### 7.3 SDK 依赖

```toml
# envs/voice-voxcpm/pyproject.toml
[tool.uv.sources]
aireliable-engine-sdk = { path = "../../sdk/aireliable-engine-sdk", editable = true }
```

> 开发期 SDK 采用 editable 模式，源码修改即时生效，无需重新安装。

---

## 八、设备选择

引擎支持三种设备模式：

| 设备 | 说明 | 适用场景 |
|---|---|---|
| `auto` | 自动选择（优先 cuda:0） | 默认值，推荐 |
| `cpu` | CPU 推理 | 无 GPU 或调试 |
| `cuda` | CUDA 推理 | GPU 加速 |

> 通过 `EngineConfig.device` 字段配置，默认 `auto`。

### 8.1 torch CUDA 版本切换（v0.2.5+ 前置条件）

> **重要**：`device=auto` 或 `device=cuda` 能生效的前提是引擎 `.venv` 安装了 **CUDA 版 torch**（如 `2.13.0+cu128`）。
> 平台默认安装 CPU 版 torch（`2.13.0+cpu`），此时 `torch.cuda.is_available()` 返回 `False`，引擎会降级到 CPU 推理（速度极慢）。

**如何切换到 GPU**：

1. 进入 VoxCPM 引擎详情页
2. 查看设备状态卡片，确认显示 "torch 2.13.0+cpu (CPU 模式)"
3. 点击"切换到 GPU"按钮
4. 在切换对话框中确认 wheel URL 和大小，点击"开始切换"
5. 等待下载完成（观察进度条、速度、剩余时间，支持断点续传和镜像 fallback）
6. 自动安装 + 验证 `torch.cuda.is_available()` + 重启引擎
7. 引擎重启后，设备状态卡片显示 "torch 2.13.0+cu128 (GPU 模式)"
8. 之后启动引擎即可使用 GPU 推理

**相关文档**：
- 架构详解：[docs/docs/architecture/torch-cuda-switch.md](../../docs/docs/architecture/torch-cuda-switch.md)
- 用户指南：[docs/docs/how-to/torch-cuda.md](../../docs/docs/how-to/torch-cuda.md)
- 开发规范：[PR/development-guide.md §36](../../PR/development-guide.md)

> **注**：VoxCPM 引擎存在 device 未应用的已知 bug（`_resolve_device` 结果未传给 `VoxCPM.from_pretrained`），将在 SDK §7 设备选择规范实施时修复。当前即使切换到 CUDA 版 torch，仍需确认 device 透传链路正确。

---

## 九、测试

### 9.1 运行单元测试

```bash
cd d:\myProInstallDir\AIReliable
uv run --no-sync --project envs/voice-voxcpm pytest engines/voxcpm_tts_c4a8f2e7/tests/ -v
```

### 9.2 测试覆盖

| 测试文件 | 覆盖范围 | 用例数 |
|---|---|---|
| `tests/test_engine.py` | 7 个 `_xxx_impl` 方法 + SSE 帧 + 状态机 | 16 |
| `tests/test_voxcpm_wrapper.py` | 模型加载/卸载/参数构造/设备解析 | 29 |
| `tests/test_ui.py` | UI API 调用函数（Mock httpx） | 12 + 1 skipped |

> 测试通过 Mock `VoxCPM.from_pretrained` 和 `torch.cuda.is_available()` 避免真实模型加载。

---

## 十、常见问题

### Q1：启动后 `/capabilities` 返回空 variants 列表？

**原因**：模型文件未放置到正确目录。

**解决**：
1. 检查 `models/voice/tts/voxcpm/voxcpm2/model_index.json` 是否存在
2. 若不存在，按 §2.2 下载模型
3. 重启引擎

### Q2：合成音频有杂音或质量差？

**原因**：可能使用了 VoxCPM-0.5B 轻量模型，或参数配置不当。

**解决**：
1. 切换到 VoxCPM2 主模型（48kHz 高保真）
2. 调整 `cfg_value`（推荐 2.0）和 `inference_timesteps`（推荐 10-32）
3. 克隆模式使用 Ultimate Cloning（最高保真）

### Q3：显存不足？

**原因**：VoxCPM2 需要 8GB 显存。

**解决**：
1. 切换到 VoxCPM-0.5B（仅需 5GB 显存）
2. 关闭其他占用显存的引擎
3. 使用 CPU 模式（`device: cpu`，速度较慢）

### Q4：克隆模式参考音频格式不支持？

**支持格式**：wav / mp3 / flac

**解决**：使用 ffmpeg 转换格式：
```bash
ffmpeg -i input.ogg output.wav
```

---

## 十一、引擎环境设置面板（v0.2.5+ 引入）

> **用户指南**：[docs/docs/how-to/engine-env-management.md](../../docs/docs/how-to/engine-env-management.md)
> **架构文档**：[docs/docs/architecture/engine-env-settings.md](../../docs/docs/architecture/engine-env-settings.md)

### 11.1 VoxCPM 引擎的 env_config 配置

VoxCPM 引擎依赖 torch，`manifest.yaml` 中 `env_config` 配置如下：

```yaml
env_config:
  requires_torch: true
  torch_version_constraint: ">=2.5.0"
  default_cuda_version: "cu128"
  supported_cuda_versions:
    - "cu121"
    - "cu124"
    - "cu126"
    - "cu128"
  notes: "VoxCPM 需要 torch >=2.5.0，建议使用 CUDA 版本以获得 GPU 加速"
```

### 11.2 环境设置面板入口

在 VoxCPM 引擎详情页向下滚动，找到 **环境设置** 卡片（位于"设备状态"卡片下方）。

环境设置卡片包含三个区域：
- **Python 运行时**：`.venv` 状态、Python 版本、版本约束、环境路径
- **核心依赖**：torch 版本、SDK 模式、依赖数量 + "查看列表"按钮 + "切换到 GPU"按钮
- **环境操作**：初始化/重置/更新按钮

### 11.3 常见操作

#### 11.3.1 初始化环境

首次安装 VoxCPM 引擎后，如果 `.venv` 未自动创建：
1. 进入引擎详情页，找到"环境设置"卡片
2. 点击"初始化环境"按钮
3. 确认对话框信息，点击"开始初始化"
4. 等待 `uv sync` 完成（3-10 分钟）

#### 11.3.2 切换到 GPU

VoxCPM 在 CPU 版 torch 下推理速度极慢（一句话 30+ 秒），强烈建议切换到 CUDA 版：
1. 在"环境设置"卡片的"核心依赖"区域点击"切换到 GPU"按钮
2. 详见 [切换到 GPU 用户指南](../../docs/docs/how-to/torch-cuda.md)
3. 切换后推理速度提升约 10 倍（2-3 秒/句）

#### 11.3.3 重置环境

如果 VoxCPM 引擎报 `ModuleNotFoundError` 或 torch 检测失败：
1. **先停止引擎**
2. 在"环境设置"卡片点击"重置环境"按钮
3. 确认破坏性警告（torch 会回退到 CPU 版）
4. 等待重置完成
5. 重新点击"切换到 GPU"切换 torch

### 11.4 VoxCPM 已知问题

#### 11.4.1 device 未应用 bug

VoxCPM 引擎存在 `_resolve_device` 结果未传给 `VoxCPM.from_pretrained` 的已知 bug，
需修复后才能正确响应 `--device` 参数。详见
[SDK README §7 设备选择规范](../../sdk/aireliable-engine-sdk/README.md)。

#### 11.4.2 torch 版本要求

VoxCPM 要求 torch >=2.5.0，环境设置面板会根据 `env_config.torch_version_constraint`
显示版本约束。如检测到的 torch 版本低于 2.5.0，建议点击"重置环境"重新安装。

### 11.5 相关文档

- [管理引擎环境用户指南](../../docs/docs/how-to/engine-env-management.md)
- [切换到 GPU 用户指南](../../docs/docs/how-to/torch-cuda.md)
- [引擎环境设置面板架构](../../docs/docs/architecture/engine-env-settings.md)
- [VoxCPM 模型下载指南](../../docs/docs/how-to/voxcpm-model-download.md)

---

## 十二、参考链接

- [VoxCPM 官方仓库](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 HuggingFace](https://huggingface.co/OpenBMB/VoxCPM2)
- [VoxCPM-0.5B HuggingFace](https://huggingface.co/OpenBMB/VoxCPM-0.5B)
- [AIReliable 架构文档](../../PR/architecture.md)
- [AIReliable 开发指南](../../PR/development-guide.md)
- [CustomEngine API 参考](../../docs/docs/api-reference/custom-engine.md)
- [VoxCPM TTS 引擎需求](../../PR/requirements.md)
- [VoxCPM TTS 引擎分析](../../voxcpm-tts-engine-analysis.md)
- [VoxCPM TTS 引擎任务清单](../../voxcpm-tts-engine-tasks.md)
