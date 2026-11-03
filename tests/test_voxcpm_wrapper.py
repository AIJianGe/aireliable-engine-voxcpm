"""VoxCPMWrapper 单元测试（T-061）。

覆盖：
- 模型加载/卸载（mock voxcpm.VoxCPM）
- 四种合成模式的参数构造
- 设备解析（auto/cpu/cuda:0）
- 错误处理（模型未加载、缺参数、generate 失败）
- 属性访问（is_loaded / loaded_variant / sample_rate / device）

不依赖真实 voxcpm 包，通过 patch sys.modules 注入 mock。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engines.voxcpm_tts_c4a8f2e7.config import EngineConfig
from engines.voxcpm_tts_c4a8f2e7.voxcpm_wrapper import VoxCPMWrapper


# ===== Mock voxcpm 模块 =====


def _make_mock_voxcpm_module(
    sample_rate: int = 48000,
    generate_return: Any = None,
    fail_from_pretrained: bool = False,
    fail_generate: bool = False,
) -> MagicMock:
    """构造一个 mock 的 voxcpm 模块。"""
    mock_module = MagicMock()

    # VoxCPM.from_pretrained
    mock_model = MagicMock()
    mock_model.tts_model = MagicMock()
    mock_model.tts_model.sample_rate = sample_rate

    if fail_from_pretrained:
        mock_module.VoxCPM.from_pretrained.side_effect = RuntimeError("mock load failed")
    else:
        mock_module.VoxCPM.from_pretrained.return_value = mock_model

    # generate
    if generate_return is None:
        generate_return = np.zeros(sample_rate, dtype=np.float32)

    if fail_generate:
        mock_model.generate.side_effect = RuntimeError("mock generate failed")
    else:
        mock_model.generate.return_value = generate_return

    mock_module.VoxCPM.from_pretrained.return_value = mock_model
    return mock_module


def _make_wrapper(tmp_path: Path, device: str = "cpu") -> VoxCPMWrapper:
    config = EngineConfig(
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        project_root=tmp_path,
        device=device,
    )
    return VoxCPMWrapper(config)


@pytest.fixture
def mock_voxcpm():
    """注入 mock voxcpm 模块到 sys.modules。"""
    mock_module = _make_mock_voxcpm_module()
    with patch.dict(sys.modules, {"voxcpm": mock_module}):
        yield mock_module


@pytest.fixture
def mock_torch_cpu():
    """mock torch 模块（CPU 模式，无 CUDA）。"""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        yield mock_torch


# ===== 测试类 =====


class TestWrapperProperties:
    """VoxCPMWrapper 属性测试。"""

    def test_initial_state(self, tmp_path: Path):
        wrapper = _make_wrapper(tmp_path)
        assert wrapper.is_loaded is False
        assert wrapper.loaded_variant is None
        assert wrapper.sample_rate == 48000  # 默认值
        assert wrapper.device == "cpu"

    def test_initial_device_auto_with_cuda(self, tmp_path: Path):
        """v0.2.5+ __init__ 时传入 device='auto' 且 GPU 可用，应立即解析为 cuda:0。

        修复前：__init__ 硬编码 _device='cpu'，/status 误报 device=cpu
        修复后：__init__ 调用 _resolve_device，/status 立即返回 cuda:0
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path, device="auto")
            # 关键断言：未加载模型时 device 已是 cuda:0
            assert wrapper.device == "cuda:0"
            assert wrapper.is_loaded is False

    def test_initial_device_auto_without_cuda(self, tmp_path: Path):
        """__init__ 时传入 device='auto' 但无 GPU，应解析为 cpu。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path, device="auto")
            assert wrapper.device == "cpu"

    def test_initial_device_explicit_cpu(self, tmp_path: Path):
        """__init__ 时传入 device='cpu'，应保持 cpu（不触发 GPU 检测）。"""
        wrapper = _make_wrapper(tmp_path, device="cpu")
        assert wrapper.device == "cpu"


class TestWrapperLoad:
    """load() 测试。"""

    def test_load_success(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path/voxcpm2", "cpu")

        assert wrapper.is_loaded is True
        # v0.2.5+ loaded_variant 必须是 variant 名,不是路径
        assert wrapper.loaded_variant == "voxcpm2"
        assert wrapper.device == "cpu"
        mock_voxcpm.VoxCPM.from_pretrained.assert_called_once()

    def test_load_sets_sample_rate_from_model(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        mock_voxcpm.VoxCPM.from_pretrained.return_value.tts_model.sample_rate = 24000
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        assert wrapper.sample_rate == 24000

    def test_load_failure_resets_state(self, tmp_path: Path, mock_torch_cpu):
        mock_module = _make_mock_voxcpm_module(fail_from_pretrained=True)
        with patch.dict(sys.modules, {"voxcpm": mock_module}):
            wrapper = _make_wrapper(tmp_path)
            with pytest.raises(RuntimeError, match="VoxCPM 模型加载失败"):
                wrapper.load("voxcpm2", "/fake/path", "cpu")

            assert wrapper.is_loaded is False
            assert wrapper.loaded_variant is None

    def test_load_reloads_when_already_loaded(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path1", "cpu")
        assert wrapper.loaded_variant == "voxcpm2"

        # 第二次加载应先卸载
        wrapper.load("voxcpm-0.5b", "/fake/path2", "cpu")
        assert wrapper.loaded_variant == "voxcpm-0.5b"

    def test_load_missing_voxcpm_package(self, tmp_path: Path, mock_torch_cpu):
        """voxcpm 包未安装时抛 RuntimeError。"""
        # 确保 voxcpm 不在 sys.modules
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("voxcpm", None)
            wrapper = _make_wrapper(tmp_path)
            with pytest.raises(RuntimeError, match="voxcpm 包未安装"):
                wrapper.load("voxcpm2", "/fake/path", "cpu")


class TestWrapperUnload:
    """unload() 测试。"""

    def test_unload_when_loaded(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")
        assert wrapper.is_loaded is True

        wrapper.unload()
        assert wrapper.is_loaded is False
        assert wrapper.loaded_variant is None

    def test_unload_when_not_loaded(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        # 未加载时 unload 不应报错
        wrapper.unload()
        assert wrapper.is_loaded is False


class TestWrapperGenerate:
    """generate() 测试。"""

    def test_generate_without_load_raises(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        with pytest.raises(RuntimeError, match="模型未加载"):
            wrapper.generate(text="你好", mode="tts")

    def test_generate_tts_mode(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        samples, sr = wrapper.generate(text="你好世界", mode="tts")

        assert isinstance(samples, np.ndarray)
        assert sr == 48000
        # 检查 generate 被调用时的参数
        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        assert kwargs.get("cfg_value") == 2.0
        assert kwargs.get("inference_timesteps") == 10

    def test_generate_design_mode_with_description(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        """v0.2.5+ design 模式无括号时自动包裹 ()。"""
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(
            text="你好",
            mode="design",
            voice_description="A young woman",
        )

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        # design 模式无括号时自动包裹 (A young woman)你好
        assert kwargs.get("text") == "(A young woman)你好"

    def test_generate_design_mode_with_english_parens(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        """v0.2.5+ design 模式英文括号保持不变。"""
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(
            text="你好",
            mode="design",
            voice_description="(A young woman)",
        )

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        # 英文括号保持不变
        assert kwargs.get("text") == "(A young woman)你好"

    def test_generate_design_mode_with_chinese_parens(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        """v0.2.5+ design 模式中文括号转英文括号。"""
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(
            text="你好",
            mode="design",
            voice_description="（一个大叔沙哑声音）",
        )

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        # 中文括号转英文括号
        assert kwargs.get("text") == "(一个大叔沙哑声音)你好"

    def test_generate_design_mode_without_description(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(text="你好", mode="design", voice_description="")

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        assert kwargs.get("text") == "你好"

    def test_generate_clone_mode_with_reference(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(
            text="你好",
            mode="clone",
            reference_audio_path="/fake/ref.wav",
        )

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        assert kwargs.get("reference_wav_path") == "/fake/ref.wav"

    def test_generate_clone_mode_without_reference_raises(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        with pytest.raises(ValueError, match="Clone 模式需要 reference_audio_path"):
            wrapper.generate(text="你好", mode="clone")

    def test_generate_ultimate_clone_mode(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(
            text="你好",
            mode="ultimate_clone",
            prompt_audio_path="/fake/prompt.wav",
            prompt_text="这是提示文本",
        )

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        assert kwargs.get("prompt_wav_path") == "/fake/prompt.wav"
        assert kwargs.get("prompt_text") == "这是提示文本"

    def test_generate_ultimate_clone_missing_prompt_audio_raises(
        self, tmp_path: Path, mock_voxcpm, mock_torch_cpu
    ):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        with pytest.raises(ValueError, match="Ultimate Clone 模式需要 prompt_audio_path"):
            wrapper.generate(text="你好", mode="ultimate_clone", prompt_text="text")

    def test_generate_ultimate_clone_missing_prompt_text_raises(
        self, tmp_path: Path, mock_voxcpm, mock_torch_cpu
    ):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        with pytest.raises(ValueError, match="Ultimate Clone 模式需要 prompt_text"):
            wrapper.generate(text="你好", mode="ultimate_clone", prompt_audio_path="/p.wav")

    def test_generate_unknown_mode_raises(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        with pytest.raises(ValueError, match="未知合成模式"):
            wrapper.generate(text="你好", mode="unknown_mode")

    def test_generate_with_seed(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        """v0.3.6+ 行为：VoxCPM 2.0.3 不支持 seed，传入 seed >= 0 时记 WARNING 后忽略，
        不注入 kwargs，避免 VoxCPM._generate() 抛 TypeError。"""
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(text="你好", mode="tts", seed=42)

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        # seed 不应注入 kwargs（VoxCPM 2.0.3 不接受）
        assert "seed" not in kwargs

    def test_generate_without_seed_omits_seed(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        wrapper.generate(text="你好", mode="tts", seed=-1)

        _, kwargs = mock_voxcpm.VoxCPM.from_pretrained.return_value.generate.call_args
        assert "seed" not in kwargs

    def test_generate_failure_wraps_error(self, tmp_path: Path, mock_torch_cpu):
        mock_module = _make_mock_voxcpm_module(fail_generate=True)
        with patch.dict(sys.modules, {"voxcpm": mock_module}):
            wrapper = _make_wrapper(tmp_path)
            wrapper.load("voxcpm2", "/fake/path", "cpu")

            with pytest.raises(RuntimeError, match="VoxCPM 生成失败"):
                wrapper.generate(text="你好", mode="tts")

    def test_generate_returns_tuple_from_voxcpm(self, tmp_path: Path, mock_voxcpm, mock_torch_cpu):
        """VoxCPM 某些版本返回 (wav, sr) 元组。"""
        expected_samples = np.zeros(24000, dtype=np.float32)
        mock_model = mock_voxcpm.VoxCPM.from_pretrained.return_value
        mock_model.generate.return_value = (expected_samples, 24000)

        wrapper = _make_wrapper(tmp_path)
        wrapper.load("voxcpm2", "/fake/path", "cpu")

        samples, sr = wrapper.generate(text="你好", mode="tts")
        assert sr == 24000
        assert len(samples) == 24000


class TestDeviceResolution:
    """_resolve_device() 测试。"""

    def test_auto_with_cuda(self, tmp_path: Path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path)
            assert wrapper._resolve_device("auto") == "cuda:0"

    def test_auto_without_cuda(self, tmp_path: Path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path)
            assert wrapper._resolve_device("auto") == "cpu"

    def test_auto_without_torch(self, tmp_path: Path):
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("torch", None)
            wrapper = _make_wrapper(tmp_path)
            assert wrapper._resolve_device("auto") == "cpu"

    def test_explicit_device(self, tmp_path: Path):
        wrapper = _make_wrapper(tmp_path)
        assert wrapper._resolve_device("cuda:1") == "cuda:1"
        assert wrapper._resolve_device("cpu") == "cpu"


class TestGetDeviceName:
    """_get_device_name() 测试（v0.2.5+）。"""

    def test_cpu_returns_CPU(self, tmp_path: Path):
        wrapper = _make_wrapper(tmp_path, device="cpu")
        assert wrapper._get_device_name("cpu") == "CPU"

    def test_cuda0_returns_gpu_name(self, tmp_path: Path):
        """cuda:0 时应调用 torch.cuda.get_device_name(0) 返回 GPU 名称。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path, device="cpu")  # 避免初始化时解析
            name = wrapper._get_device_name("cuda:0")
            assert name == "NVIDIA GeForce RTX 4090"
            mock_torch.cuda.get_device_name.assert_called_once_with(0)

    def test_cuda1_returns_second_gpu_name(self, tmp_path: Path):
        """cuda:1 应获取第 2 块 GPU 的名称。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path, device="cpu")
            name = wrapper._get_device_name("cuda:1")
            assert name == "NVIDIA GeForce RTX 3090"
            mock_torch.cuda.get_device_name.assert_called_once_with(1)

    def test_cuda_without_colon_uses_index_0(self, tmp_path: Path):
        """device="cuda"（无冒号）应使用索引 0。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4080"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path, device="cpu")
            name = wrapper._get_device_name("cuda")
            assert name == "NVIDIA GeForce RTX 4080"
            mock_torch.cuda.get_device_name.assert_called_once_with(0)

    def test_cuda_returns_empty_name_falls_back_to_device(self, tmp_path: Path):
        """torch 拿不到卡名时回落到系统探测；系统探测也为空则用 device 字符串。

        任务 71：设备探测已下沉到 ``aireliable_engine.device``，需在 SDK 层
        mock 系统探测，避免用例结果依赖测试机是否真的有 GPU。
        """
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = ""
        with patch.dict(sys.modules, {"torch": mock_torch}), patch(
            "aireliable_engine.device._probe_system_layer",
            return_value={"device_name": "", "smi_ok": False},
        ):
            wrapper = _make_wrapper(tmp_path, device="cpu")
            name = wrapper._get_device_name("cuda:0")
            assert name == "cuda:0"

    def test_torch_import_error_returns_device_string(self, tmp_path: Path):
        """torch 导入失败时，卡名回落到系统探测；仍为空则用 device 字符串。"""
        with patch.dict(sys.modules, {}, clear=False), patch(
            "aireliable_engine.device._probe_system_layer",
            return_value={"device_name": "", "smi_ok": False},
        ):
            sys.modules.pop("torch", None)
            wrapper = _make_wrapper(tmp_path, device="cpu")
            name = wrapper._get_device_name("cuda:0")
            assert name == "cuda:0"


class TestInitialDeviceName:
    """__init__ 时 device_name 解析测试（v0.2.5+）。"""

    def test_init_device_name_with_cuda(self, tmp_path: Path):
        """__init__ 时 device='auto' + GPU 可用，device_name 应为 GPU 名称。"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path, device="auto")
            assert wrapper.device == "cuda:0"
            assert wrapper.device_name == "NVIDIA GeForce RTX 4090"

    def test_init_device_name_with_cpu(self, tmp_path: Path):
        """__init__ 时 device='cpu'，device_name 应为 'CPU'。"""
        wrapper = _make_wrapper(tmp_path, device="cpu")
        assert wrapper.device == "cpu"
        assert wrapper.device_name == "CPU"


class TestVramUsed:
    """get_vram_used() 测试。"""

    def test_cpu_returns_zero(self, tmp_path: Path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path)
            assert wrapper.get_vram_used() == 0

    def test_cuda_returns_allocated(self, tmp_path: Path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
        with patch.dict(sys.modules, {"torch": mock_torch}):
            wrapper = _make_wrapper(tmp_path)
            wrapper._device = "cuda:0"
            assert wrapper.get_vram_used() == 100

    def test_without_torch_returns_zero(self, tmp_path: Path):
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("torch", None)
            wrapper = _make_wrapper(tmp_path)
            assert wrapper.get_vram_used() == 0
