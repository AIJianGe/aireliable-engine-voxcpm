"""VoxCPMEngine 单元测试（T-060）。

覆盖 7 个 `_xxx_impl` 方法：
- _capabilities_impl：返回扫描到的变体
- _load_impl：调用 wrapper.load，未找到 variant 抛错
- _unload_impl：未加载时返回 idle
- _infer_impl：四种模式的参数校验、推理成功/失败路径
- _status_impl：返回设备/采样率等额外信息
- _load_lora_impl / _unload_lora_impl：v0.1 返回 501

Mock VoxCPMWrapper 和 ModelScanner 避免真实模型加载。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aireliable_engine.schemas import AudioInput, InferRequest

from engines.voxcpm_tts_c4a8f2e7.config import EngineConfig
from engines.voxcpm_tts_c4a8f2e7.schemas import TTSCapability, TTSInputs


# ===== Mock 工厂 =====


def _make_variant(
    name: str = "OpenBMB/voxcpm2",
    sample_rate: int = 48000,
    vram_required: int = 8192,
) -> TTSCapability:
    return TTSCapability(
        name=name,
        alias=name.upper(),
        precision="fp16",
        vram_required=vram_required,
        sample_rate=sample_rate,
        languages=["zh", "en"],
    )


def _make_mock_wrapper(
    is_loaded: bool = False,
    loaded_variant: Optional[str] = None,
    device: str = "cpu",
    sample_rate: int = 48000,
    vram_used: int = 0,
    generate_samples: Optional[np.ndarray] = None,
    fail_generate: bool = False,
    fail_load: bool = False,
) -> MagicMock:
    """构造一个 Mock 的 VoxCPMWrapper。"""
    wrapper = MagicMock()
    wrapper.is_loaded = is_loaded
    wrapper.loaded_variant = loaded_variant
    wrapper.device = device
    wrapper.sample_rate = sample_rate
    wrapper.get_vram_used.return_value = vram_used

    if fail_load:
        def _fail_load(*a, **kw):
            raise RuntimeError("mock load failed")
        wrapper.load.side_effect = _fail_load
    else:
        wrapper.load.return_value = None

    if fail_generate:
        def _fail_gen(*a, **kw):
            raise RuntimeError("mock generate failed")
        wrapper.generate.side_effect = _fail_gen
    else:
        samples = generate_samples if generate_samples is not None else np.zeros(
            sample_rate, dtype=np.float32
        )
        wrapper.generate.return_value = (samples, sample_rate)

    wrapper.unload.return_value = None
    return wrapper


# sentinel 用于区分 "未传参" 和 "显式传 None"
_SENTINEL = object()


def _make_mock_scanner(
    variants: Optional[list] = None,
    variant_path: object = _SENTINEL,
) -> MagicMock:
    """构造一个 Mock 的 ModelScanner。

    Args:
        variants: 变体列表，None 表示用默认
        variant_path: resolve_variant_path 的返回值。
                      未传（_SENTINEL）用默认路径，显式传 None 返回 None。
    """
    scanner = MagicMock()
    if variants is None:
        variants = [_make_variant()]
    scanner.scan_variants.return_value = variants
    if variant_path is _SENTINEL:
        variant_path = Path("/fake/voxcpm2")
    scanner.resolve_variant_path.return_value = variant_path
    return scanner


def _fake_save_audio(samples, sr, path, fmt="wav"):
    """模拟 save_audio：直接写空文件，避免依赖 soundfile。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FAKE_WAV")
    return path


def _fake_get_duration(samples, sr):
    """模拟 get_audio_duration。"""
    return 1.0


@pytest.fixture
def mock_audio_utils():
    """mock engine 模块中的 save_audio / get_audio_duration，避免依赖 soundfile。"""
    with patch("engines.voxcpm_tts_c4a8f2e7.engine.save_audio", side_effect=_fake_save_audio), \
         patch("engines.voxcpm_tts_c4a8f2e7.engine.get_audio_duration", side_effect=_fake_get_duration):
        yield


def _make_engine(
    tmp_path: Path,
    wrapper: Optional[MagicMock] = None,
    scanner: Optional[MagicMock] = None,
) -> "VoxCPMEngine":
    """构造一个带 mock 依赖的 VoxCPMEngine。

    注意：save_audio / get_audio_duration 的 mock 由 ``mock_audio_utils`` fixture
    提供，需要在测试方法中声明该 fixture。
    """
    from engines.voxcpm_tts_c4a8f2e7.engine import VoxCPMEngine

    config = EngineConfig(
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        project_root=tmp_path,
        device="cpu",
    )

    with patch("engines.voxcpm_tts_c4a8f2e7.engine.VoxCPMWrapper") as MockWrapper, \
         patch("engines.voxcpm_tts_c4a8f2e7.engine.ModelScanner") as MockScanner:
        MockWrapper.return_value = wrapper or _make_mock_wrapper()
        MockScanner.return_value = scanner or _make_mock_scanner()
        engine = VoxCPMEngine(config)
        # 把 mock 引用挂上去便于断言
        engine.wrapper = MockWrapper.return_value
        engine.scanner = MockScanner.return_value
    return engine


def _parse_sse_frames(frames: list[str]) -> list[dict]:
    """解析 SSE 帧字符串列表为 dict 列表。"""
    out = []
    for f in frames:
        body = f
        if body.startswith("data: "):
            body = body[len("data: "):]
        if body.endswith("\n\n"):
            body = body[:-2]
        out.append(json.loads(body))
    return out


# ===== 测试类 =====


class TestCapabilitiesImpl:
    """_capabilities_impl 测试。"""

    @pytest.mark.asyncio
    async def test_returns_scanned_variants(self, tmp_path: Path):
        v1 = _make_variant(name="OpenBMB/voxcpm2", sample_rate=48000)
        v2 = _make_variant(name="OpenBMB/voxcpm-0.5b", sample_rate=16000)
        scanner = _make_mock_scanner(variants=[v1, v2])
        eng = _make_engine(tmp_path, scanner=scanner)

        r = await eng._capabilities_impl()

        assert r.engine_id == eng.engine_id
        assert r.capability == "text-to-speech"
        assert r.type == "voice"
        assert len(r.variants) == 2
        assert r.variants[0].name == "OpenBMB/voxcpm2"
        assert r.variants[0].sample_rate == 48000

    @pytest.mark.asyncio
    async def test_empty_variants_when_no_models(self, tmp_path: Path):
        scanner = _make_mock_scanner(variants=[])
        eng = _make_engine(tmp_path, scanner=scanner)

        r = await eng._capabilities_impl()
        assert r.variants == []
        assert r.loras == []


class TestLoadImpl:
    """_load_impl 测试。"""

    @pytest.mark.asyncio
    async def test_load_success(self, tmp_path: Path):
        wrapper = _make_mock_wrapper()
        scanner = _make_mock_scanner(variant_path=Path("/fake/voxcpm2"))
        eng = _make_engine(tmp_path, wrapper=wrapper, scanner=scanner)

        r = await eng._load_impl("OpenBMB/voxcpm2")

        wrapper.load.assert_called_once()
        assert r.variant == "OpenBMB/voxcpm2"
        assert r.state == "ready"
        assert "MB" in r.vram_used

    @pytest.mark.asyncio
    async def test_load_unknown_variant_raises(self, tmp_path: Path):
        scanner = _make_mock_scanner(variant_path=None)
        eng = _make_engine(tmp_path, scanner=scanner)

        with pytest.raises(FileNotFoundError, match="variant 不存在"):
            await eng._load_impl("unknown")

    @pytest.mark.asyncio
    async def test_load_failure_wraps_error(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(fail_load=True)
        scanner = _make_mock_scanner(variant_path=Path("/fake/voxcpm2"))
        eng = _make_engine(tmp_path, wrapper=wrapper, scanner=scanner)

        with pytest.raises(RuntimeError, match="VoxCPM 模型加载失败"):
            await eng._load_impl("OpenBMB/voxcpm2")


class TestUnloadImpl:
    """_unload_impl 测试。"""

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=False, vram_used=0)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._unload_impl()

        wrapper.unload.assert_not_called()
        assert r.state == "idle"
        assert "0MB" in r.vram_freed

    @pytest.mark.asyncio
    async def test_unload_when_loaded(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=True, vram_used=2048)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._unload_impl()

        wrapper.unload.assert_called_once()
        assert r.state == "idle"
        assert "2048MB" in r.vram_freed


class TestInferImpl:
    """_infer_impl 测试。"""

    @pytest.mark.asyncio
    async def test_infer_tts_mode_success(self, tmp_path: Path, mock_audio_utils):
        wrapper = _make_mock_wrapper(
            is_loaded=True, loaded_variant="OpenBMB/voxcpm2",
            generate_samples=np.zeros(48000, dtype=np.float32),
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="OpenBMB/voxcpm2",
            inputs={"text": "你好世界", "mode": "tts"},
            task_id="test-task-1",
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        stages = [f.get("stage") for f in parsed]

        # 应包含 start / model_ready / text_processing / generating / saving / completed
        assert "tts.start" in stages
        assert "completed" in stages
        # 最后一帧含 audio_url
        last = parsed[-1]
        assert last["stage"] == "completed"
        assert "/output/audio/test-task-1.wav" in last["result"]["audio_url"]
        wrapper.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_infer_clone_mode_requires_reference_audio(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=True, loaded_variant="OpenBMB/voxcpm2")
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="OpenBMB/voxcpm2",
            inputs={"text": "你好", "mode": "clone"},
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        error_frames = [f for f in parsed if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert "reference_audio" in error_frames[0]["error"]["type"]
        wrapper.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_infer_clone_mode_with_speaker_audio(self, tmp_path: Path, mock_audio_utils):
        wrapper = _make_mock_wrapper(is_loaded=True, loaded_variant="OpenBMB/voxcpm2")
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="OpenBMB/voxcpm2",
            inputs={"text": "你好", "mode": "clone"},
            audios=[AudioInput(path="/fake/ref.wav", role="speaker")],
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        assert parsed[-1]["stage"] == "completed"
        wrapper.generate.assert_called_once()
        # 检查 reference_audio_path 被传入
        _, kwargs = wrapper.generate.call_args
        assert kwargs.get("reference_audio_path") == "/fake/ref.wav"

    @pytest.mark.asyncio
    async def test_infer_ultimate_clone_requires_prompt_text(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=True, loaded_variant="OpenBMB/voxcpm2")
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="OpenBMB/voxcpm2",
            inputs={"text": "你好", "mode": "ultimate_clone"},
            audios=[AudioInput(path="/fake/p.wav", role="prompt")],
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        error_frames = [f for f in parsed if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert "prompt_text" in error_frames[0]["error"]["type"]

    @pytest.mark.asyncio
    async def test_infer_invalid_inputs_emits_error(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=True)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        # 缺少必填的 text 字段
        req = InferRequest(
            variant="OpenBMB/voxcpm2",
            inputs={"mode": "tts"},
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        error_frames = [f for f in parsed if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert error_frames[0]["error"]["type"] == "invalid_inputs"

    @pytest.mark.asyncio
    async def test_infer_generate_failure_emits_error(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(
            is_loaded=True, fail_generate=True,
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="OpenBMB/voxcpm2",
            inputs={"text": "你好", "mode": "tts"},
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        error_frames = [f for f in parsed if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert error_frames[0]["error"]["type"] == "infer_failed"


class TestStatusImpl:
    """_status_impl 测试。"""

    @pytest.mark.asyncio
    async def test_status_returns_extra_fields(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(
            is_loaded=True, loaded_variant="OpenBMB/voxcpm2",
            device="cuda:0", sample_rate=48000, vram_used=4096,
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._status_impl()

        assert r.state == "idle"  # 引擎刚启动未触发状态机转换
        assert r.loaded_variant == "OpenBMB/voxcpm2"
        assert "4096MB" in r.vram_used
        assert r.extra["device"] == "cuda:0"
        assert r.extra["sample_rate"] == 48000
        assert r.extra["is_loaded"] is True


class TestLoraImpl:
    """_load_lora_impl / _unload_lora_impl 测试（v0.1 不支持）。"""

    @pytest.mark.asyncio
    async def test_load_lora_returns_501(self, tmp_path: Path):
        from fastapi import HTTPException

        eng = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc_info:
            await eng._load_lora_impl("l1", 0.5)
        assert exc_info.value.status_code == 501

    @pytest.mark.asyncio
    async def test_unload_lora_returns_501(self, tmp_path: Path):
        from fastapi import HTTPException

        eng = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc_info:
            await eng._unload_lora_impl("l1")
        assert exc_info.value.status_code == 501
