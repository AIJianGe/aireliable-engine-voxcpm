"""VoxCPM ModelScanner 单元测试（T13 + v0.2.11+ 方案 B）。

覆盖：
- scan_variants（同步 fallback）：完整模型 / 部分模型 / 空目录 / 目录不存在
- scan_variants_async：平台 API 返回 / 平台不可达 fallback / engine 未注入
- resolve_variant_path：已知 / 未知变体
- _check_model_files：config.json 标志文件检查逻辑
- _merge_with_known：平台数据 + _KNOWN_VARIANTS 引擎特有字段合并
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from engines.voxcpm_tts_c4a8f2e7.model_scanner import ModelScanner, _KNOWN_VARIANTS
from engines.voxcpm_tts_c4a8f2e7.schemas import TTSCapability


# ===== Fixtures =====


@pytest.fixture
def variant_dir(tmp_path: Path) -> Path:
    """创建一个完整的 voxcpm2 变体目录（含 config.json + safetensors）。"""
    v_dir = tmp_path / "models" / "voice" / "tts" / "voxcpm" / "voxcpm2"
    v_dir.mkdir(parents=True)

    # 创建标志文件
    (v_dir / "config.json").write_text("{}")
    (v_dir / "model.safetensors").write_bytes(b"fake")

    return tmp_path / "models" / "voice" / "tts" / "voxcpm"


@pytest.fixture
def partial_variant_dir(tmp_path: Path) -> Path:
    """创建一个不含任何模型文件的空变体目录。"""
    v_dir = tmp_path / "models" / "voice" / "tts" / "voxcpm" / "voxcpm2"
    v_dir.mkdir(parents=True)
    # 不创建任何文件
    return tmp_path / "models" / "voice" / "tts" / "voxcpm"


@pytest.fixture
def multi_variant_dir(tmp_path: Path) -> Path:
    """创建包含两个变体（voxcpm2 + voxcpm-0.5b）的目录。"""
    for subdir in ("voxcpm2", "voxcpm-0.5b"):
        v_dir = tmp_path / "models" / "voice" / "tts" / "voxcpm" / subdir
        v_dir.mkdir(parents=True)
        (v_dir / "config.json").write_text("{}")
        (v_dir / "model.safetensors").write_bytes(b"fake")

    return tmp_path / "models" / "voice" / "tts" / "voxcpm"


# ===== 测试类 =====


class TestScanVariantsSync:
    """scan_variants 同步接口测试（fallback 本地扫描）。"""

    def test_scan_finds_complete_variant(self, variant_dir: Path):
        """完整变体目录应被识别为可用。"""
        scanner = ModelScanner(variant_dir)
        variants = scanner.scan_variants()

        assert len(variants) == 1
        v = variants[0]
        assert isinstance(v, TTSCapability)
        assert v.name == "OpenBMB/voxcpm2"
        assert v.alias == "VoxCPM2"
        assert v.precision == "fp16"
        assert v.vram_required == 8192
        assert v.sample_rate == 48000
        assert "zh" in v.languages
        assert "en" in v.languages

    def test_scan_finds_multiple_variants(self, multi_variant_dir: Path):
        """多变体目录应识别全部可用变体。"""
        scanner = ModelScanner(multi_variant_dir)
        variants = scanner.scan_variants()

        names = [v.name for v in variants]
        assert "OpenBMB/voxcpm2" in names
        assert "OpenBMB/voxcpm-0.5b" in names

    def test_scan_skips_empty_variant_dir(self, partial_variant_dir: Path):
        """空变体目录应被跳过。"""
        scanner = ModelScanner(partial_variant_dir)
        variants = scanner.scan_variants()
        assert len(variants) == 0

    def test_scan_returns_empty_when_dir_not_exists(self, tmp_path: Path):
        """目录不存在时应返回空列表。"""
        scanner = ModelScanner(tmp_path / "nonexistent")
        variants = scanner.scan_variants()
        assert variants == []


class TestScanVariantsAsync:
    """scan_variants_async 测试（v0.2.11+ 方案 B）。

    覆盖三种场景：
        1. 平台 API 返回 variant 列表 → 合并 _KNOWN_VARIANTS 引擎特有字段
        2. 平台 API 不可达（get_variants 返回 None）→ fallback 到本地扫描
        3. engine 未注入（engine=None）→ 直接走 fallback 本地扫描
    """

    @pytest.mark.asyncio
    async def test_platform_api_returns_variants(self, variant_dir: Path):
        """场景 1：平台 API 返回 variant 列表，应合并 _KNOWN_VARIANTS 特有字段。"""
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[
            {
                "name": "OpenBMB/voxcpm2",
                "alias": "VoxCPM2",
                "precision": "fp16",
                "vram_required": "8GB",
                "available": True,
                "model_file_ids": ["voxcpm2-config"],
                "missing_file_ids": [],
            },
        ])

        scanner = ModelScanner(variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        assert len(variants) == 1
        v = variants[0]
        assert isinstance(v, TTSCapability)
        # 平台返回的通用字段
        assert v.name == "OpenBMB/voxcpm2"
        assert v.alias == "VoxCPM2"
        assert v.precision == "fp16"
        # 引擎特有字段（来自 _KNOWN_VARIANTS）
        assert v.sample_rate == 48000
        assert "zh" in v.languages
        assert "ja" in v.languages  # voxcpm2 支持 7 种语言

    @pytest.mark.asyncio
    async def test_platform_api_returns_multiple_variants(self, multi_variant_dir: Path):
        """场景 1b：平台返回多变体，应全部合并。"""
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[
            {
                "name": "OpenBMB/voxcpm2",
                "alias": "VoxCPM2",
                "precision": "fp16",
                "vram_required": "8GB",
                "available": True,
            },
            {
                "name": "OpenBMB/voxcpm-0.5b",
                "alias": "VoxCPM-0.5B",
                "precision": "fp16",
                "vram_required": "5GB",
                "available": True,
            },
        ])

        scanner = ModelScanner(multi_variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        assert len(variants) == 2
        names = [v.name for v in variants]
        assert "OpenBMB/voxcpm2" in names
        assert "OpenBMB/voxcpm-0.5b" in names

        # 验证引擎特有字段被正确合并
        v2 = next(v for v in variants if v.name == "OpenBMB/voxcpm2")
        v05b = next(v for v in variants if v.name == "OpenBMB/voxcpm-0.5b")
        assert v2.sample_rate == 48000
        assert v05b.sample_rate == 16000  # 0.5b 采样率不同
        assert len(v2.languages) == 7  # voxcpm2 支持 7 种语言
        assert len(v05b.languages) == 2  # voxcpm-0.5b 仅 zh+en

    @pytest.mark.asyncio
    async def test_platform_api_returns_unavailable_variant(self, variant_dir: Path):
        """场景 1c：平台 API 返回 available=False 的 variant，应保留状态。"""
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[
            {
                "name": "OpenBMB/voxcpm2",
                "alias": "VoxCPM2",
                "precision": "fp16",
                "vram_required": "8GB",
                "available": False,
                "model_file_ids": ["voxcpm2-config"],
                "missing_file_ids": ["voxcpm2-config"],
            },
        ])

        scanner = ModelScanner(variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        # TTSCapability 没有 available 字段，但应保留变体记录
        assert len(variants) == 1
        assert variants[0].name == "OpenBMB/voxcpm2"
        # 引擎特有字段仍应被合并
        assert variants[0].sample_rate == 48000

    @pytest.mark.asyncio
    async def test_platform_api_unreachable_falls_back_to_local(
        self, variant_dir: Path,
    ):
        """场景 2：平台 API 不可达（返回 None），应 fallback 到本地扫描。"""
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=None)

        scanner = ModelScanner(variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        # fallback 到本地扫描，应识别出 OpenBMB/voxcpm2
        assert len(variants) == 1
        assert variants[0].name == "OpenBMB/voxcpm2"

    @pytest.mark.asyncio
    async def test_no_engine_uses_local_scan(self, variant_dir: Path):
        """场景 3：engine 未注入（None），应直接走 fallback 本地扫描。"""
        scanner = ModelScanner(variant_dir, engine=None)
        variants = await scanner.scan_variants_async()

        assert len(variants) == 1
        assert variants[0].name == "OpenBMB/voxcpm2"


class TestResolveVariantPath:
    """resolve_variant_path 测试。"""

    def test_resolve_known_variant(self, variant_dir: Path):
        scanner = ModelScanner(variant_dir)
        path = scanner.resolve_variant_path("OpenBMB/voxcpm2")
        assert path is not None
        assert path.name == "voxcpm2"
        assert path.exists()

    def test_resolve_unknown_variant_returns_none(self, variant_dir: Path):
        scanner = ModelScanner(variant_dir)
        path = scanner.resolve_variant_path("unknown-variant")
        assert path is None


class TestCheckModelFiles:
    """_check_model_files 测试。"""

    def test_config_json_marks_variant_ready(self, variant_dir: Path):
        """config.json 标志文件存在时应识别为可用。"""
        scanner = ModelScanner(variant_dir)
        variant_path = variant_dir / "voxcpm2"
        assert scanner._check_model_files(variant_path) is True

    def test_empty_dir_not_ready(self, partial_variant_dir: Path):
        """空目录应识别为不可用。"""
        scanner = ModelScanner(partial_variant_dir)
        variant_path = partial_variant_dir / "voxcpm2"
        assert scanner._check_model_files(variant_path) is False

    def test_safetensors_only_marks_ready(self, tmp_path: Path):
        """仅有 safetensors 文件（无 config.json）也应识别为可用。"""
        v_dir = tmp_path / "models" / "voice" / "tts" / "voxcpm" / "voxcpm2"
        v_dir.mkdir(parents=True)
        (v_dir / "model.safetensors").write_bytes(b"fake")

        scanner = ModelScanner(tmp_path / "models" / "voice" / "tts" / "voxcpm")
        assert scanner._check_model_files(v_dir) is True


class TestMergeWithKnown:
    """_merge_with_known 测试。"""

    def test_merges_platform_data_with_known_variants(self, variant_dir: Path):
        """应正确合并平台通用字段 + _KNOWN_VARIANTS 引擎特有字段。"""
        scanner = ModelScanner(variant_dir)
        platform_variants = [
            {
                "name": "OpenBMB/voxcpm2",
                "alias": "VoxCPM2",
                "precision": "fp16",
                "vram_required": "8GB",
                "available": True,
            },
        ]

        result = scanner._merge_with_known(platform_variants)

        assert len(result) == 1
        v = result[0]
        assert v.name == "OpenBMB/voxcpm2"
        # 引擎特有字段来自 _KNOWN_VARIANTS
        assert v.sample_rate == 48000
        assert "zh" in v.languages

    def test_unknown_variant_uses_defaults(self, variant_dir: Path):
        """平台返回未知 variant 时，应使用默认值。"""
        scanner = ModelScanner(variant_dir)
        platform_variants = [
            {
                "name": "voxcpm-3-unknown",
                "alias": "V3",
                "available": True,
            },
        ]

        result = scanner._merge_with_known(platform_variants)

        assert len(result) == 1
        v = result[0]
        assert v.name == "voxcpm-3-unknown"
        # 未在 _KNOWN_VARIANTS 中找到 → 使用默认值
        assert v.sample_rate == 16000  # 默认值
        assert v.languages == []  # 默认空列表

    def test_empty_platform_list_returns_empty(self, variant_dir: Path):
        """平台返回空列表时，应返回空列表。"""
        scanner = ModelScanner(variant_dir)
        result = scanner._merge_with_known([])
        assert result == []


class TestKnownVariants:
    """_KNOWN_VARIANTS 配置测试。"""

    def test_known_variants_contains_voxcpm2(self):
        """确认 _KNOWN_VARIANTS 包含 OpenBMB/voxcpm2 定义。"""
        names = [v["name"] for v in _KNOWN_VARIANTS]
        assert "OpenBMB/voxcpm2" in names
        assert "OpenBMB/voxcpm-0.5b" in names

    def test_voxcpm2_has_correct_sample_rate(self):
        """确认 OpenBMB/voxcpm2 配置了正确的采样率。"""
        v2 = next(v for v in _KNOWN_VARIANTS if v["name"] == "OpenBMB/voxcpm2")
        assert v2["sample_rate"] == 48000
        assert "ja" in v2["languages"]

    def test_voxcpm_05b_has_correct_sample_rate(self):
        """确认 OpenBMB/voxcpm-0.5b 配置了正确的采样率。"""
        v05b = next(v for v in _KNOWN_VARIANTS if v["name"] == "OpenBMB/voxcpm-0.5b")
        assert v05b["sample_rate"] == 16000
        assert v05b["languages"] == ["zh", "en"]
