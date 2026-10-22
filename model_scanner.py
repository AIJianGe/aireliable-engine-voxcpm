"""VoxCPM 模型文件扫描器。

扫描 models/voice/tts/voxcpm/ 目录，识别可用的 VoxCPM 模型变体。

v0.2.11+ 方案 B 改造：
    优先调用平台 API（`GET /engines/{id}/variants`）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import TTSCapability

logger = logging.getLogger(__name__)


# ===== 已知模型变体定义（fallback + 引擎特有字段源）=====
# v0.3.6+ variant.name 必须为 author/model 格式（命名空间规范，防同名冲突）
# 与 manifest.yaml variants 列表保持一致（契约约束）

_KNOWN_VARIANTS: list[dict] = [
    {
        "name": "OpenBMB/voxcpm2",
        "alias": "VoxCPM2",
        "precision": "fp16",
        "vram_required": 8192,
        "sample_rate": 48000,
        "languages": ["zh", "en", "ja", "ko", "fr", "de", "es"],
        "subdir": "voxcpm2",
    },
    {
        "name": "OpenBMB/voxcpm-0.5b",
        "alias": "VoxCPM-0.5B",
        "precision": "fp16",
        "vram_required": 5120,
        "sample_rate": 16000,
        "languages": ["zh", "en"],
        "subdir": "voxcpm-0.5b",
    },
]


class ModelScanner:
    """模型目录扫描器。

    扫描 model_dir 下的子目录，识别已下载的 VoxCPM 模型变体。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(self, model_dir: Path, engine=None):
        """初始化扫描器。

        Args:
            model_dir: 模型根目录（如 models/voice/tts/voxcpm）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑（兼容旧测试）。
        """
        self.model_dir = model_dir
        self.engine = engine

    async def scan_variants_async(self) -> list[TTSCapability]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。

        调用链：
            1. engine.get_variants() → 平台 API（声明式检测）
            2. 平台返回 None → fallback 到 _scan_locally()
            3. 合并平台 available 字段 + _KNOWN_VARIANTS 引擎特有字段
        """
        if self.engine is not None:
            platform_variants = await self.engine.get_variants()
            if platform_variants is not None:
                return self._merge_with_known(platform_variants)

        # fallback：本地扫描
        return self._scan_locally()

    def scan_variants(self) -> list[TTSCapability]:
        """同步扫描接口（向后兼容）。

        v0.2.11+ 注：若需使用方案 B 平台 API，请调用 scan_variants_async()。
        此同步方法走 fallback 本地扫描逻辑。
        """
        return self._scan_locally()

    def _merge_with_known(self, platform_variants: list[dict]) -> list[TTSCapability]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS 引擎特有字段。

        Args:
            platform_variants: 平台 API 返回的 variant 列表（含 available 字段）

        Returns:
            TTSCapability 列表（available 来自平台，sample_rate/languages 来自 _KNOWN_VARIANTS）
        """
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[TTSCapability] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            result.append(TTSCapability(
                name=v_name,
                alias=pv.get("alias", kv.get("alias", "")),
                precision=pv.get("precision", kv.get("precision", "fp16")),
                vram_required=kv.get("vram_required", 0),
                sample_rate=kv.get("sample_rate", 16000),
                languages=kv.get("languages", []),
            ))
            if pv.get("available"):
                logger.info("发现模型变体: %s (%s)", v_name, pv.get("alias", ""))

        if not any(pv.get("available") for pv in platform_variants):
            logger.warning(
                "未在 %s 中找到任何 VoxCPM 模型，"
                "请下载模型到对应目录（如 models/voice/tts/voxcpm/voxcpm2/）",
                self.model_dir,
            )

        return result

    def _scan_locally(self) -> list[TTSCapability]:
        """fallback 本地扫描（原 scan_variants 逻辑）。"""
        capabilities: list[TTSCapability] = []

        if not self.model_dir.exists():
            logger.warning("模型目录不存在: %s", self.model_dir)
            return capabilities

        for variant_def in _KNOWN_VARIANTS:
            variant_dir = self.model_dir / variant_def["subdir"]

            if not variant_dir.exists():
                logger.debug("变体 %s 目录不存在: %s", variant_def["name"], variant_dir)
                continue

            # 检查是否有模型文件
            has_model = self._check_model_files(variant_dir)
            if not has_model:
                logger.debug("变体 %s 目录内无模型文件: %s", variant_def["name"], variant_dir)
                continue

            cap = TTSCapability(
                name=variant_def["name"],
                alias=variant_def["alias"],
                precision=variant_def["precision"],
                vram_required=variant_def["vram_required"],
                sample_rate=variant_def["sample_rate"],
                languages=variant_def["languages"],
            )
            capabilities.append(cap)
            logger.info("发现模型变体: %s (%s)", cap.name, cap.alias)

        if not capabilities:
            logger.warning(
                "未在 %s 中找到任何 VoxCPM 模型，"
                "请下载模型到对应目录（如 models/voice/tts/voxcpm/voxcpm2/）",
                self.model_dir,
            )

        return capabilities

    def _check_model_files(self, variant_dir: Path) -> bool:
        """检查目录内是否有模型文件。

        VoxCPM 模型为单文件格式（非 diffusers 目录格式），通常包含：
            - config.json（HF 模型标准标志文件，优先检查）
            - model.safetensors（VoxCPM2 主权重）
            - pytorch_model.bin（VoxCPM-0.5B 主权重）
            - audiovae.pth（AudioVAE 权重）
            - tokenizer.json / tokenizer_config.json
        """
        # 优先检查 config.json（两个变体都有，HF 模型标准标志）
        if (variant_dir / "config.json").exists():
            return True

        # 兼容 diffusers 目录格式（备用）
        if (variant_dir / "model_index.json").exists():
            return True

        # 检查模型权重文件
        for pattern in ("*.safetensors", "*.bin", "*.pt", "*.ckpt"):
            if list(variant_dir.rglob(pattern)):
                return True

        return False

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析模型路径。

        Args:
            variant_name: 变体名（如 voxcpm2）

        Returns:
            模型目录路径，不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                path = self.model_dir / variant_def["subdir"]
                return path if path.exists() else None
        return None
