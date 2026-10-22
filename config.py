"""VoxCPM TTS 引擎配置。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.2.1 双源真相消除策略：
    - MODEL_SUBDIR 默认读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ===== 常量 =====

ENGINE_ID = "voxcpm_tts_c4a8f2e7"
ENGINE_NAME = "VoxCPM-TTS"
CAPABILITY = "text-to-speech"
ENGINE_TYPE = "voice"

# 默认端口（与 config/engines/voxcpm_tts_c4a8f2e7.yaml 一致）
DEFAULT_PORT = 7003

# 模型目录布局（相对 MODELS_ROOT）
# v0.2.1：优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
_DEFAULT_MODEL_SUBDIR = "voice/tts/voxcpm/"


def _resolve_model_subdir() -> str:
    """解析模型子目录（v0.2.1：优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()

# v0.2.15+ 日志统一到 logs/engines/（项目顶层）
# 旧的 LOG_SUBDIR = "logs/"（相对 data_root）已废弃


@dataclass
class EngineConfig:
    """VoxCPM TTS 引擎运行配置。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础 =====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"

    # ===== 路径 =====
    models_root: Path = Path("models")
    data_root: Path = Path("data")
    project_root: Path = Path(".")
    # v0.2.5+ 输出文件根目录统一为 output（项目顶层目录）
    # 规范：{project_root}/output/{category}/{filename}
    output_root: Path = Path("output")

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # ===== 设备 =====
    # auto: 自动检测 GPU / cpu: 强制 CPU / cuda:0: 指定 GPU
    device: str = "auto"

    # ===== 日志 =====
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关
        self.model_dir = self.models_root / MODEL_SUBDIR

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = self.project_root / "logs" / "engines"

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "log_level": self.log_level,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="VoxCPM TTS 引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")

    args = parser.parse_args()

    # 环境变量 fallback
    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))

    config = EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        log_level=args.log_level,
        models_root=models_root,
    )

    return config
