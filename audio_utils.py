"""音频处理工具。

提供音频保存、格式转换等功能。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def save_audio(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    format: str = "wav",
) -> Path:
    """保存音频到文件。

    Args:
        samples: 音频采样数据，形状 (samples,) 或 (channels, samples)
        sample_rate: 采样率
        output_path: 输出路径
        format: 音频格式（wav/mp3/flac）

    Returns:
        实际保存路径
    """
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保数据类型为 float32
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    # 归一化到 [-1.0, 1.0]
    max_val = np.max(np.abs(samples))
    if max_val > 1.0:
        samples = samples / max_val

    try:
        import soundfile as sf

        # 根据格式选择子类型
        if format == "wav":
            subtype = "PCM_16"
        elif format == "flac":
            subtype = "PCM_16"
        elif format == "mp3":
            # soundfile 可能不支持 mp3 写入，fallback 到 wav
            logger.warning("soundfile 不支持 mp3 写入，fallback 到 wav")
            format = "wav"
            output_path = output_path.with_suffix(".wav")
            subtype = "PCM_16"
        else:
            subtype = "PCM_16"

        sf.write(
            str(output_path),
            samples,
            sample_rate,
            subtype=subtype,
        )

        duration = len(samples) / sample_rate
        logger.info(
            "音频已保存: %s, format=%s, sr=%d, duration=%.2fs",
            output_path,
            format,
            sample_rate,
            duration,
        )
        return output_path

    except ImportError:
        raise RuntimeError("soundfile 未安装，无法保存音频")
    except Exception as e:
        raise RuntimeError(f"保存音频失败: {e}") from e


def get_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """计算音频时长（秒）。"""
    if samples.ndim == 1:
        return len(samples) / sample_rate
    return samples.shape[-1] / sample_rate


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """归一化音频到 [-1.0, 1.0]。"""
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        return samples / max_val
    return samples
