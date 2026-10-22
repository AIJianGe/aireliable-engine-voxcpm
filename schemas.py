"""VoxCPM TTS 引擎 Pydantic 数据模型。

扩展 SDK 的 InferRequest，添加 TTS 专属字段。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ===== 合成模式枚举 =====

# tts: 基础文本转语音
# design: 声音设计（文字描述音色）
# clone: 可控克隆（参考音频 + 文本）
# ultimate_clone: 极致克隆（提示音频 + 转写文本）
TTS_MODES = ("tts", "design", "clone", "ultimate_clone")


class TTSInputs(BaseModel):
    """TTS 推理输入参数（放入 InferRequest.inputs）。

    所有模式共用此 schema，根据 mode 字段决定使用哪些参数。
    """

    # ===== 必填 =====
    text: str = Field(..., description="要合成的文本")
    mode: str = Field("tts", description=f"合成模式：{TTS_MODES}")

    # ===== Design 模式 =====
    voice_description: str = Field(
        "", description="音色描述（Design 模式），如 (A young woman, gentle voice)"
    )

    # ===== Clone 模式 =====
    # 参考音频路径（通过 InferRequest.audios[role=speaker].path 传入，
    # 此字段作为备选，直接指定路径）
    reference_audio_path: Optional[str] = Field(
        None, description="参考音频路径（Clone 模式）"
    )

    # ===== Ultimate Clone 模式 =====
    prompt_text: str = Field(
        "", description="转写文本（Ultimate Clone 模式，对应提示音频的内容）"
    )
    # 提示音频路径（通过 InferRequest.audios[role=prompt].path 传入）
    prompt_audio_path: Optional[str] = Field(
        None, description="提示音频路径（Ultimate Clone 模式）"
    )

    # ===== 推理参数 =====
    cfg_value: float = Field(2.0, ge=0.1, le=10.0, description="CFG 引导强度")
    inference_timesteps: int = Field(10, ge=1, le=100, description="推理步数")
    seed: int = Field(-1, ge=-1, description="随机种子（-1=随机）")

    # ===== 输出参数 =====
    output_format: str = Field("wav", description="输出音频格式：wav/mp3/flac")


class TTSCapability(BaseModel):
    """TTS 引擎能力声明项。"""

    name: str = Field(..., description="变体名（如 voxcpm2）")
    alias: str = Field(..., description="显示名（如 VoxCPM2）")
    precision: str = Field("fp16", description="精度")
    vram_required: int = Field(0, description="显存需求（MB）")
    sample_rate: int = Field(48000, description="输出采样率（Hz）")
    languages: list[str] = Field(
        default_factory=lambda: ["zh", "en"],
        description="支持的语言列表",
    )
    modes: list[str] = Field(
        default_factory=lambda: list(TTS_MODES),
        description="支持的合成模式",
    )
    loaded: bool = Field(False, description="是否已加载")


class TTSStatusExtra(BaseModel):
    """TTS 引擎状态附加信息。"""

    device: str = Field("cpu", description="当前推理设备")
    vram_used: int = Field(0, description="已用显存（MB）")
    sample_rate: int = Field(48000, description="输出采样率")
    loaded_variant: Optional[str] = Field(None, description="已加载的模型变体")
