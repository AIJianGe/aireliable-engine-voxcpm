"""VoxCPM Python API 封装。

封装 VoxCPM 的 from_pretrained / generate / generate_streaming，
提供线程安全的模型加载和推理接口。

参考：
    - VoxCPM Python API: from voxcpm import VoxCPM
    - VoxCPM generate(text, reference_wav_path, prompt_wav_path, prompt_text, ...)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import EngineConfig

logger = logging.getLogger(__name__)

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为：显存预检不足时抛错）
#   gpu     ：强制全 GPU，显存不足直接报错（与 auto 的预检行为一致）
#   gpu+cpu ：GPU 优先，显存不足时警告 + 降级 CPU 加载
#   cpu     ：强制 CPU（跳过显存预检）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _read_device_policy() -> str:
    """读取设备策略（v0.3.5+，平台注入 AIRELIABLE_DEVICE_POLICY）。"""
    policy = (os.environ.get("AIRELIABLE_DEVICE_POLICY", "auto").strip() or "auto")
    if policy not in DEVICE_POLICIES:
        logger.warning("非法设备策略 %r（合法值：%s），回退 auto", policy, DEVICE_POLICIES)
        return "auto"
    return policy


class VoxCPMWrapper:
    """VoxCPM 模型封装器。

    封装 VoxCPM 的 Python API，提供：
        - 线程安全的模型加载/卸载
        - 四种合成模式的参数构造
        - 设备选择（auto/cpu/cuda）
        - 流式生成支持
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._lock = threading.Lock()
        self._loaded_variant: Optional[str] = None
        self._sample_rate: int = 48000
        # v0.3.5+ 设备策略（平台注入 AIRELIABLE_DEVICE_POLICY）
        self._device_policy: str = _read_device_policy()
        # v0.2.5+ 启动时立即解析真实设备，避免 /status 在未加载模型时
        # 误报 "device=cpu"（实际 torch 已是 CUDA 版时会让用户误解切换未生效）
        # 未加载模型时返回解析结果，加载模型后会更新为实际使用的设备
        self._device: str = self._resolve_device(config.device)
        # v0.3.5+ cpu 策略强制 CPU（优先于 auto 检测与 config.device）
        if self._device_policy == "cpu":
            self._device = "cpu"
        # v0.2.5+ 同时获取设备显示名称（如 "NVIDIA GeForce RTX 4090"），用于 UI 友好显示
        self._device_name: str = self._get_device_name(self._device)
        logger.info(
            "VoxCPMWrapper 初始化: config.device=%s, resolved_device=%s, device_name=%s, device_policy=%s",
            config.device, self._device, self._device_name, self._device_policy,
        )

    @property
    def device_name(self) -> str:
        """设备显示名称（如 "NVIDIA GeForce RTX 4090" / "CPU"）。"""
        return self._device_name

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._model is not None

    @property
    def loaded_variant(self) -> Optional[str]:
        """已加载的变体名。"""
        return self._loaded_variant

    @property
    def sample_rate(self) -> int:
        """输出采样率。"""
        return self._sample_rate

    @property
    def device(self) -> str:
        """当前推理设备。"""
        return self._device

    def load(self, variant_name: str, variant_path: str, device: str = "auto") -> None:
        """加载 VoxCPM 模型。

        Args:
            variant_name: variant 名称（如 "voxcpm2"），用于状态查询
            variant_path: 模型目录路径，用于加载权重
            device: 推理设备（auto/cpu/cuda:0）

        Raises:
            RuntimeError: 加载失败

        v0.2.5+ 修复:`_loaded_variant` 存 variant 名而非路径,避免 UI 层
        比较时永远不相等导致每次推理都触发 unload + load。
        """
        with self._lock:
            if self._model is not None:
                logger.warning("模型已加载（variant=%s），先卸载", self._loaded_variant)
                self._unload_unsafe()

            try:
                from voxcpm import VoxCPM
            except ImportError as e:
                raise RuntimeError(
                    f"voxcpm 包未安装: {e}。"
                    f"请在 envs/voice-voxcpm/ 环境中运行 `pip install voxcpm`"
                ) from e

            # 解析设备
            self._device = self._resolve_device(device)
            # v0.3.5+ cpu 策略强制 CPU（优先于 auto 检测与显式 device 参数）
            if self._device_policy == "cpu":
                self._device = "cpu"
            self._device_name = self._get_device_name(self._device)
            logger.info(
                "加载 VoxCPM 模型: name=%s, path=%s, device=%s, device_policy=%s",
                variant_name, variant_path, self._device, self._device_policy,
            )

            # v0.2.5+ 显存预检:加载前检查空闲显存,不够直接报友好错误
            # 详见 architecture.md §3.5 显存预检机制
            # v0.3.5+ gpu+cpu 策略：显存不足时不抛错，返回 False 由此处降级 CPU
            if not self._check_vram_available(self._device, required_gb=4.0):
                logger.warning(
                    "gpu+cpu 策略显存不足，VoxCPM 模型降级 CPU 加载（推理速度会显著变慢）"
                )
                self._device = "cpu"
                self._device_name = self._get_device_name("cpu")

            # 加载模型
            # v0.2.5+：显式传递 device 给 from_pretrained，避免 VoxCPM 内部
            # resolve_runtime_device(None, config.device) 与 wrapper 的 _resolve_device
            # 结果不一致（此前未传 device，导致 _resolve_device 的结果被忽略）。
            try:
                self._model = VoxCPM.from_pretrained(
                    variant_path,
                    load_denoiser=False,
                    device=self._device,
                )
                # 获取采样率
                if hasattr(self._model, "tts_model"):
                    self._sample_rate = getattr(
                        self._model.tts_model, "sample_rate", 48000
                    )
                else:
                    self._sample_rate = 48000

                # v0.2.5+ 必须存 variant 名（如 "voxcpm2"），禁止存路径
                # 详见 development-guide.md §28.19.3 loaded_variant 字段规范
                self._loaded_variant = variant_name
                logger.info(
                    "VoxCPM 模型加载成功: variant=%s, sample_rate=%d, device=%s",
                    variant_name, self._sample_rate, self._device,
                )
            except Exception as e:
                self._model = None
                self._loaded_variant = None
                err_str = str(e)
                # v0.2.5+ OOM 错误友好化
                if "CUDA out of memory" in err_str or "显存不足" in err_str:
                    raise RuntimeError(
                        "GPU 显存不足,无法加载 VoxCPM 模型。"
                        "建议:1) 关闭其他占用显存的程序 2) 切换到 CPU 模式 3) 卸载其他已加载的引擎"
                    ) from e
                raise RuntimeError(f"VoxCPM 模型加载失败: {e}") from e

    def _check_vram_available(self, device: str, required_gb: float = 4.0) -> bool:
        """v0.2.5+ 加载前预检显存。

        空闲显存 < required_gb 时的行为（v0.3.5+ 按设备策略区分）：
            - gpu+cpu 策略：记录警告并返回 False（调用方降级 CPU 加载）
            - 其余策略（auto/gpu）：抛 RuntimeError(中文提示 + 解决建议),
              不尝试加载,避免 PyTorch OOM 异常的原始英文信息暴露给用户

        Args:
            device: 推理设备
            required_gb: 模型预估需要的显存(GB),VoxCPM2 ≈ 4GB

        Returns:
            bool: True=预检通过（或 CPU 模式/检测异常跳过）；False=显存不足
                  （仅 gpu+cpu 策略会返回 False，其余策略显存不足直接抛错）

        详见 architecture.md §3.5 显存预检机制
        """
        if not device.startswith("cuda"):
            return True  # CPU 模式无需预检

        try:
            import torch
            if not torch.cuda.is_available():
                return True
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < required_gb:
                if self._device_policy == "gpu+cpu":
                    logger.warning(
                        "显存预检不足: free=%.2fGB < required=%.2fGB（gpu+cpu 策略，降级 CPU）",
                        free_gb, required_gb,
                    )
                    return False
                logger.warning(
                    "显存预检失败: free=%.2fGB < required=%.2fGB",
                    free_gb, required_gb,
                )
                raise RuntimeError(
                    f"GPU 显存不足:空闲 {free_gb:.2f}GB < 需要 {required_gb}GB。"
                    f"建议:1) 关闭其他占用显存的程序 2) 切换到 CPU 模式 3) 卸载其他已加载的引擎"
                )
            logger.info("显存预检通过: free=%.2fGB >= required=%.2fGB", free_gb, required_gb)
            return True
        except RuntimeError as e:
            if "显存不足" in str(e):
                raise  # 友好错误直接抛出
            # 其他异常(如 mem_get_info 失败)不阻塞加载,记录警告
            logger.warning("显存预检异常(跳过): %s", e)
        except Exception as e:
            logger.warning("显存预检异常(跳过): %s", e)
        return True

    def unload(self) -> None:
        """卸载模型，释放显存。"""
        with self._lock:
            self._unload_unsafe()

    def _unload_unsafe(self) -> None:
        """卸载模型（无锁版本）。"""
        if self._model is None:
            return

        logger.info("卸载 VoxCPM 模型: variant=%s", self._loaded_variant)
        self._model = None
        self._loaded_variant = None

        # 尝试释放显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清空 CUDA 缓存")
        except ImportError:
            pass

    def generate(
        self,
        text: str,
        mode: str = "tts",
        voice_description: str = "",
        reference_audio_path: Optional[str] = None,
        prompt_audio_path: Optional[str] = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: int = -1,
    ) -> tuple[np.ndarray, int]:
        """同步生成音频。

        Args:
            text: 要合成的文本
            mode: 合成模式（tts/design/clone/ultimate_clone）
            voice_description: 音色描述（Design 模式）
            reference_audio_path: 参考音频路径（Clone 模式）
            prompt_audio_path: 提示音频路径（Ultimate Clone 模式）
            prompt_text: 转写文本（Ultimate Clone 模式）
            cfg_value: CFG 引导强度
            inference_timesteps: 推理步数
            seed: 随机种子（-1=随机）

        Returns:
            (samples, sample_rate): 音频采样数据和采样率

        Raises:
            RuntimeError: 模型未加载或生成失败
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("模型未加载，请先调用 load()")

            # 构造参数
            kwargs: dict[str, Any] = {
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            }

            # seed 处理
            # VoxCPM 2.0.3 的 _generate() 不接受 seed 参数（包内无随机种子控制能力），
            # 传 seed 会抛 TypeError: unexpected keyword argument 'seed'。
            # v0.3.6+ 暂不支持 seed，保留参数仅为 API 兼容，记 WARNING 后忽略。
            # 后续 VoxCPM 包若支持 seed，可在此处恢复 kwargs["seed"] = seed。
            if seed >= 0:
                logger.warning(
                    "VoxCPM 2.0.3 暂不支持 seed 参数，已忽略 seed=%d（不影响生成结果）",
                    seed,
                )

            # 根据模式构造文本和参考音频
            if mode == "tts":
                # 基础 TTS：直接使用文本
                actual_text = text

            elif mode == "design":
                # 声音设计：自动包裹 () + 处理中文括号
                # VoxCPM2 官方语法：(音色描述)要合成的文本
                # 模型识别 () 作为分隔符,() 内的内容不被朗读
                # v0.2.5+ 代码自动处理三种输入,用户无需手动加括号
                # 详见 requirements.md REQ-v0.2.5-TTS-006A
                if voice_description:
                    desc = voice_description.strip()
                    if desc.startswith("(") and desc.endswith(")"):
                        # 用户已输入英文括号,保持不变
                        actual_text = f"{desc}{text}"
                    elif desc.startswith("（") and desc.endswith("）"):
                        # 中文括号转英文括号
                        inner = desc[1:-1]
                        actual_text = f"({inner}){text}"
                    else:
                        # 无括号,自动包裹
                        actual_text = f"({desc}){text}"
                else:
                    actual_text = text

            elif mode == "clone":
                # 可控克隆：参考音频 + 文本
                if not reference_audio_path:
                    raise ValueError("Clone 模式需要 reference_audio_path")
                kwargs["reference_wav_path"] = reference_audio_path
                actual_text = text

            elif mode == "ultimate_clone":
                # 极致克隆：提示音频 + 转写文本 + 目标文本
                if not prompt_audio_path:
                    raise ValueError("Ultimate Clone 模式需要 prompt_audio_path")
                if not prompt_text:
                    raise ValueError("Ultimate Clone 模式需要 prompt_text")
                kwargs["prompt_wav_path"] = prompt_audio_path
                kwargs["prompt_text"] = prompt_text
                actual_text = text

            else:
                raise ValueError(f"未知合成模式: {mode}")

            logger.info(
                "VoxCPM 生成: mode=%s, text=%r, cfg=%.1f, steps=%d, seed=%d",
                mode,
                actual_text[:50] + ("..." if len(actual_text) > 50 else ""),
                cfg_value,
                inference_timesteps,
                seed,
            )

            # 调用 VoxCPM generate
            try:
                wav = self._model.generate(
                    text=actual_text,
                    **kwargs,
                )
                # VoxCPM generate 返回 numpy 数组
                if isinstance(wav, tuple):
                    # 某些版本返回 (wav, sr)
                    samples, sr = wav
                else:
                    samples = wav
                    sr = self._sample_rate

                logger.info(
                    "VoxCPM 生成完成: samples_shape=%s, duration=%.2fs, sr=%d",
                    samples.shape,
                    len(samples) / sr,
                    sr,
                )
                return samples, sr

            except Exception as e:
                logger.error("VoxCPM 生成失败: %s", e)
                raise RuntimeError(f"VoxCPM 生成失败: {e}") from e

    def generate_streaming(
        self,
        text: str,
        **kwargs: Any,
    ):
        """流式生成音频（返回 chunk 迭代器）。

        Args:
            text: 要合成的文本
            **kwargs: 其他参数（同 generate）

        Yields:
            numpy.ndarray: 音频 chunk

        Raises:
            RuntimeError: 模型未加载或生成失败
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("模型未加载，请先调用 load()")

            if not hasattr(self._model, "generate_streaming"):
                raise RuntimeError("当前 VoxCPM 版本不支持流式生成")

            try:
                yield from self._model.generate_streaming(text=text, **kwargs)
            except Exception as e:
                logger.error("VoxCPM 流式生成失败: %s", e)
                raise RuntimeError(f"VoxCPM 流式生成失败: {e}") from e

    def _resolve_device(self, device: str) -> str:
        """解析设备字符串。

        Args:
            device: auto / cpu / cuda:0 / cuda:1

        Returns:
            实际设备字符串
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
                return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _get_device_name(self, device: str) -> str:
        """获取设备的显示名称。

        Args:
            device: cuda:0 / cpu / cuda:1 等

        Returns:
            设备显示名称：
                - cuda:0 → "NVIDIA GeForce RTX 4090"（实际 GPU 名称）
                - cpu    → "CPU"
        """
        if device == "cpu":
            return "CPU"
        # cuda:0 / cuda:1 / cuda 等
        try:
            import torch
            if torch.cuda.is_available():
                # 解析 GPU 索引（cuda:0 → 0，cuda → 0）
                if ":" in device:
                    gpu_idx = int(device.split(":")[1])
                else:
                    gpu_idx = 0
                name = torch.cuda.get_device_name(gpu_idx)
                return name or device
        except (ImportError, ValueError, RuntimeError):
            pass
        return device

    def get_vram_used(self) -> int:
        """获取已用显存（MB）。"""
        try:
            import torch
            if torch.cuda.is_available() and "cuda" in self._device:
                return int(torch.cuda.memory_allocated() / 1024 / 1024)
        except ImportError:
            pass
        return 0
