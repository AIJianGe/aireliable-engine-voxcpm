"""VoxCPM TTS 引擎 Gradio Web UI。

页面元素：
    - 状态栏（引擎状态 / 已加载模型 / 显存）
    - 模型选择 + 加载/卸载/刷新按钮
    - 四模式 Tab（TTS / Design / Clone / Ultimate Clone）
    - 文本输入框
    - [Design] 音色描述输入
    - [Clone] 参考音频上传
    - [Ultimate] 提示音频上传 + 转写文本
    - 高级参数（cfg_value / inference_timesteps / seed / output_format）
    - 生成按钮 + 进度状态
    - 音频播放器 + 下载
    - 日志显示

设计原则：
    - 调用自身 API（不直接调 engine.py），保证 UI 和 API 行为一致
    - SSE 流式消费进度帧（tts.start → tts.completed）
    - 自动加载逻辑（未加载时自动 /load）
    - 支持 iframe 嵌入（root_path 由 server.py 配置）
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import gradio as gr
import httpx

logger = logging.getLogger(__name__)


def create_ui(engine_url: str, theme: Optional[gr.Theme] = None) -> gr.Blocks:
    """创建 VoxCPM TTS 引擎 Web UI。

    Args:
        engine_url: 引擎 HTTP 地址（如 http://127.0.0.1:7003）
        theme: Gradio 主题（Gradio 6.0+ 不再传给 Blocks，由 mount_gradio_app 应用）

    Returns:
        gr.Blocks: UI 实例
    """
    # Gradio 6.0+ 将 theme 从 Blocks 构造器移到 launch/mount_gradio_app
    # 这里保留 theme 参数仅用于向后兼容，实际应用在 server.py 的 mount_gradio_app
    with gr.Blocks(title="VoxCPM TTS 语音合成") as demo:
        gr.Markdown("# VoxCPM TTS 语音合成引擎")
        gr.Markdown("多语言 TTS / 声音设计 / 可控克隆 / 极致克隆")

        # ===== 状态栏 =====
        with gr.Row():
            status_display = gr.Textbox(
                label="引擎状态",
                interactive=False,
                value="未连接",
                scale=2,
            )
            variant_display = gr.Textbox(
                label="已加载模型",
                interactive=False,
                value="-",
                scale=1,
            )
            vram_display = gr.Textbox(
                label="显存占用",
                interactive=False,
                value="-",
                scale=1,
            )

        # ===== 模型选择 + 操作按钮 =====
        with gr.Row():
            variant_select = gr.Dropdown(
                label="模型变体（Variant）",
                choices=[],
                value=None,
                interactive=True,
                info="选择要加载的 VoxCPM 模型",
                scale=2,
            )
            refresh_btn = gr.Button("刷新状态", variant="secondary", scale=1)
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
            unload_btn = gr.Button("卸载模型", variant="stop", scale=1)

        # ===== 四模式 Tab =====
        with gr.Tabs() as tabs:
            # ----- Tab 1: 基础 TTS -----
            with gr.Tab("基础 TTS") as tab_tts:
                gr.Markdown("**基础文本转语音**：输入文本，合成语音")
                tts_text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="输入要合成的文本...",
                    lines=5,
                    max_lines=10,
                )

            # ----- Tab 2: 声音设计 -----
            with gr.Tab("声音设计") as tab_design:
                gr.Markdown("**声音设计**：用文字描述音色（如 *A young woman, gentle voice*）")
                design_description = gr.Textbox(
                    label="音色描述",
                    placeholder="输入音色描述(不需要加括号),如:一个大叔沙哑声音,温和又带些奸诈",
                    lines=2,
                )
                design_text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="输入要合成的文本...",
                    lines=5,
                    max_lines=10,
                )

            # ----- Tab 3: 可控克隆 -----
            with gr.Tab("可控克隆") as tab_clone:
                gr.Markdown("**可控克隆**：上传参考音频，克隆音色合成新文本")
                clone_reference_audio = gr.Audio(
                    label="参考音频（speaker）",
                    type="filepath",
                    sources=["upload"],
                )
                clone_text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="输入要用克隆音色合成的文本...",
                    lines=5,
                    max_lines=10,
                )

            # ----- Tab 4: 极致克隆 -----
            with gr.Tab("极致克隆") as tab_ultimate:
                gr.Markdown("**极致克隆**：上传提示音频 + 转写文本，最大化还原音色")
                ultimate_prompt_audio = gr.Audio(
                    label="提示音频（prompt）",
                    type="filepath",
                    sources=["upload"],
                )
                ultimate_prompt_text = gr.Textbox(
                    label="转写文本（提示音频的内容）",
                    placeholder="输入提示音频对应的文字内容...",
                    lines=3,
                )
                ultimate_text_input = gr.Textbox(
                    label="目标合成文本",
                    placeholder="输入要合成的新文本...",
                    lines=5,
                    max_lines=10,
                )

        # ===== 高级参数 =====
        with gr.Accordion("高级参数", open=False):
            with gr.Row():
                cfg_slider = gr.Slider(
                    label="CFG 引导强度",
                    minimum=0.1,
                    maximum=10.0,
                    value=2.0,
                    step=0.1,
                    info="越高越贴合文本，越低音色越多样",
                )
                timesteps_slider = gr.Slider(
                    label="推理步数",
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1,
                    info="越多质量越好但越慢",
                )
                seed_input = gr.Number(
                    label="随机种子",
                    value=-1,
                    info="-1 = 随机",
                )
                format_select = gr.Dropdown(
                    label="输出格式",
                    choices=["wav", "mp3", "flac"],
                    value="wav",
                )

        # ===== 生成按钮 + 进度 =====
        with gr.Row():
            generate_btn = gr.Button("生成语音", variant="primary", scale=2)
            progress_display = gr.Textbox(
                label="进度",
                interactive=False,
                value="就绪",
                scale=3,
            )

        # ===== 结果区 =====
        with gr.Row():
            audio_output = gr.Audio(
                label="合成结果",
                interactive=False,
                scale=3,
            )
            download_link = gr.Textbox(
                label="下载链接",
                interactive=False,
                value="-",
                scale=2,
            )

        # ===== 日志 =====
        log_display = gr.Textbox(
            label="日志",
            interactive=False,
            lines=4,
            max_lines=8,
        )

        # ===== 事件绑定 =====

        # 页面加载 + 刷新按钮
        def _on_refresh():
            return refresh_all(engine_url)

        demo.load(
            fn=_on_refresh,
            outputs=[status_display, variant_display, vram_display, variant_select],
        )
        refresh_btn.click(
            fn=_on_refresh,
            outputs=[status_display, variant_display, vram_display, variant_select],
        )

        # 加载按钮
        def _on_load(variant):
            return load_variant(engine_url, variant)

        load_btn.click(
            fn=_on_load,
            inputs=[variant_select],
            outputs=[status_display, variant_display, vram_display, log_display],
        )

        # 卸载按钮
        def _on_unload():
            return unload_variant(engine_url)

        unload_btn.click(
            fn=_on_unload,
            outputs=[status_display, variant_display, vram_display, log_display],
        )

        # 生成按钮（根据当前 Tab 决定参数）
        # 用一个隐藏的状态组件记录当前 Tab
        current_tab = gr.State("tts")

        # Tab 切换时更新 current_tab 状态
        tab_tts.select(fn=lambda: "tts", outputs=current_tab)
        tab_design.select(fn=lambda: "design", outputs=current_tab)
        tab_clone.select(fn=lambda: "clone", outputs=current_tab)
        tab_ultimate.select(fn=lambda: "ultimate_clone", outputs=current_tab)

        # 生成函数（生成器，流式更新进度）
        def _on_generate(
            tab, variant,
            tts_text, design_desc, design_text,
            clone_ref, clone_text,
            ultimate_prompt_aud, ultimate_prompt_text, ultimate_text,
            cfg, timesteps, seed, fmt,
        ):
            yield from generate_tts(
                engine_url, tab, variant,
                tts_text, design_desc, design_text,
                clone_ref, clone_text,
                ultimate_prompt_aud, ultimate_prompt_text, ultimate_text,
                cfg, timesteps, seed, fmt,
            )

        generate_btn.click(
            fn=_on_generate,
            inputs=[
                current_tab, variant_select,
                tts_text_input, design_description, design_text_input,
                clone_reference_audio, clone_text_input,
                ultimate_prompt_audio, ultimate_prompt_text, ultimate_text_input,
                cfg_slider, timesteps_slider, seed_input, format_select,
            ],
            outputs=[progress_display, audio_output, download_link, log_display],
        )

    return demo


# ===== API 调用函数 =====

def refresh_all(engine_url: str) -> tuple:
    """刷新状态、variant 列表。"""
    try:
        status = _http_get(engine_url, "/status")
        caps = _http_get(engine_url, "/capabilities")

        status_text = f"state={status.get('state', '?')}"
        extra = status.get("extra", {})
        # v0.2.5+ 优先显示显卡名称（如 "NVIDIA GeForce RTX 4090"），无名称时回退到 device 字符串
        device_display = extra.get("device_name") or extra.get("device")
        if device_display:
            status_text += f" | device={device_display}"
        if extra.get("is_loaded"):
            status_text += " | 已加载"

        variant_display = status.get("loaded_variant") or "-"
        vram_display = status.get("vram_used", "-")

        # variant 列表（显示 name）
        variants = []
        for v in caps.get("variants", []):
            name = v.get("name", "")
            sample_rate = v.get("sample_rate", 0)
            label = f"{name} ({sample_rate//1000}kHz)" if sample_rate else name
            variants.append(label)

        # 默认选第一个
        default = variants[0].split(" ")[0] if variants else None

        return status_text, variant_display, vram_display, gr.update(choices=variants, value=default)
    except Exception as e:
        logger.error("刷新失败: %s", e)
        return f"连接失败: {e}", "-", "-", gr.update(choices=[])


def load_variant(engine_url: str, variant_label: str) -> tuple:
    """加载 variant。"""
    if not variant_label:
        return "请先选择 variant", "-", "-", "未选择 variant"

    # variant_label 格式可能是 "voxcpm2 (48kHz)"，取第一段
    variant = variant_label.split(" ")[0]

    try:
        response = _http_post(engine_url, "/load", {"variant": variant}, timeout=180.0)
        status_text = f"state={response.get('state', '?')}"
        variant_display = response.get("variant", variant)
        vram = response.get("vram_used", "-")
        load_time = response.get("load_time", 0)
        log = f"加载 {variant} 完成，耗时 {load_time:.2f}s，显存 {vram}"
        return status_text, variant_display, vram, log
    except Exception as e:
        logger.error("加载失败: %s", e)
        return f"加载失败: {e}", "-", "-", f"加载失败: {e}"


def unload_variant(engine_url: str) -> tuple:
    """卸载模型。"""
    try:
        response = _http_post(engine_url, "/unload", {}, timeout=30.0)
        return "state=idle", "-", response.get("vram_freed", "0MB"), "卸载完成"
    except Exception as e:
        return f"卸载失败: {e}", "-", "-", f"卸载失败: {e}"


def generate_tts(
    engine_url: str,
    tab: str,
    variant_label: str,
    tts_text: str,
    design_desc: str,
    design_text: str,
    clone_ref: Optional[str],
    clone_text: str,
    ultimate_prompt_aud: Optional[str],
    ultimate_prompt_text: str,
    ultimate_text: str,
    cfg: float,
    timesteps: int,
    seed: int,
    fmt: str,
):
    """流式 TTS 生成。

    自动加载逻辑：
        1. 发送前检查 /status
        2. 如果未加载 → 自动 /load
        3. 加载完成后执行 /infer（SSE 流式）
    """
    log_lines: list[str] = []

    # ----- 1. 根据 Tab 解析参数 -----
    variant = (variant_label or "").split(" ")[0]
    if not variant:
        yield "请先选择模型", None, "-", "未选择 variant"
        return

    mode_map = {
        "tts": "tts",
        "design": "design",
        "clone": "clone",
        "ultimate_clone": "ultimate_clone",
    }
    mode = mode_map.get(tab, "tts")

    # 解析文本和音频
    if mode == "tts":
        text = tts_text or ""
        if not text.strip():
            yield "请输入合成文本", None, "-", "文本为空"
            return
        audios = []
        inputs_extra = {}

    elif mode == "design":
        text = design_text or ""
        if not text.strip():
            yield "请输入合成文本", None, "-", "文本为空"
            return
        audios = []
        inputs_extra = {"voice_description": design_desc or ""}

    elif mode == "clone":
        text = clone_text or ""
        if not text.strip():
            yield "请输入合成文本", None, "-", "文本为空"
            return
        if not clone_ref:
            yield "请上传参考音频", None, "-", "参考音频为空"
            return
        audios = [{"path": clone_ref, "role": "speaker"}]
        inputs_extra = {}

    elif mode == "ultimate_clone":
        text = ultimate_text or ""
        if not text.strip():
            yield "请输入目标合成文本", None, "-", "目标文本为空"
            return
        if not ultimate_prompt_aud:
            yield "请上传提示音频", None, "-", "提示音频为空"
            return
        if not (ultimate_prompt_text or "").strip():
            yield "请输入转写文本", None, "-", "转写文本为空"
            return
        audios = [{"path": ultimate_prompt_aud, "role": "prompt"}]
        inputs_extra = {"prompt_text": ultimate_prompt_text}
    else:
        yield f"未知模式: {tab}", None, "-", f"未知 tab: {tab}"
        return

    # ----- 2. 自动加载 -----
    try:
        status = _http_get(engine_url, "/status")
    except Exception as e:
        yield "获取状态失败", None, "-", f"获取状态失败: {e}"
        return

    current_state = status.get("state", "unknown")
    # v0.2.5+ 防御性归一化:即使后端返回路径,也取最后一段作为 variant 名
    # 详见 development-guide.md §28.19.3 loaded_variant 字段规范
    raw_loaded = status.get("loaded_variant") or ""
    loaded_variant = raw_loaded.replace("\\", "/").rstrip("/").split("/")[-1].split(" ")[0]

    if current_state != "ready" or not loaded_variant:
        log_lines.append(f"模型未加载（state={current_state}），自动加载 {variant}")
        yield "正在加载模型...", None, "-", "\n".join(log_lines)
        try:
            _http_post(engine_url, "/load", {"variant": variant}, timeout=180.0)
            log_lines.append("加载完成")
            yield "模型加载完成，开始推理...", None, "-", "\n".join(log_lines)
        except Exception as e:
            yield "加载失败", None, "-", f"自动加载失败: {e}"
            return
    elif loaded_variant != variant:
        log_lines.append(f"切换模型: {loaded_variant} → {variant}")
        yield "正在切换模型...", None, "-", "\n".join(log_lines)
        try:
            _http_post(engine_url, "/unload", {}, timeout=30.0)
            _http_post(engine_url, "/load", {"variant": variant}, timeout=180.0)
            log_lines.append("切换完成")
            yield "模型切换完成，开始推理...", None, "-", "\n".join(log_lines)
        except Exception as e:
            yield "切换失败", None, "-", f"切换失败: {e}"
            return
    else:
        # v0.2.5+ 新增:模型已加载且 variant 匹配,直接推理
        log_lines.append(f"模型已加载（variant={loaded_variant}），直接推理")
        yield "模型已加载，开始推理...", None, "-", "\n".join(log_lines)

    # ----- 3. 构造推理请求 -----
    request_body = {
        "variant": variant,
        "inputs": {
            "text": text,
            "mode": mode,
            "cfg_value": float(cfg),
            "inference_timesteps": int(timesteps),
            "seed": int(seed),
            "output_format": fmt,
            **inputs_extra,
        },
        "audios": audios,
    }

    # ----- 4. SSE 流式推理 -----
    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream(
                "POST",
                f"{engine_url}/infer",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        frame = json.loads(data_str)
                    except Exception:
                        continue

                    stage = frame.get("stage", "")
                    progress = frame.get("progress", 0.0)

                    # 阶段进度
                    if stage == "tts.start":
                        msg = f"开始推理（{progress:.0%}）"
                        yield msg, None, "-", "\n".join(log_lines)

                    elif stage == "tts.model_ready":
                        msg = f"模型就绪（{progress:.0%}）"
                        yield msg, None, "-", "\n".join(log_lines)

                    elif stage == "tts.text_processing":
                        msg = f"文本处理中（{progress:.0%}）"
                        yield msg, None, "-", "\n".join(log_lines)

                    elif stage == "tts.generating":
                        msg = f"生成音频中...（{progress:.0%}）"
                        yield msg, None, "-", "\n".join(log_lines)

                    elif stage == "tts.saving":
                        msg = f"保存音频中（{progress:.0%}）"
                        yield msg, None, "-", "\n".join(log_lines)

                    elif stage == "completed":
                        result = frame.get("result", {})
                        timing = frame.get("timing", {})
                        audio_url = result.get("audio_url", "")
                        audio_path = result.get("audio_path", "")
                        duration = result.get("duration", 0)

                        # 拼完整 URL（用于下载链接文本框，可点击下载）
                        full_url = f"{engine_url}{audio_url}" if audio_url else ""

                        # v0.2.5+ gr.Audio 使用本地绝对路径 audio_path 播放，
                        # 绕过 Gradio safehttpx SSRF 防护对 127.0.0.1 的拦截
                        gradio_audio_value = audio_path if audio_path else full_url

                        log_lines.append(
                            f"完成: 时长 {duration:.2f}s, "
                            f"总耗时 {timing.get('total_seconds', 0):.2f}s"
                        )
                        yield "完成", gradio_audio_value, full_url, "\n".join(log_lines)
                        return

                    elif stage == "error":
                        err = frame.get("error", {})
                        msg = f"错误: {err.get('message', '?')}"
                        log_lines.append(msg)
                        yield msg, None, "-", "\n".join(log_lines)
                        return

    except Exception as e:
        logger.exception("TTS 推理失败")
        yield f"推理异常: {e}", None, "-", f"推理异常: {e}"


# ===== HTTP 工具 =====

def _http_get(base_url: str, path: str) -> dict:
    """GET 请求。"""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{base_url}{path}")
        response.raise_for_status()
        return response.json()


def _http_post(base_url: str, path: str, json_body: dict, timeout: float = 600.0) -> dict:
    """POST 请求。"""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}{path}", json=json_body)
        response.raise_for_status()
        return response.json()
