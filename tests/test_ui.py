"""UI 集成测试（T-062）。

覆盖：
- create_ui() 返回 gr.Blocks（需 gradio 安装，否则跳过）
- API 调用函数（refresh_all / load_variant / unload_variant）用 mock httpx
- generate_tts 流式生成器消费 SSE 帧
- 错误处理（连接失败、加载失败）

策略：
- API 函数不依赖 gradio 运行时，但 ui.py 顶层 `import gradio`，
  所以用 mock gradio 模块注入 sys.modules 来导入这些函数。
- create_ui 测试用 pytest.importorskip 真实 gradio。
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===== Mock gradio 模块（用于导入 ui.py 的 API 函数）=====


def _make_mock_gradio():
    """构造一个最小 mock gradio 模块，满足 ui.py 顶层 import。"""
    mock_gradio = MagicMock()
    # gr.themes.Soft
    mock_gradio.themes.Soft.return_value = MagicMock()
    # gr.update
    mock_gradio.update.return_value = {"__type__": "update"}
    # gr.Blocks 作为 context manager
    mock_gradio.Blocks.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_gradio.Blocks.return_value.__exit__ = MagicMock(return_value=False)
    return mock_gradio


@pytest.fixture
def mock_gradio():
    """注入 mock gradio 模块。"""
    mock = _make_mock_gradio()
    with patch.dict("sys.modules", {"gradio": mock}):
        yield mock


@pytest.fixture
def ui_module(mock_gradio):
    """导入 ui 模块（gradio 已 mock）。"""
    import importlib
    import engines.voxcpm_tts_c4a8f2e7.ui as ui
    importlib.reload(ui)
    return ui


# ===== Mock httpx 响应 =====


def _make_mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.raise_for_status.return_value = None
    return resp


def _make_mock_stream_response(lines: list[str], status_code: int = 200):
    """构造一个 mock 的 httpx 流式响应。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    resp.iter_lines.return_value = iter(lines)
    return resp


# ===== 测试类 =====


class TestRefreshAll:
    """refresh_all() 测试。"""

    def test_success(self, ui_module):
        status_resp = _make_mock_response(json_data={
            "state": "ready",
            "loaded_variant": "voxcpm2",
            "vram_used": "2048MB",
            "extra": {"device": "cuda:0", "is_loaded": True},
        })
        caps_resp = _make_mock_response(json_data={
            "variants": [
                {"name": "voxcpm2", "sample_rate": 48000},
                {"name": "voxcpm-0.5b", "sample_rate": 16000},
            ],
        })

        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.get.side_effect = [status_resp, caps_resp]

            result = ui_module.refresh_all("http://127.0.0.1:7003")

        assert len(result) == 4
        status_text, variant_display, vram_display, variant_update = result
        assert "state=ready" in status_text
        assert "device=cuda:0" in status_text
        assert variant_display == "voxcpm2"
        assert vram_display == "2048MB"

    def test_connection_failure(self, ui_module):
        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.get.side_effect = ConnectionError("refused")

            result = ui_module.refresh_all("http://127.0.0.1:7003")

        assert len(result) == 4
        status_text, variant_display, vram_display, variant_update = result
        assert "连接失败" in status_text
        assert variant_display == "-"
        assert vram_display == "-"


class TestLoadVariant:
    """load_variant() 测试。"""

    def test_success(self, ui_module):
        resp = _make_mock_response(json_data={
            "state": "ready",
            "variant": "voxcpm2",
            "vram_used": "2048MB",
            "load_time": 3.5,
        })
        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.return_value = resp

            result = ui_module.load_variant("http://127.0.0.1:7003", "voxcpm2 (48kHz)")

        assert len(result) == 4
        status_text, variant_display, vram_display, log = result
        assert "state=ready" in status_text
        assert variant_display == "voxcpm2"
        assert "3.50s" in log

    def test_empty_variant(self, ui_module):
        result = ui_module.load_variant("http://127.0.0.1:7003", "")
        assert "请先选择" in result[0]

    def test_failure(self, ui_module):
        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.side_effect = RuntimeError("load failed")

            result = ui_module.load_variant("http://127.0.0.1:7003", "voxcpm2")

        assert "加载失败" in result[0]
        assert "加载失败" in result[3]


class TestUnloadVariant:
    """unload_variant() 测试。"""

    def test_success(self, ui_module):
        resp = _make_mock_response(json_data={"vram_freed": "2048MB"})
        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.return_value = resp

            result = ui_module.unload_variant("http://127.0.0.1:7003")

        assert len(result) == 4
        assert result[0] == "state=idle"
        assert result[3] == "卸载完成"

    def test_failure(self, ui_module):
        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.side_effect = RuntimeError("unload failed")

            result = ui_module.unload_variant("http://127.0.0.1:7003")

        assert "卸载失败" in result[0]


class TestGenerateTts:
    """generate_tts() 流式生成器测试。"""

    def test_tts_mode_success(self, ui_module):
        # mock /status 返回 ready + loaded
        status_resp = _make_mock_response(json_data={
            "state": "ready",
            "loaded_variant": "voxcpm2",
        })
        # mock SSE 流
        sse_lines = [
            'data: {"stage": "tts.start", "progress": 0.0}',
            'data: {"stage": "tts.model_ready", "progress": 0.05}',
            'data: {"stage": "tts.text_processing", "progress": 0.1}',
            'data: {"stage": "tts.generating", "progress": 0.5}',
            'data: {"stage": "tts.saving", "progress": 0.9}',
            'data: {"stage": "completed", "progress": 1.0, "result": {"audio_url": "/output/audio/test.wav", "audio_path": "/abs/path/data/output/audio/test.wav", "duration": 1.5}, "timing": {"total_seconds": 2.0}}',
        ]
        stream_resp = _make_mock_stream_response(sse_lines)

        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            # 第一次 GET /status，第二次 POST /infer (stream)
            client.get.return_value = status_resp
            client.stream.return_value.__enter__ = MagicMock(return_value=stream_resp)
            client.stream.return_value.__exit__ = MagicMock(return_value=False)

            gen = ui_module.generate_tts(
                "http://127.0.0.1:7003",
                tab="tts",
                variant_label="voxcpm2",
                tts_text="你好",
                design_desc="",
                design_text="",
                clone_ref=None,
                clone_text="",
                ultimate_prompt_aud=None,
                ultimate_prompt_text="",
                ultimate_text="",
                cfg=2.0,
                timesteps=10,
                seed=-1,
                fmt="wav",
            )

            frames = list(gen)

        # 应有多帧，最后一帧含完成信息
        assert len(frames) > 0
        last_frame = frames[-1]
        # last_frame = (progress, audio, download, log)
        # v0.2.5+ gr.Audio 收到本地路径 audio_path，download_link 收到 URL
        assert last_frame[0] == "完成"
        # gr.Audio 收到本地绝对路径（绕过 SSRF 防护）
        assert last_frame[1] == "/abs/path/data/output/audio/test.wav"
        # download_link 收到完整 URL
        assert "http://127.0.0.1:7003/output/audio/test.wav" in last_frame[2]

    def test_empty_text(self, ui_module):
        gen = ui_module.generate_tts(
            "http://127.0.0.1:7003",
            tab="tts",
            variant_label="voxcpm2",
            tts_text="",
            design_desc="",
            design_text="",
            clone_ref=None,
            clone_text="",
            ultimate_prompt_aud=None,
            ultimate_prompt_text="",
            ultimate_text="",
            cfg=2.0,
            timesteps=10,
            seed=-1,
            fmt="wav",
        )
        frames = list(gen)
        assert "请输入合成文本" in frames[0][0]

    def test_no_variant_selected(self, ui_module):
        gen = ui_module.generate_tts(
            "http://127.0.0.1:7003",
            tab="tts",
            variant_label="",
            tts_text="你好",
            design_desc="",
            design_text="",
            clone_ref=None,
            clone_text="",
            ultimate_prompt_aud=None,
            ultimate_prompt_text="",
            ultimate_text="",
            cfg=2.0,
            timesteps=10,
            seed=-1,
            fmt="wav",
        )
        frames = list(gen)
        assert "请先选择模型" in frames[0][0]

    def test_clone_mode_without_reference(self, ui_module):
        gen = ui_module.generate_tts(
            "http://127.0.0.1:7003",
            tab="clone",
            variant_label="voxcpm2",
            tts_text="",
            design_desc="",
            design_text="",
            clone_ref=None,
            clone_text="你好",
            ultimate_prompt_aud=None,
            ultimate_prompt_text="",
            ultimate_text="",
            cfg=2.0,
            timesteps=10,
            seed=-1,
            fmt="wav",
        )
        frames = list(gen)
        assert "请上传参考音频" in frames[0][0]

    def test_ultimate_clone_without_prompt_audio(self, ui_module):
        gen = ui_module.generate_tts(
            "http://127.0.0.1:7003",
            tab="ultimate_clone",
            variant_label="voxcpm2",
            tts_text="",
            design_desc="",
            design_text="",
            clone_ref=None,
            clone_text="",
            ultimate_prompt_aud=None,
            ultimate_prompt_text="转写文本",
            ultimate_text="你好",
            cfg=2.0,
            timesteps=10,
            seed=-1,
            fmt="wav",
        )
        frames = list(gen)
        assert "请上传提示音频" in frames[0][0]

    def test_loaded_variant_path_normalized(self, ui_module):
        """v0.2.5+ loaded_variant 为路径时,UI 归一化后不触发切换。

        场景:后端返回 loaded_variant="models/voice/voxcpm/voxcpm2"(路径),
        UI 归一化取最后一段 "voxcpm2",与用户选择的 variant 匹配,
        不触发 unload + load,直接推理。
        """
        # mock /status 返回 ready + loaded_variant 为路径
        status_resp = _make_mock_response(json_data={
            "state": "ready",
            "loaded_variant": "models/voice/voxcpm/voxcpm2",
        })
        # mock SSE 流
        sse_lines = [
            'data: {"stage": "completed", "progress": 1.0, "result": {"audio_url": "/output/audio/test.wav", "audio_path": "/abs/path/test.wav", "duration": 1.5}, "timing": {"total_seconds": 2.0}}',
        ]
        stream_resp = _make_mock_stream_response(sse_lines)

        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.get.return_value = status_resp
            client.stream.return_value.__enter__ = MagicMock(return_value=stream_resp)
            client.stream.return_value.__exit__ = MagicMock(return_value=False)

            gen = ui_module.generate_tts(
                "http://127.0.0.1:7003",
                tab="tts",
                variant_label="voxcpm2",
                tts_text="你好",
                design_desc="",
                design_text="",
                clone_ref=None,
                clone_text="",
                ultimate_prompt_aud=None,
                ultimate_prompt_text="",
                ultimate_text="",
                cfg=2.0,
                timesteps=10,
                seed=-1,
                fmt="wav",
            )

            frames = list(gen)

        # 应该直接推理,不触发 unload + load
        # 验证 client.post 没有被调用(没有 /load 或 /unload)
        client.post.assert_not_called()
        # 第一帧应该是"模型已加载"提示
        assert "模型已加载" in frames[0][0] or "开始推理" in frames[0][0]

    def test_variant_match_no_reload(self, ui_module):
        """v0.2.5+ variant 匹配时不触发 unload + load,直接推理。"""
        status_resp = _make_mock_response(json_data={
            "state": "ready",
            "loaded_variant": "voxcpm2",
        })
        sse_lines = [
            'data: {"stage": "completed", "progress": 1.0, "result": {"audio_url": "/output/audio/test.wav", "audio_path": "/abs/path/test.wav", "duration": 1.5}, "timing": {"total_seconds": 2.0}}',
        ]
        stream_resp = _make_mock_stream_response(sse_lines)

        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.get.return_value = status_resp
            client.stream.return_value.__enter__ = MagicMock(return_value=stream_resp)
            client.stream.return_value.__exit__ = MagicMock(return_value=False)

            gen = ui_module.generate_tts(
                "http://127.0.0.1:7003",
                tab="tts",
                variant_label="voxcpm2",
                tts_text="你好",
                design_desc="",
                design_text="",
                clone_ref=None,
                clone_text="",
                ultimate_prompt_aud=None,
                ultimate_prompt_text="",
                ultimate_text="",
                cfg=2.0,
                timesteps=10,
                seed=-1,
                fmt="wav",
            )

            frames = list(gen)

        # 不触发 unload + load
        client.post.assert_not_called()
        assert "模型已加载" in frames[0][0] or "开始推理" in frames[0][0]


class TestCreateUi:
    """create_ui() 测试（需真实 gradio）。"""

    def test_create_ui_returns_blocks(self):
        """create_ui() 返回 gr.Blocks 实例。

        需要真实 gradio 安装，否则跳过。
        """
        gradio = pytest.importorskip("gradio")
        from engines.voxcpm_tts_c4a8f2e7.ui import create_ui

        demo = create_ui("http://127.0.0.1:7003")
        assert isinstance(demo, gradio.Blocks)
