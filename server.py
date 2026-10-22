"""VoxCPM TTS 引擎 HTTP 服务（FastAPI + 原生 WebUI 单端口）。

实现 8 个标准 HTTP 端点 + 挂载原生 WebUI + 挂载输出目录。

启动：
    uv run --no-sync --project envs/voice-voxcpm \\
        python -m engines.voxcpm_tts_c4a8f2e7.server --port 7003

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体）
    POST /load           加载 VoxCPM 模型权重
    POST /unload         卸载模型（释放显存）
    POST /infer          TTS 推理（SSE 流式进度）
    GET  /status         状态查询
    POST /lora/load      v0.1 返回 501
    POST /lora/unload    v0.1 返回 501
    GET  /output/*       静态音频文件（推理产物）
    POST /upload         参考音频上传（WebUI 克隆/极致克隆模式，返回引擎侧路径）
    GET  /               307 重定向到 /ui/
    GET  /ui/*           原生 WebUI 静态文件（webui/ 三件套，FLUX 风格）

WebUI 说明（引擎 WebUI 风格统一）：
    - Gradio UI（ui.py）已废弃，改为原生 HTML/CSS/JS 三件套（webui/）
    - 风格基底与 FLUX 引擎一致（ENGINE_WEBUI_FLUX_STYLE_UNIFY 分析）
    - 四模式（tts/design/clone/ultimate_clone）参考音频经 ``POST /upload``
      落盘后以 ``audios[role=speaker|prompt].path`` 传入 ``/infer``
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import VoxCPMEngine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
# FastAPI 启动时初始化，所有端点共享
_engine: Optional[VoxCPMEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> VoxCPMEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。

    启动时初始化引擎，关闭时清理资源（卸载模型）。
    """
    global _engine, _config

    # 解析配置（如果尚未从命令行注入）
    if _config is None:
        _config = parse_args()

    # 配置日志
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== VoxCPM TTS 引擎启动 =====")
    logger.info("配置: %s", _config.to_dict())

    # 初始化引擎
    _engine = VoxCPMEngine(_config)
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
    )

    yield

    # 关闭清理：卸载模型
    logger.info("===== VoxCPM TTS 引擎关闭 =====")
    if _engine is not None and _engine.wrapper.is_loaded:
        try:
            await _engine._unload_impl()
        except Exception as e:
            logger.warning("关闭时卸载模型失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="VoxCPM TTS Engine",
        description="AIReliable 平台的 VoxCPM 语音合成引擎（多语言 TTS / 声音设计 / 克隆）",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用，iframe 嵌入需要）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 =====
    _register_endpoints(app)

    # ===== 挂载输出目录（/output/* 提供音频文件下载）=====
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

        # ===== 挂载原生 WebUI（webui/ 三件套，FLUX 风格）=====
        _mount_webui(app)

    return app


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（可用模型变体 / 支持的合成模式）。"""
        return (await get_engine().capabilities()).model_dump()

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """加载 VoxCPM 模型权重。"""
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """卸载模型（释放显存）。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """TTS 推理（SSE 流式返回进度 + 最终音频 URL）。

        请求体参考 InferRequest schema：
            - variant: 模型变体名（如 voxcpm2）
            - inputs: TTSInputs 字段（text / mode / cfg_value / ...）
            - audios: 参考音频列表（Clone/Ultimate Clone 模式）
            - task_id: 可选任务 ID
        """
        engine = get_engine()

        # BaseEngine.infer 返回异步迭代器（SSE 帧）
        sse_stream = engine.infer(request)

        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v0.1 返回 501）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v0.1 返回 501）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 错误处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+ T6.2：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)

    # ===== WebUI 专属端点 =====

    # 根路径 307 重定向到 /ui/（与 FLUX 引擎范式一致）
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    # 参考音频上传（WebUI 可控克隆/极致克隆模式）
    # 前端 FormData 上传 → 落盘 output/uploads/ → 返回引擎侧绝对路径
    # 供 /infer 的 audios[role=speaker|prompt].path 引用
    @app.post("/upload")
    async def upload(file: UploadFile) -> dict:
        config = get_config()
        upload_dir = config.output_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload.bin").name  # 防路径穿越
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"  # uuid 前缀防重名
        size = 0
        try:
            with open(dest, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 1MB 分块
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
        finally:
            await file.close()
        logger.info("参考音频上传: %s (%d bytes) -> %s", safe_name, size, dest)
        return {"path": str(dest), "filename": safe_name, "size": size}


def _mount_webui(app: FastAPI) -> None:
    """挂载原生 WebUI 静态文件到 /ui（webui/ 三件套，FLUX 风格）。"""
    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").exists():
        app.mount(
            "/ui",
            StaticFiles(directory=str(webui_dir), html=True),
            name="webui",
        )
        logger.info("原生 WebUI 挂载: /ui -> %s", webui_dir)
    else:
        logger.warning("WebUI 未挂载（webui/index.html 不存在）: %s", webui_dir)


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    # 解析命令行参数
    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("启动 VoxCPM TTS 引擎: host=%s, port=%d", _config.host, _config.port)

    # 创建 app
    app = create_app(_config)

    # 用 uvicorn 启动
    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
