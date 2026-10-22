"""VoxCPM 模型下载脚本。

使用 huggingface_hub.snapshot_download() 下载 VoxCPM 模型（diffusers 目录格式）
到 models/voice/tts/voxcpm/<variant>/ 目录。

用法：
    # 通过 uv 环境运行（确保 huggingface_hub 已安装）
    uv run --no-sync --project envs/voice-voxcpm \\
        python -m engines.voxcpm_tts_c4a8f2e7.scripts.download_voxcpm --variant voxcpm2

    # 使用 HuggingFace 镜像（国内推荐）
    $env:HF_ENDPOINT="https://hf-mirror.com"
    uv run --no-sync --project envs/voice-voxcpm \\
        python -m engines.voxcpm_tts_c4a8f2e7.scripts.download_voxcpm --variant voxcpm2

支持的变体：
    - voxcpm2      OpenBMB/VoxCPM2，~8GB，48kHz，多语言
    - voxcpm-0.5b  OpenBMB/VoxCPM-0.5B，~1GB，16kHz，中英
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 项目根目录（engines/voxcpm_tts_c4a8f2e7/scripts/ 往上三级）
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 已知变体定义（与 model_scanner.py _KNOWN_VARIANTS 保持一致）
VARIANTS: dict[str, dict] = {
    "voxcpm2": {
        "repo_id": "OpenBMB/VoxCPM2",
        "subdir": "voxcpm2",
        "description": "VoxCPM2 主力模型（48kHz，多语言，~8GB）",
    },
    "voxcpm-0.5b": {
        "repo_id": "OpenBMB/VoxCPM-0.5B",
        "subdir": "voxcpm-0.5b",
        "description": "VoxCPM-0.5B 轻量模型（16kHz，中英，~1GB）",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 VoxCPM 模型到规范目录 models/voice/tts/voxcpm/<variant>/"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS.keys()),
        help="要下载的模型变体名",
    )
    parser.add_argument(
        "--models-root",
        default=str(PROJECT_ROOT / "models"),
        help="模型根目录（默认: <project>/models）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使目标目录已存在也重新下载（覆盖）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    variant_info = VARIANTS[args.variant]
    repo_id = variant_info["repo_id"]
    subdir = variant_info["subdir"]

    # 目标目录：models/voice/tts/voxcpm/<variant>/
    target_dir = Path(args.models_root) / "voice" / "tts" / "voxcpm" / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"===== VoxCPM 模型下载 =====")
    print(f"变体: {args.variant} ({variant_info['description']})")
    print(f"HuggingFace Repo: {repo_id}")
    print(f"目标目录: {target_dir}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(未设置，使用默认 huggingface.co)')}")
    print()

    # 检查是否已下载
    index_file = target_dir / "model_index.json"
    if index_file.exists() and not args.force:
        print(f"[SKIP] 目标目录已存在 model_index.json，模型似乎已下载。")
        print(f"       如需重新下载，加 --force 参数。")
        return 0

    # 导入 huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"[ERROR] huggingface_hub 未安装: {e}")
        print(f"        请运行: uv run --no-sync --project envs/voice-voxcpm pip install huggingface_hub")
        return 2

    # 开始下载
    print(f"[INFO] 开始 snapshot_download: {repo_id} -> {target_dir}")
    print(f"       （VoxCPM2 ~8GB，可能需要 10-30 分钟，视网络情况）")
    print(f"       （支持断点续传，中断后重跑会跳过已下载文件）")
    print()

    start_time = time.time()

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            repo_type="model",
            # 不指定 allow_patterns，下载整个 repo
            # diffusers 目录格式需要所有子目录（loc_encoder / tslm / loc_dit / audio_vae_v2）
            resume_download=True,  # 断点续传
            max_workers=4,         # 并发下载线程
        )
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print(f"        如果是网络问题，可以尝试:")
        print(f"        1. 设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像")
        print(f"        2. 配置代理后重试")
        print(f"        3. 已下载的文件会保留，重跑可断点续传")
        return 3

    elapsed = time.time() - start_time

    print()
    print(f"===== 下载完成 =====")
    print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
    print(f"下载到: {downloaded_path}")
    print()

    # 验证关键文件（VoxCPM 为单文件格式，非 diffusers 目录）
    print(f"===== 验证文件 =====")
    # 两个变体共有的标志文件
    key_files = ["config.json", "audiovae.pth", "tokenizer.json", "tokenizer_config.json"]
    # 主权重文件因变体而异
    if args.variant == "voxcpm2":
        key_files.append("model.safetensors")
    elif args.variant == "voxcpm-0.5b":
        key_files.append("pytorch_model.bin")
    # VoxCPM 为单文件格式，无子目录需要验证
    key_subdirs: list[str] = []

    all_ok = True
    for fname in key_files:
        fpath = target_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 1024 * 1024:
                print(f"  [OK] {fname} ({size/1024/1024:.1f} MB)")
            else:
                print(f"  [OK] {fname} ({size} bytes)")
        else:
            print(f"  [MISS] {fname}")
            all_ok = False

    for subdir_name in key_subdirs:
        sub_path = target_dir / subdir_name
        if sub_path.exists() and any(sub_path.iterdir()):
            file_count = sum(1 for _ in sub_path.rglob("*") if _.is_file())
            total_size = sum(f.stat().st_size for f in sub_path.rglob("*") if f.is_file())
            print(f"  [OK] {subdir_name}/ ({file_count} files, {total_size/1024/1024:.1f} MB)")
        else:
            print(f"  [MISS] {subdir_name}/")
            all_ok = False

    # 计算总大小
    total_size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
    print()
    print(f"总大小: {total_size/1024/1024/1024:.2f} GB")

    if all_ok:
        print()
        print(f"[SUCCESS] VoxCPM 模型 {args.variant} 下载完成！")
        print(f"          现在可以启动 VoxCPM-TTS 引擎，UI 下拉框将显示该变体。")
        return 0
    else:
        print()
        print(f"[WARNING] 部分文件缺失，可能下载不完整。请重跑此脚本（支持断点续传）。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
