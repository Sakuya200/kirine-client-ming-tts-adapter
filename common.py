"""Ming-3.0-tts 适配器公共工具：路径解析、内置音色名单与 subprocess 调用上游 CLI。

上游项目经 ``download.py`` 克隆到 ``src-model/base-models/ming_tts_05b/repo``，
权重下载到 ``src-model/base-models/ming_tts_05b/Ming-3.0-tts-0.5b``（目录名与
``configs/model-config.json`` 的 ``requiredModelNameList`` 一致）。本模块的所有脚本由
``begin_llm_task.ps1`` 用适配器共享的 conda_env/venv 解释器运行；subprocess 调用
上游脚本时使用 ``sys.executable`` 保证同一解释器，并以克隆仓库根目录为 ``cwd``，
使 ``ming_tts`` 包（上游 ``ming_tts.py``）经 ``sys.path[0]`` 自动可导入。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 适配器目录：src-model/ming_tts/
_ADAPTER_DIR = Path(__file__).resolve().parent
# src-model/
_SRC_MODEL_ROOT = _ADAPTER_DIR.parent

BASE_MODEL_NAME = "ming_tts"
MODEL_ARTIFACTS_DIR = "base-models"
# 落盘目录名：必须与 model-config.json 的 requiredModelNameList 一致（卸载按同名回收）。
ARTIFACTS_DIR_NAME = "ming_tts_05b"
REPO_SUBDIR = "repo"
WEIGHTS_SUBDIR = "Ming-3.0-tts-0.5b"

# 内置音色（上游 controls.ip 接受的逐字 id，来源：cookbooks/instruction_en.md）。
BUILTIN_VOICES: frozenset[str] = frozenset({"灵小柔", "灵楷", "UK_George", "US_Emma"})


def artifacts_root() -> Path:
    """落盘产物根目录（``base-models/ming_tts_05b``）。"""
    return _SRC_MODEL_ROOT / MODEL_ARTIFACTS_DIR / ARTIFACTS_DIR_NAME


def _require_existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} 不存在: {path}\n"
            f"请先在模型管理页安装 Ming-3.0-tts 模型。"
        )
    return path


def repo_dir() -> Path:
    """上游 Ming-3.0-tts 克隆仓库根目录（含 ``infer.py`` / ``ming_tts.py``）。"""
    return _require_existing(
        artifacts_root() / REPO_SUBDIR, "Ming-3.0-tts 上游代码仓库"
    )


def weights_dir() -> Path:
    """HF 权重目录（``Ming-3.0-tts-0.5b``，作为 ``--model_path``）。"""
    return _require_existing(
        artifacts_root() / WEIGHTS_SUBDIR, "Ming-3.0-tts-0.5b 权重"
    )


def normalize_device(device: str | None) -> str:
    """归一化设备参数为上游 ``infer.py --device`` 接受的值（``cuda:0`` / ``cpu``）。"""
    if not device:
        return "cuda:0"
    normalized = device.strip().lower()
    if normalized.startswith("cuda"):
        return "cuda:0"
    if normalized == "cpu":
        return "cpu"
    return normalized


def _run_upstream_script(script_name: str, args: list[str]) -> None:
    """以当前解释器运行上游脚本，流式转发输出。"""
    script_path = repo_dir() / script_name
    _require_existing(script_path, f"上游脚本 {script_name}")

    cmd = [sys.executable, "-u", str(script_path), *args]
    print(f"[ming_tts] run {' '.join(cmd)}", flush=True)
    print(f"[ming_tts] cwd={repo_dir()}", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(repo_dir()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    finally:
        process.stdout.close()
        process.wait()

    if process.returncode != 0:
        raise SystemExit(
            f"❌ {script_name} 退出码 {process.returncode}，任务失败。"
        )


def run_infer(args: list[str]) -> None:
    """调用上游 ``infer.py``（TTS / FREEFORM / VOICE_DESIGN / 多语言等任务）。"""
    _run_upstream_script("infer.py", args)
