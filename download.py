"""Ming-3.0-tts Custom 下载脚本。

流程：
1. 将上游项目 git clone 到 ``<target-root-dir>/ming_tts_05b/repo``（已存在则跳过）。
2. 从 Hugging Face 下载权重到 ``<target-root-dir>/ming_tts_05b/Ming-3.0-tts-0.5b``。
3. 可选安装 pynini（文本正则化）：无可靠 Windows wheel，失败仅告警不阻断
   （默认 ``--no_text_normalization`` 即可正常推理）。

注意：落盘目录名常量 ``ming_tts_05b`` 与 ``configs/model-config.json`` 的
``requiredModelNameList`` 一致（卸载按同名回收），不取 ``--base-model`` 拼接。

依赖安装不在本脚本内完成——由 ``init_task_runtime.ps1`` 通过适配器的
``requirements.txt`` / ``requirements-torch.txt`` 装入共享的 conda_env/venv。
本脚本不做 editable install：``infer.py`` 从克隆仓库根目录运行即可经 ``sys.path[0]``
导入 ``ming_tts`` 包。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/inclusionAI/Ming-3.0-tts"
DEFAULT_REPO_BRANCH = "main"

# 落盘目录名：与 model-config.json 的 requiredModelNameList 逐字一致。
ARTIFACTS_DIR_NAME = "ming_tts_05b"
REPO_SUBDIR = "repo"
WEIGHTS_REPO_ID = "inclusionAI/Ming-3.0-tts-0.5b"
WEIGHTS_SUBDIR = "Ming-3.0-tts-0.5b"


def _emit(message: str, *, stderr: bool = False) -> None:
    print(message, file=sys.stderr if stderr else sys.stdout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", dest="base_model", type=str, required=True)
    parser.add_argument("--model-version", dest="model_version", type=str, required=True)
    parser.add_argument(
        "--target-root-dir", dest="target_root_dir", type=str, required=True
    )
    parser.add_argument("--log-path", dest="log_path", type=str, required=False)
    parser.add_argument(
        "--task-log-file", dest="task_log_file", type=str, required=False
    )
    parser.add_argument(
        "--repo-url", dest="repo_url", type=str, default=DEFAULT_REPO_URL
    )
    parser.add_argument(
        "--repo-branch", dest="repo_branch", type=str, default=DEFAULT_REPO_BRANCH
    )
    return parser.parse_args(argv)


def _clone_repo(repo_url: str, branch: str, destination: Path) -> Path:
    """Clone a git repository with retry logic and timeout handling."""
    git_bin = shutil.which("git")
    if git_bin is None:
        raise SystemExit(
            "Custom download requires git for automatic clone. "
            "Install git and make sure it is available in PATH."
        )

    max_retries = 3
    retry_delay = 5  # seconds
    timeout = 600  # seconds (10 minutes)

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                [
                    git_bin,
                    "clone",
                    "--branch",
                    branch,
                    "--single-branch",
                    repo_url,
                    str(destination),
                ],
                check=True,
                timeout=timeout,
            )
            return destination
        except subprocess.TimeoutExpired:
            error_msg = (
                f"Git clone timed out after {timeout}s (attempt {attempt}/{max_retries})"
            )
            if attempt < max_retries:
                print(f"⚠️  {error_msg}. Retrying in {retry_delay}s...", file=sys.stderr)
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise SystemExit(
                    f"❌ Git clone failed: {error_msg}.\n"
                    f"Please check your network connection and try again.\n"
                    f"Alternatively, you can:\n"
                    f"  1. Use a proxy: git config --global http.proxy <proxy_url>\n"
                    f"  2. Retry later if GitHub is unavailable"
                )
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Git clone failed with exit code {e.returncode} "
                f"(attempt {attempt}/{max_retries})"
            )
            if attempt < max_retries:
                print(f"⚠️  {error_msg}. Retrying in {retry_delay}s...", file=sys.stderr)
                if destination.exists():
                    shutil.rmtree(destination)
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise SystemExit(
                    f"❌ {error_msg}.\n"
                    f"Error details: {e}\n"
                    f"Common causes:\n"
                    f"  - Network connectivity issues (RPC failed, connection reset)\n"
                    f"  - GitHub is temporarily unavailable\n"
                    f"  - Firewall/proxy blocking connection\n"
                    f"\n"
                    f"Solutions:\n"
                    f"  1. Check your network connection\n"
                    f"  2. Try again in a few moments\n"
                    f"  3. Use a proxy if behind firewall: "
                    f"git config --global http.proxy <proxy_url>\n"
                    f"  4. Retry later if GitHub service is unstable"
                )

    raise SystemExit("❌ Git clone exhausted retries.")


def _download_weights(target_dir: Path) -> None:
    weights_dir = target_dir / WEIGHTS_SUBDIR
    weights_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        _emit(f"📥 Downloading {WEIGHTS_REPO_ID} ...")
        snapshot_download(repo_id=WEIGHTS_REPO_ID, local_dir=str(weights_dir))
        _emit(f"✓ {WEIGHTS_REPO_ID} downloaded")
    except Exception as e:  # noqa: BLE001
        _emit(f"⚠️  Model download failed: {e}", stderr=True)
        raise SystemExit(
            f"❌ Failed to download Ming-3.0-tts-0.5b weights.\n"
            f"Please check your network connection and try again.\n"
            f"Alternatively, manually download the HF repo and place it under "
            f"'{weights_dir}':\n"
            f"  - {WEIGHTS_REPO_ID} → {weights_dir}"
        )


def _install_pynini_optional() -> None:
    """可选安装 pynini（文本正则化）。无可靠 Windows wheel，失败仅告警。"""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pynini==2.1.6"],
            check=True,
            timeout=600,
        )
        _emit("✓ pynini installed (text normalization enabled)")
    except Exception as e:  # noqa: BLE001
        _emit(
            f"WARN: pynini unavailable ({e}); text normalization disabled "
            f"(inference runs with --no_text_normalization). "
            f"To enable: conda install -c conda-forge pynini==2.1.6"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_root = Path(args.target_root_dir).expanduser().resolve()
    target_dir = target_root / ARTIFACTS_DIR_NAME
    target_root.mkdir(parents=True, exist_ok=True)

    repo_dir = target_dir / REPO_SUBDIR
    if not repo_dir.exists():
        try:
            _emit(f"📥 Cloning Ming-3.0-tts into target directory: {repo_dir}")
            _clone_repo(args.repo_url, args.repo_branch, repo_dir)
            _emit("✓ Clone completed successfully")
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"❌ Setup failed after direct clone: Ming-3.0-tts runtime is incomplete, "
                f"target directory: {repo_dir}\n\n"
                f"Troubleshooting:\n"
                f"  1. Ensure you have sufficient disk space\n"
                f"  2. Check your network connection\n"
                f"  3. Manually clone from: {args.repo_url}\n"
                f"  4. Retry later if GitHub service is unstable"
            ) from e
    else:
        _emit(f"✓ Ming-3.0-tts checkout already exists at {repo_dir}; skip clone")

    _download_weights(target_dir)
    _install_pynini_optional()
    _emit(f"✅ Ming-3.0-tts is ready at {target_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
