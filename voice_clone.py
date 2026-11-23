"""声音克隆入口：转译 --params-file 后调用上游 infer.py。

参考音频来自前端本地文件（voice-clone 任务自带参考音频/参考文本顶层字段）。
``synthesisMode`` 表单参数决定上游任务：clone→TTS、emotion→EMOTION、
attributes→BASIC、dialect→DIALECT、multilingual→MULTILINGUAL。
上游任务对参考文本的接受面不同：TTS/MULTILINGUAL/EMOTION 必带 prompt_text；
BASIC/DIALECT 不接受（即使 ref_text 有值也不传）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import run_infer, weights_dir
from params import (
    MingGenerationParams,
    MingVoiceCloneParams,
    load_voice_clone_params,
)

# 上游接受 --prompt_text 的任务（其余任务传了会被上游拒绝）。
PROMPT_TEXT_TASKS = frozenset({"TTS", "MULTILINGUAL", "EMOTION"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ming-3.0-tts voice cloning inference.")
    parser.add_argument(
        "--params-file",
        dest="params_file",
        type=str,
        required=True,
        help="Path to a JSON params file produced by the kirine-client UI.",
    )
    return parser.parse_args(argv)


def build_generation_args(gen: MingGenerationParams) -> list[str]:
    """生成参数段（三入口共用）：cfg/temperature 恒传，seed/正则化/裁剪按值与开关。"""
    args = ["--cfg", str(gen.cfg), "--temperature", str(gen.temperature)]
    if gen.seed is not None:
        args += ["--seed", str(gen.seed)]
    # Windows 无 pynini：文本正则化默认关闭，关闭即传 --no_text_normalization（懒加载跳过）。
    if not gen.text_normalization:
        args.append("--no_text_normalization")
    if not gen.trim_silence:
        args.append("--no_trim_silence")
    return args


def build_clone_args(params: MingVoiceCloneParams, model_path: Path) -> list[str]:
    args = [
        "--model_path", str(model_path),
        "--task", params.task,
        "--text", params.text,
        "--output", params.output_path,
        "--device", params.device,
        "--prompt_wav_path", params.ref_audio_path,
        "--use_spk_emb",
    ]
    if params.language_id is not None:
        args += ["--language_id", params.language_id]
    if params.ref_text and params.task in PROMPT_TEXT_TASKS:
        args += ["--prompt_text", params.ref_text]
    if params.task == "EMOTION" and params.emotion:
        args += ["--emotion", params.emotion]
    elif params.task == "BASIC":
        if params.speed:
            args += ["--speed", params.speed]
        if params.pitch:
            args += ["--pitch", params.pitch]
        if params.volume:
            args += ["--volume", params.volume]
    elif params.task == "DIALECT" and params.dialect:
        args += ["--dialect", params.dialect]
    return args + build_generation_args(params.generation)


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    params = load_voice_clone_params(cli_args.params_file)
    print(
        f"[ming_tts] voice-clone mode={params.mode} task={params.task} "
        f"text_len={len(params.text)} ref_wav={params.ref_audio_path} device={params.device}",
        flush=True,
    )
    run_infer(build_clone_args(params, weights_dir()))


if __name__ == "__main__":
    main()
