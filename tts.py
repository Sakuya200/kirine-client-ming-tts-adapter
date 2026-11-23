"""内置音色 TTS / 自由指令 FREEFORM 入口：转译 --params-file 后调用上游 infer.py。

语音条件（互斥）：
- 选择了预置说话人（内置音色）→ 上游 ``IP`` 任务，``controls.ip = <speaker>``；
- 未选说话人时填写自然语言指令 → 上游 ``FREEFORM`` 任务；
- 两者皆空 → 任务失败（上游必须有语音条件）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import BUILTIN_VOICES, run_infer, weights_dir
from params import MingGenerationParams, MingTTSParams, load_tts_params


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ming-3.0-tts text-to-speech inference.")
    parser.add_argument(
        "--params-file",
        dest="params_file",
        type=str,
        required=True,
        help="Path to a JSON params file produced by the kirine-client UI.",
    )
    return parser.parse_args(argv)


def resolve_voice_mode(params: MingTTSParams) -> tuple[str, list[str]]:
    """按语音条件解析上游 task 与任务段 CLI 片段。"""
    speaker = (params.speaker or "").strip()
    instruction = (params.instruction or "").strip()
    if speaker:
        if speaker not in BUILTIN_VOICES:
            raise ValueError(
                f"内置音色不存在: {speaker}。可用音色: {', '.join(sorted(BUILTIN_VOICES))}"
            )
        return "IP", ["--ip", speaker, "--use_zero_spk_emb"]
    if instruction:
        return "FREEFORM", ["--instruction", instruction, "--use_zero_spk_emb"]
    raise ValueError("需要选择内置音色说话人，或填写自然语言指令（FREEFORM）。")


def build_infer_args(params: MingTTSParams, model_path: Path) -> list[str]:
    task, task_segment = resolve_voice_mode(params)
    args = [
        "--model_path", str(model_path),
        "--task", task,
        "--text", params.text,
        "--output", params.output_path,
        "--device", params.device,
        *task_segment,
    ]
    if params.language_id is not None:
        args += ["--language_id", params.language_id]
    return args + _generation_segment(params.generation)


def _generation_segment(gen: MingGenerationParams) -> list[str]:
    # build_generation_args 在 voice_clone.py（三入口共用）；延迟导入避免环。
    from voice_clone import build_generation_args

    return build_generation_args(gen)


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    params = load_tts_params(cli_args.params_file)
    print(
        f"[ming_tts] TTS text_len={len(params.text)} speaker={params.speaker!r} "
        f"language_id={params.language_id} device={params.device}",
        flush=True,
    )
    run_infer(build_infer_args(params, weights_dir()))


if __name__ == "__main__":
    main()
