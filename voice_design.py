"""音色设计入口：转译 --params-file 后调用上游 infer.py（上游 VOICE_DESIGN 任务）。

音色描述（style）来自 voice-design 任务顶层的描述字段（Rust 侧
``VoiceDesignArgs.instruct``），无参考音频，``use_zero_spk_emb`` 生成。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import run_infer, weights_dir
from params import MingVoiceDesignParams, load_voice_design_params


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ming-3.0-tts voice design inference.")
    parser.add_argument(
        "--params-file",
        dest="params_file",
        type=str,
        required=True,
        help="Path to a JSON params file produced by the kirine-client UI.",
    )
    return parser.parse_args(argv)


def build_design_args(params: MingVoiceDesignParams, model_path: Path) -> list[str]:
    args = [
        "--model_path", str(model_path),
        "--task", "VOICE_DESIGN",
        "--text", params.text,
        "--output", params.output_path,
        "--device", params.device,
        "--style", params.style,
        "--use_zero_spk_emb",
    ]
    from voice_clone import build_generation_args

    return args + build_generation_args(params.generation)


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    params = load_voice_design_params(cli_args.params_file)
    print(
        f"[ming_tts] voice-design text_len={len(params.text)} "
        f"style_len={len(params.style)} device={params.device}",
        flush=True,
    )
    run_infer(build_design_args(params, weights_dir()))


if __name__ == "__main__":
    main()
