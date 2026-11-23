"""Ming-3.0-tts 专用参数映射：把 ``ParamsEntity`` 转成入口脚本可消费的结构。

key 映射契约：``model_param_xxx`` 读取的 key 与 ``configs/params-config.json`` 的
``name`` 逐字一致（``cfg`` / ``temperature`` / ``seed`` / ``textNormalization`` /
``trimSilence`` / ``synthesisMode`` / ``emotion`` / ``speed`` / ``pitch`` /
``volume`` / ``dialect`` / ``languageId``）。

路径解析约定：权重路径经 ``common.weights_dir()`` 在入口脚本拼装 CLI 时解析，
本模块不做路径存在性检查（仅校验参考音频文件存在——真实管线必传）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common import normalize_device
from params_entity import ParamsEntity

_CLONE_MODE_TASKS = {
    "clone": "TTS",
    "emotion": "EMOTION",
    "attributes": "BASIC",
    "dialect": "DIALECT",
    "multilingual": "MULTILINGUAL",
}


def _float_param(params: ParamsEntity, key: str, default: float) -> float:
    raw = params.model_param_str(key, str(default))
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _optional_int_param(params: ParamsEntity, key: str) -> int | None:
    raw = params.model_param(key, None)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(float(str(raw)))
    except (TypeError, ValueError):
        return None


def _bool_param(params: ParamsEntity, key: str, default: bool) -> bool:
    return params.model_param_bool(key, default)


def _str_param(params: ParamsEntity, key: str, default: str = "") -> str:
    raw = params.model_param(key, None)
    if raw is None:
        return default
    return str(raw)


@dataclass(frozen=True)
class MingGenerationParams:
    cfg: float
    temperature: float
    seed: int | None
    text_normalization: bool
    trim_silence: bool


def _load_generation(params: ParamsEntity) -> MingGenerationParams:
    return MingGenerationParams(
        cfg=_float_param(params, "cfg", 2.0),
        temperature=_float_param(params, "temperature", 0.0),
        seed=_optional_int_param(params, "seed"),
        text_normalization=_bool_param(params, "textNormalization", False),
        trim_silence=_bool_param(params, "trimSilence", True),
    )


def _language_id(language: str | None) -> str:
    """任务级语言（AppLanguage）→ 上游 ``language_id``；仅中英有意义。"""
    if language is not None and language.strip().lower() == "english":
        return "English"
    return "Chinese"


@dataclass(frozen=True)
class MingTTSParams:
    text: str
    language_id: str
    speaker: str
    instruction: str
    generation: MingGenerationParams
    output_path: str
    device: str


def load_tts_params(path: str | Path) -> MingTTSParams:
    params = ParamsEntity.from_file(path)
    args = params.tts_args()

    text = (args.text or "").strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    # speaker / instruction 双空合法：互斥判定由入口脚本的 resolve_voice_mode 负责。
    return MingTTSParams(
        text=text,
        language_id=_language_id(args.language),
        speaker=(args.speaker or "").strip(),
        instruction=(params.model_param_str("instruction", "") or "").strip(),
        generation=_load_generation(params),
        output_path=args.output_path,
        device=normalize_device(params.runtime.device),
    )


@dataclass(frozen=True)
class MingVoiceCloneParams:
    mode: str
    task: str
    text: str
    ref_audio_path: str
    ref_text: str
    language_id: str | None
    emotion: str | None
    speed: str | None
    pitch: str | None
    volume: str | None
    dialect: str | None
    generation: MingGenerationParams
    output_path: str
    device: str


def load_voice_clone_params(path: str | Path) -> MingVoiceCloneParams:
    params = ParamsEntity.from_file(path)
    args = params.voice_clone_args()

    text = (args.text or "").strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    mode = (_str_param(params, "synthesisMode", "clone") or "clone").strip()
    if mode not in _CLONE_MODE_TASKS:
        raise ValueError(
            f"Unsupported synthesisMode: {mode} (expected one of {sorted(_CLONE_MODE_TASKS)})"
        )
    task = _CLONE_MODE_TASKS[mode]

    ref_audio_path = (args.ref_audio_path or "").strip()
    if not ref_audio_path:
        raise ValueError("Voice cloning requires a reference audio path.")
    resolved_ref = Path(ref_audio_path).expanduser().resolve()
    if not resolved_ref.exists():
        raise FileNotFoundError(f"Reference audio file not found: {resolved_ref}")

    if mode == "multilingual":
        language_id: str | None = (_str_param(params, "languageId", "") or "").strip() or None
        if language_id is None:
            raise ValueError("MULTILINGUAL synthesis requires a languageId.")
    elif mode == "clone":
        language_id = _language_id(args.language)
    else:
        language_id = None

    def _mode_value(key: str) -> str | None:
        raw = (_str_param(params, key, "") or "").strip()
        return raw or None

    return MingVoiceCloneParams(
        mode=mode,
        task=task,
        text=text,
        ref_audio_path=str(resolved_ref),
        ref_text=(args.ref_text or "").strip(),
        language_id=language_id,
        emotion=_mode_value("emotion") if mode == "emotion" else None,
        speed=_mode_value("speed") if mode == "attributes" else None,
        pitch=_mode_value("pitch") if mode == "attributes" else None,
        volume=_mode_value("volume") if mode == "attributes" else None,
        dialect=_mode_value("dialect") if mode == "dialect" else None,
        generation=_load_generation(params),
        output_path=args.output_path,
        device=normalize_device(params.runtime.device),
    )


@dataclass(frozen=True)
class MingVoiceDesignParams:
    text: str
    style: str
    generation: MingGenerationParams
    output_path: str
    device: str


def load_voice_design_params(path: str | Path) -> MingVoiceDesignParams:
    params = ParamsEntity.from_file(path)
    args = params.voice_design_args()

    text = (args.text or "").strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    style = (args.instruct or "").strip()
    if not style:
        raise ValueError("Voice design requires a style description.")

    return MingVoiceDesignParams(
        text=text,
        style=style,
        generation=_load_generation(params),
        output_path=args.output_path,
        device=normalize_device(params.runtime.device),
    )
