from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LLM_CONFIG = SCRIPT_DIR / "llm_config.yaml"


DEFAULTS: Dict[str, Any] = {
    "provider": None,
    "model": None,
    "temperature": 0.0,
    "extra": {},
    "output": None,
    "prompt_output": None,
    "report_output": None,
    "use_existing_rules": False,
    "max_samples": 30,
    "max_prompt_chars": 60000,
    "max_trace_chars": 2000,
    "max_dynamic_fields": 80,
}


def _strip_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif quote_char == char:
                in_quote = False
        elif char == "#" and not in_quote:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "" or text.lower() in {"null", "none", "~"}:
        return None
    if text.lower() in {"true", "yes", "on"}:
        return True
    if text.lower() in {"false", "no", "off"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_simple_yaml(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    data: Dict[str, Any] = {}
    current_map: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8-sig").splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            if current_map is None:
                raise ValueError(f"Nested YAML value without a parent key: {raw_line}")
            if ":" not in line:
                raise ValueError(f"Unsupported YAML line: {raw_line}")
            key, value = line.strip().split(":", 1)
            parent = data.setdefault(current_map, {})
            if not isinstance(parent, dict):
                raise ValueError(f"YAML key is not a mapping: {current_map}")
            parent[key.strip()] = _parse_scalar(value)
            continue
        current_map = None
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        if value.strip() == "":
            data[key] = {}
            current_map = key
        else:
            data[key] = _parse_scalar(value)
    return data


def load_llm_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_LLM_CONFIG
    config = dict(DEFAULTS)
    config.update(load_simple_yaml(path))
    if not isinstance(config.get("extra"), dict):
        raise ValueError("llm_config.yaml field `extra` must be a mapping.")
    return config


def apply_llm_config(args: Any) -> Any:
    config = load_llm_config(getattr(args, "llm_config", None))
    args.llm_provider = config.get("provider")
    args.llm_model = config.get("model")
    args.llm_temperature = float(config.get("temperature", 0.0) or 0.0)
    args.llm_extra = dict(config.get("extra") or {})
    args.llm_output = config.get("output")
    args.llm_prompt_output = config.get("prompt_output")
    args.llm_report_output = config.get("report_output")
    args.llm_use_existing_rules = bool(config.get("use_existing_rules", False))
    args.llm_max_samples = int(config.get("max_samples", 30) or 30)
    args.llm_max_prompt_chars = int(config.get("max_prompt_chars", 60000) or 60000)
    args.llm_max_trace_chars = int(config.get("max_trace_chars", 2000) or 2000)
    args.llm_max_dynamic_fields = int(config.get("max_dynamic_fields", 80) or 80)
    return args

