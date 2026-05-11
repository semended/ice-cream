from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    key: str
    role: str
    canonical_model: str
    provider: str
    provider_model: str
    enabled_by_default: bool
    heavy: bool
    temperature: float
    max_output_tokens: int
    image_max_side: int
    response_format: str
    openrouter_model: str | None = None
    deepinfra_model: str | None = None
    local_vllm_model: str | None = None

    def with_provider(self, provider: str | None) -> "ModelConfig":
        if not provider:
            return self
        data = asdict(self)
        data["provider"] = provider
        provider_model = data.get(f"{provider}_model")
        if provider_model:
            data["provider_model"] = provider_model
        return ModelConfig(**data)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_models(path: Path) -> dict[str, ModelConfig]:
    models: dict[str, dict[str, Any]] = {}
    current_key: str | None = None
    in_models = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line == "models:":
            in_models = True
            continue
        if not in_models:
            continue
        if indent == 2 and line.endswith(":"):
            current_key = line[:-1]
            models[current_key] = {}
            continue
        if indent == 4 and current_key and ":" in line:
            name, value = line.split(":", 1)
            models[current_key][name.strip()] = _parse_scalar(value)

    result: dict[str, ModelConfig] = {}
    for key, data in models.items():
        result[key] = ModelConfig(
            key=key,
            role=str(data["role"]),
            canonical_model=str(data["canonical_model"]),
            provider=str(data["provider"]),
            provider_model=str(data["provider_model"]),
            enabled_by_default=bool(data["enabled_by_default"]),
            heavy=bool(data["heavy"]),
            temperature=float(data["temperature"]),
            max_output_tokens=int(data["max_output_tokens"]),
            image_max_side=int(data["image_max_side"]),
            response_format=str(data["response_format"]),
            openrouter_model=data.get("openrouter_model"),
            deepinfra_model=data.get("deepinfra_model"),
            local_vllm_model=data.get("local_vllm_model"),
        )
    return result


def select_models(
    all_models: dict[str, ModelConfig],
    requested: str | None,
    include_heavy: bool,
    provider_override: str | None,
) -> list[ModelConfig]:
    if requested:
        keys = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        keys = [
            key
            for key, config in all_models.items()
            if config.enabled_by_default and (include_heavy or not config.heavy)
        ]
    unknown = [key for key in keys if key not in all_models]
    if unknown:
        raise ValueError(f"Unknown model keys: {', '.join(unknown)}")
    return [all_models[key].with_provider(provider_override) for key in keys]


def dump_config_snapshot(path: Path, args: dict[str, Any], models: list[ModelConfig]) -> None:
    lines = ["args:"]
    for key in sorted(args):
        lines.append(f"  {key}: {_format_yaml(args[key])}")
    lines.append("models:")
    for model in models:
        lines.append(f"  {model.key}:")
        for key, value in asdict(model).items():
            if key == "key":
                continue
            lines.append(f"    {key}: {_format_yaml(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_yaml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(char in text for char in [":", "#", "{", "}", "[", "]", ","]):
        return '"' + text.replace('"', '\\"') + '"'
    return text
