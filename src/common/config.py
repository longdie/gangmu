from __future__ import annotations

import os
import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import ROOT_DIR


CONFIG_PATH = ROOT_DIR / "config.toml"
CONFIG_EXAMPLE_PATH = ROOT_DIR / "config.example.toml"
_MISSING = object()


class ConfigError(ValueError):
    """Raised when Gangmu configuration is missing or invalid."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str = field(repr=False)
    base_url: str
    protocol: str
    trust_env: bool


@dataclass(frozen=True)
class ModelConfig:
    task: str
    name: str
    provider: str
    dimensions: int | None = None


@dataclass(frozen=True)
class OCRConfig:
    method: str
    parallel_workers: int
    render_dpi: int
    jpeg_quality: int
    page_retry_attempts: int
    retry_delay_seconds: float
    fast_max_tokens: int
    premium_max_tokens: int
    timeout_seconds: float
    toc_pages: tuple[int, ...]
    premium_retry_delay_seconds: float
    premium_call_delay_seconds: float


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a table: {path}")
    return data


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class GangmuConfig:
    def __init__(self, data: dict[str, Any], source_path: Path | None):
        self._data = data
        self.source_path = source_path
        version = self.get_int("config_version")
        if version != 1:
            raise ConfigError(f"Unsupported config_version: {version}")

    def get(self, dotted_path: str, default: Any = _MISSING) -> Any:
        value: Any = self._data
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                if default is not _MISSING:
                    return default
                raise ConfigError(f"Missing configuration value: {dotted_path}")
            value = value[part]
        return value

    def get_str(self, dotted_path: str, default: Any = _MISSING) -> str:
        value = self.get(dotted_path, default)
        if not isinstance(value, str):
            raise ConfigError(f"{dotted_path} must be a string.")
        return value.strip()

    def get_int(
        self,
        dotted_path: str,
        default: Any = _MISSING,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        value = self.get(dotted_path, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{dotted_path} must be an integer.")
        if minimum is not None and value < minimum:
            raise ConfigError(f"{dotted_path} must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ConfigError(f"{dotted_path} must be at most {maximum}.")
        return value

    def get_float(
        self,
        dotted_path: str,
        default: Any = _MISSING,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        value = self.get(dotted_path, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{dotted_path} must be a number.")
        result = float(value)
        if minimum is not None and result < minimum:
            raise ConfigError(f"{dotted_path} must be at least {minimum}.")
        if maximum is not None and result > maximum:
            raise ConfigError(f"{dotted_path} must be at most {maximum}.")
        return result

    def get_bool(self, dotted_path: str, default: Any = _MISSING) -> bool:
        value = self.get(dotted_path, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{dotted_path} must be a boolean.")
        return value

    def provider(self, name: str, *, require_api_key: bool = False) -> ProviderConfig:
        section_path = f"providers.{name}"
        section = self.get(section_path)
        if not isinstance(section, dict):
            raise ConfigError(f"{section_path} must be a table.")

        model_names = section.get("models", {})
        if not isinstance(model_names, dict):
            raise ConfigError(f"{section_path}.models must be a table.")
        for alias, model_name in model_names.items():
            if not isinstance(alias, str) or not isinstance(model_name, str):
                raise ConfigError(
                    f"{section_path}.models must map string aliases to model names."
                )
            if not alias.strip() or not model_name.strip():
                raise ConfigError(
                    f"{section_path}.models cannot contain empty aliases or names."
                )

        api_key_env = self.get_str(f"{section_path}.api_key_env", "")
        base_url_env = self.get_str(f"{section_path}.base_url_env", "")
        protocol = self.get_str(
            f"{section_path}.protocol",
            "openai_chat_completions",
        ).lower()
        trust_env = self.get_bool(f"{section_path}.trust_env", True)
        if protocol not in {"openai_chat_completions", "claude_messages", "openai_responses"}:
            raise ConfigError(
                f"{section_path}.protocol must be 'openai_chat_completions' "
                "or 'claude_messages' or 'openai_responses'."
            )
        api_key = (
            os.environ.get(api_key_env, "").strip() if api_key_env else ""
        ) or self.get_str(f"{section_path}.api_key", "")
        base_url = (
            os.environ.get(base_url_env, "").strip() if base_url_env else ""
        ) or self.get_str(f"{section_path}.base_url", "")

        if require_api_key and not api_key:
            hint = f" or environment variable {api_key_env}" if api_key_env else ""
            raise ConfigError(
                f"Missing API key for provider '{name}'. "
                f"Set {section_path}.api_key in config.toml{hint}."
            )
        if require_api_key and not base_url:
            raise ConfigError(
                f"Missing API base URL for connection '{name}'. "
                f"Set {section_path}.base_url or its configured environment variable."
            )
        return ProviderConfig(
            name=name,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            trust_env=trust_env,
        )

    def provider_names(self) -> tuple[str, ...]:
        providers = self.get("providers")
        if not isinstance(providers, dict) or not providers:
            raise ConfigError("providers must be a non-empty table.")
        names = tuple(sorted(providers))
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ConfigError("Provider names must be non-empty strings.")
        return names

    def provider_model_name(self, provider: str, alias: str) -> str:
        self.provider(provider)
        model_names = self.get(f"providers.{provider}.models", {})
        if not isinstance(model_names, dict):
            raise ConfigError(f"providers.{provider}.models must be a table.")
        model_name = model_names.get(alias)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ConfigError(
                f"Provider '{provider}' does not define model alias '{alias}'. "
                f"Set providers.{provider}.models.{alias} in config.toml."
            )
        return model_name.strip()

    def log_dir(self, task: str | None = None) -> Path:
        configured_value = self.get_str("logging.directory")
        if not configured_value:
            raise ConfigError("logging.directory cannot be empty.")
        configured_path = Path(configured_value).expanduser()
        base_dir = (
            configured_path
            if configured_path.is_absolute()
            else ROOT_DIR / configured_path
        ).resolve()
        if task is None:
            return base_dir
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", task):
            raise ConfigError(
                "Log task names must use lowercase letters, numbers, underscores, "
                "or hyphens."
            )
        return base_dir / task

    def model(self, task: str) -> ModelConfig:
        default_section = self.get("models.default")
        task_section = self.get(f"models.{task}")
        if not isinstance(default_section, dict) or not isinstance(task_section, dict):
            raise ConfigError(f"models.default and models.{task} must be tables.")

        default_provider = self.get_str("models.default.provider")
        provider = self.get_str(f"models.{task}.provider", "") or default_provider
        if not provider:
            raise ConfigError(f"No provider configured for task '{task}'.")
        self.provider(provider)

        task_name = self.get_str(f"models.{task}.name", "")
        if task_name:
            name = task_name
        else:
            default_alias = self.get_str("models.default.alias", "")
            if default_alias:
                name = self.provider_model_name(provider, default_alias)
            else:
                name = self.get_str("models.default.name", "")
        if not name:
            raise ConfigError(f"No model configured for task '{task}'.")

        dimensions_value = task_section.get("dimensions")
        dimensions = None
        if dimensions_value is not None:
            dimensions = self.get_int(f"models.{task}.dimensions", minimum=1)
        return ModelConfig(
            task=task,
            name=name,
            provider=provider,
            dimensions=dimensions,
        )

    def ocr(self) -> OCRConfig:
        method = self.get_str("ocr.method").lower()
        if method not in {"fast", "premium"}:
            raise ConfigError("ocr.method must be 'fast' or 'premium'.")

        toc_pages_raw = self.get("ocr.toc_pages")
        if not isinstance(toc_pages_raw, list) or any(
            isinstance(page, bool) or not isinstance(page, int) or page < 1
            for page in toc_pages_raw
        ):
            raise ConfigError("ocr.toc_pages must be an array of positive integers.")

        return OCRConfig(
            method=method,
            parallel_workers=self.get_int("ocr.parallel_workers", minimum=1),
            render_dpi=self.get_int("ocr.render_dpi", minimum=1),
            jpeg_quality=self.get_int(
                "ocr.jpeg_quality",
                minimum=1,
                maximum=100,
            ),
            page_retry_attempts=self.get_int("ocr.page_retry_attempts", minimum=1),
            retry_delay_seconds=self.get_float("ocr.retry_delay_seconds", minimum=0),
            fast_max_tokens=self.get_int("ocr.fast_max_tokens", minimum=1),
            premium_max_tokens=self.get_int("ocr.premium_max_tokens", minimum=1),
            timeout_seconds=self.get_float("ocr.timeout_seconds", minimum=0),
            toc_pages=tuple(toc_pages_raw),
            premium_retry_delay_seconds=self.get_float(
                "ocr.premium_retry_delay_seconds",
                minimum=0,
            ),
            premium_call_delay_seconds=self.get_float(
                "ocr.premium_call_delay_seconds",
                minimum=0,
            ),
        )

    def validate(self, *, require_api_keys: bool = False) -> None:
        self.log_dir()
        for role in self.provider_names():
            self.provider(role, require_api_key=require_api_keys)
        for task in self.get("models"):
            if task != "default":
                self.model(task)
        self.ocr()
        self.get_int("embedding.parallel_workers", minimum=1)
        self.get_int("requests.call_attempts", minimum=1)
        self.get_int("requests.max_connections", minimum=1)
        if self.get_int("requests.max_keepalive_connections", minimum=0) > self.get_int("requests.max_connections"):
            raise ConfigError("Keepalive connections cannot exceed max_connections.")
        if self.model("embedding").dimensions != 2048:
            raise ConfigError("This release requires 2048-dimensional embeddings.")


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT_DIR / value).resolve()


def load_config(config_path: str | Path | None = None) -> GangmuConfig:
    defaults = _read_toml(CONFIG_EXAMPLE_PATH)
    requested = config_path or os.environ.get("GANGMU_CONFIG")
    path = resolve_project_path(requested) if requested else CONFIG_PATH
    if requested and not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    data = _deep_merge(defaults, _read_toml(path)) if path.exists() else defaults
    config = GangmuConfig(data, path if path.exists() else None)
    config.validate()
    return config


_active_config: GangmuConfig | None = None


def set_config(config: GangmuConfig) -> None:
    global _active_config
    _active_config = config


def get_config() -> GangmuConfig:
    global _active_config
    if _active_config is None:
        _active_config = load_config()
    return _active_config


def reload_config() -> GangmuConfig:
    set_config(load_config())
    return get_config()


def get_log_dir(task: str) -> Path:
    return get_config().log_dir(task)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate configuration without displaying secrets.")
    parser.add_argument("--config")
    parser.add_argument("--require-api-keys", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config.validate(require_api_keys=args.require_api_keys)
    print("Configuration is valid.")
    for role in config.provider_names():
        print(f"{role}: credentials {'configured' if config.provider(role).api_key else 'missing'}")


if __name__ == "__main__":
    main()
