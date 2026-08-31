"""Configuration discovery, safe merging, secret resolution, and validation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from tinyCode.config.models import AppConfig, ProviderConfig, TracingConfig
from tinyCode.config.constants import (
    DEFAULT_HARD_MAX_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOTES_ENABLED,
    DEFAULT_ROUND_EXTENSION,
    DEFAULT_ROUND_LIMIT_ACTION,
    DEFAULT_SECURITY_LEVEL,
    DEFAULT_UI_MODE,
    MAX_ALLOWED_ROUNDS,
    SUPPORTED_ROUND_LIMIT_ACTIONS,
    SUPPORTED_SECURITY_LEVELS,
    SUPPORTED_UI_MODES,
)


DEFAULT_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "deepseek": "https://api.deepseek.com",
}
DEFAULT_API_KEY_ENVS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
SUPPORTED_PROTOCOLS = set(DEFAULT_BASE_URLS)
_ENV_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """A user-correctable configuration error; never exits the process."""


def _resolve_config_path() -> Path | None:
    """Return the highest-priority configuration path for compatibility."""
    explicit = os.environ.get("TINYCODE_CONFIG")
    if explicit:
        return Path(explicit)
    project = Path.cwd() / ".tinyCode.yaml"
    if project.exists():
        return project
    global_path = Path.home() / ".tinyCode" / "config.yaml"
    return global_path if global_path.exists() else None


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"配置文件解析失败 ({path}): {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是对象（mapping）: {path}")
    return raw


def _discover_raw_config() -> dict[str, Any]:
    """Load config with safe project/global merge semantics.

    An explicit TINYCODE_CONFIG is standalone. Otherwise a project config may
    override ordinary settings, while its provider list replaces (never merges
    credentials with) the global provider list. This prevents a project-owned
    base_url from silently receiving a globally stored API key.
    """
    explicit = os.environ.get("TINYCODE_CONFIG")
    if explicit:
        return _load_mapping(Path(explicit))

    global_path = Path.home() / ".tinyCode" / "config.yaml"
    project_path = Path.cwd() / ".tinyCode.yaml"
    global_raw = _load_mapping(global_path) if global_path.exists() else {}
    project_raw = _load_mapping(project_path) if project_path.exists() else {}
    if not global_raw and not project_raw:
        raise ConfigError(
            "配置文件缺失: 请在当前目录创建 .tinyCode.yaml，"
            "或在 ~/.tinyCode/config.yaml 放置全局配置"
        )

    merged = dict(global_raw)
    merged.update(project_raw)
    # Notes may contain user-wide preferences and corrections. Enabling or
    # disabling that persistent memory is a user-level decision, so a
    # repository-owned config must not override it.
    if "notes_enabled" in global_raw:
        merged["notes_enabled"] = global_raw["notes_enabled"]
    else:
        merged.pop("notes_enabled", None)
    # Trace payload capture is a user-level privacy decision. A repository may
    # not silently enable it through project-owned configuration.
    if "tracing" in global_raw:
        merged["tracing"] = global_raw["tracing"]
    else:
        merged.pop("tracing", None)
    # A repository-owned config may tighten a user-level security baseline,
    # but must not silently weaken it. Users can still make an explicit
    # process-local override with ``--mode``.
    global_security = global_raw.get("security_level")
    project_security = project_raw.get("security_level")
    if isinstance(global_security, str) and isinstance(project_security, str):
        global_level = global_security.strip().lower()
        project_level = project_security.strip().lower()
        if (
            global_level in SUPPORTED_SECURITY_LEVELS
            and project_level in SUPPORTED_SECURITY_LEVELS
        ):
            strictness = {"permissive": 0, "normal": 1, "strict": 2}
            merged["security_level"] = max(
                (global_level, project_level),
                key=strictness.__getitem__,
            )
    if "providers" in project_raw:
        merged["providers"] = project_raw["providers"]
        if "active_provider" not in project_raw:
            merged.pop("active_provider", None)
    return merged


def _secret_from_keyring(reference: str, provider_name: str) -> str:
    if not isinstance(reference, str) or "/" not in reference:
        raise ConfigError(
            f"Provider '{provider_name}' 的 api_key_keyring 必须是 service/username"
        )
    service, username = reference.split("/", 1)
    if not service or not username:
        raise ConfigError(
            f"Provider '{provider_name}' 的 api_key_keyring 必须是 service/username"
        )
    try:
        import keyring  # type: ignore
    except ImportError as exc:
        raise ConfigError(
            f"Provider '{provider_name}' 配置了 api_key_keyring，但未安装 keyring"
        ) from exc
    try:
        value = keyring.get_password(service, username)
    except Exception as exc:
        raise ConfigError(
            f"Provider '{provider_name}' 无法读取系统 Keyring: {exc}"
        ) from exc
    if not value:
        raise ConfigError(
            f"Provider '{provider_name}' 的 Keyring 条目不存在: {reference}"
        )
    return value


def _resolve_api_key(
    provider: dict[str, Any],
    name: str,
    protocol: str,
    *,
    require_value: bool = True,
) -> str | None:
    literal = provider.get("api_key")
    env_name = provider.get("api_key_env")
    keyring_reference = provider.get("api_key_keyring")
    configured = sum(value is not None for value in (literal, env_name, keyring_reference))
    if configured > 1:
        raise ConfigError(
            f"Provider '{name}' 的 api_key、api_key_env、api_key_keyring 只能配置一个"
        )

    if literal is not None:
        if not isinstance(literal, str) or not literal:
            raise ConfigError(f"Provider '{name}' 的 api_key 必须是非空字符串")
        match = _ENV_REFERENCE_RE.fullmatch(literal)
        if not match:
            return literal if require_value else None
        env_name = match.group(1)

    if env_name is not None:
        if not isinstance(env_name, str) or not _ENV_NAME_RE.fullmatch(env_name):
            raise ConfigError(f"Provider '{name}' 的 api_key_env 不是有效环境变量名")
        if not require_value:
            return None
        value = os.environ.get(env_name)
        if not value:
            raise ConfigError(f"Provider '{name}' 所需环境变量 {env_name} 未设置")
        return value

    if keyring_reference is not None:
        if not isinstance(keyring_reference, str) or "/" not in keyring_reference:
            raise ConfigError(
                f"Provider '{name}' 的 api_key_keyring 必须是 service/username"
            )
        service, username = keyring_reference.split("/", 1)
        if not service or not username:
            raise ConfigError(
                f"Provider '{name}' 的 api_key_keyring 必须是 service/username"
            )
        if not require_value:
            return None
        return _secret_from_keyring(keyring_reference, name)

    if not require_value:
        return None
    default_env = DEFAULT_API_KEY_ENVS[protocol]
    value = os.environ.get(default_env)
    if value:
        return value
    raise ConfigError(
        f"Provider '{name}' 缺少密钥；请配置 api_key、api_key_env、"
        f"api_key_keyring，或设置 {default_env}"
    )


def _validate_provider(
    provider: object,
    index: int,
    *,
    resolve_secret: bool = True,
) -> ProviderConfig:
    if not isinstance(provider, dict):
        raise ConfigError(f"Provider #{index} 必须是对象（mapping）")

    name = provider.get("name", f"provider-{index}")
    protocol = provider.get("protocol")
    model = provider.get("model")
    context_window = provider.get("context_window")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"Provider #{index} 的 name 必须是非空字符串")
    if not isinstance(protocol, str) or not protocol:
        raise ConfigError(f"Provider '{name}' 缺少字段: protocol")
    protocol = protocol.strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ConfigError(f"不支持的协议: {protocol}")
    if not isinstance(model, str) or not model:
        raise ConfigError(f"Provider '{name}' 缺少字段: model")

    base_url = provider.get("base_url") or DEFAULT_BASE_URLS[protocol]
    if not isinstance(base_url, str) or not base_url:
        raise ConfigError(f"Provider '{name}' 的 base_url 必须是非空字符串")
    if context_window is not None and (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or not 1_024 <= context_window <= 10_000_000
    ):
        raise ConfigError(
            f"Provider '{name}' 的 context_window 必须是 1024 到 10000000 之间的整数"
        )

    return ProviderConfig(
        name=name.strip(), protocol=protocol, model=model,
        base_url=base_url,
        api_key=_resolve_api_key(
            provider, name, protocol, require_value=resolve_secret,
        ),
        context_window=context_window,
    )


def load_config() -> AppConfig:
    """Discover, merge, parse, and validate the YAML configuration."""
    raw = _discover_raw_config()
    if "providers" not in raw:
        raise ConfigError("配置文件缺少 'providers' 字段")
    if not isinstance(raw["providers"], list) or not raw["providers"]:
        raise ConfigError("配置文件 'providers' 必须是非空列表")

    # Validate every provider's public schema without resolving external
    # secrets. Only the active provider needs a usable credential for this
    # process; a multi-provider config must not require every vendor's key.
    providers = [
        _validate_provider(item, i, resolve_secret=False)
        for i, item in enumerate(raw["providers"])
    ]
    provider_names = [provider.name for provider in providers]
    if len(provider_names) != len(set(provider_names)):
        duplicate = next(name for name in provider_names if provider_names.count(name) > 1)
        raise ConfigError(f"Provider 名称重复: {duplicate}")

    active_provider = raw.get("active_provider", provider_names[0])
    if not isinstance(active_provider, str) or active_provider not in provider_names:
        raise ConfigError(
            f"active_provider '{active_provider}' 不在 providers 列表中"
            f"（可用: {', '.join(provider_names)}）"
        )

    active_index = provider_names.index(active_provider)
    providers[active_index] = _validate_provider(
        raw["providers"][active_index], active_index, resolve_secret=True,
    )

    hard_max_rounds = raw.get("hard_max_rounds", DEFAULT_HARD_MAX_ROUNDS)
    if (
        isinstance(hard_max_rounds, bool) or not isinstance(hard_max_rounds, int)
        or not 1 <= hard_max_rounds <= MAX_ALLOWED_ROUNDS
    ):
        raise ConfigError(
            f"hard_max_rounds 必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"
        )

    max_rounds = raw.get("max_rounds", DEFAULT_MAX_ROUNDS)
    if (
        isinstance(max_rounds, bool) or not isinstance(max_rounds, int)
        or not 1 <= max_rounds <= hard_max_rounds
    ):
        raise ConfigError(
            f"max_rounds 必须是 1 到 hard_max_rounds（{hard_max_rounds}）之间的整数"
        )

    round_extension = raw.get("round_extension", DEFAULT_ROUND_EXTENSION)
    if (
        isinstance(round_extension, bool) or not isinstance(round_extension, int)
        or not 1 <= round_extension <= MAX_ALLOWED_ROUNDS
    ):
        raise ConfigError(
            f"round_extension 必须是 1 到 {MAX_ALLOWED_ROUNDS} 之间的整数"
        )

    round_limit_action = raw.get(
        "round_limit_action", DEFAULT_ROUND_LIMIT_ACTION,
    )
    if not isinstance(round_limit_action, str):
        raise ConfigError("round_limit_action 必须是 ask、auto 或 stop")
    round_limit_action = round_limit_action.strip().lower()
    if round_limit_action not in SUPPORTED_ROUND_LIMIT_ACTIONS:
        raise ConfigError("round_limit_action 必须是 ask、auto 或 stop")

    security_level = raw.get("security_level", DEFAULT_SECURITY_LEVEL)
    if not isinstance(security_level, str):
        raise ConfigError("security_level 必须是 strict、normal 或 permissive")
    security_level = security_level.strip().lower()
    if security_level not in SUPPORTED_SECURITY_LEVELS:
        raise ConfigError("security_level 必须是 strict、normal 或 permissive")

    ui_mode = raw.get("ui_mode", DEFAULT_UI_MODE)
    if not isinstance(ui_mode, str):
        raise ConfigError("ui_mode 必须是 stream 或 fullscreen")
    ui_mode = ui_mode.strip().lower()
    if ui_mode not in SUPPORTED_UI_MODES:
        raise ConfigError("ui_mode 必须是 stream 或 fullscreen")

    notes_enabled = raw.get("notes_enabled", DEFAULT_NOTES_ENABLED)
    if not isinstance(notes_enabled, bool):
        raise ConfigError("notes_enabled 必须是 true 或 false")

    tracing_raw = raw.get("tracing", {})
    if not isinstance(tracing_raw, dict):
        raise ConfigError("tracing 必须是对象（mapping）")
    tracing_enabled = tracing_raw.get("enabled", True)
    capture_payloads = tracing_raw.get("capture_payloads", False)
    retention_days = tracing_raw.get("retention_days", 14)
    max_trace_files = tracing_raw.get("max_files", 100)
    if not isinstance(tracing_enabled, bool):
        raise ConfigError("tracing.enabled 必须是 true 或 false")
    if not isinstance(capture_payloads, bool):
        raise ConfigError("tracing.capture_payloads 必须是 true 或 false")
    if (
        isinstance(retention_days, bool)
        or not isinstance(retention_days, int)
        or not 1 <= retention_days <= 3650
    ):
        raise ConfigError("tracing.retention_days 必须是 1 到 3650 之间的整数")
    if (
        isinstance(max_trace_files, bool)
        or not isinstance(max_trace_files, int)
        or not 1 <= max_trace_files <= 10_000
    ):
        raise ConfigError("tracing.max_files 必须是 1 到 10000 之间的整数")

    return AppConfig(
        providers=providers, active_provider=active_provider,
        max_rounds=max_rounds,
        round_extension=round_extension,
        hard_max_rounds=hard_max_rounds,
        round_limit_action=round_limit_action,
        security_level=security_level,
        ui_mode=ui_mode,
        notes_enabled=notes_enabled,
        tracing=TracingConfig(
            enabled=tracing_enabled,
            capture_payloads=capture_payloads,
            retention_days=retention_days,
            max_files=max_trace_files,
        ),
    )


def load_provider_config(provider_name: str) -> ProviderConfig:
    """Load one named provider and resolve only that provider's credential.

    Evaluation runs intentionally use two independent providers.  Resolving
    every configured credential would make an otherwise valid multi-provider
    setup fail merely because an unused vendor key is absent.
    """
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ConfigError("Provider 名称必须是非空字符串")
    wanted = provider_name.strip()
    raw = _discover_raw_config()
    providers_raw = raw.get("providers")
    if not isinstance(providers_raw, list) or not providers_raw:
        raise ConfigError("配置文件 'providers' 必须是非空列表")

    seen: set[str] = set()
    for index, item in enumerate(providers_raw):
        public = _validate_provider(item, index, resolve_secret=False)
        if public.name in seen:
            raise ConfigError(f"Provider 名称重复: {public.name}")
        seen.add(public.name)
        if public.name == wanted:
            return _validate_provider(item, index, resolve_secret=True)
    raise ConfigError(
        f"Provider '{wanted}' 不在 providers 列表中（可用: {', '.join(sorted(seen))}）"
    )
