"""Configuration data models."""

from pydantic import BaseModel, Field

from tinyCode.config.constants import (
    DEFAULT_HARD_MAX_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOTES_ENABLED,
    DEFAULT_ROUND_EXTENSION,
    DEFAULT_ROUND_LIMIT_ACTION,
    DEFAULT_SECURITY_LEVEL,
)


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str
    protocol: str  # "anthropic" or "openai"
    model: str
    base_url: str | None = None
    api_key: str | None = None
    context_window: int | None = None


class TracingConfig(BaseModel):
    """Local execution-trace configuration."""

    enabled: bool = True
    capture_payloads: bool = False
    retention_days: int = 14
    max_files: int = 100


class AppConfig(BaseModel):
    """Top-level application configuration."""

    providers: list[ProviderConfig]
    active_provider: str  # name of the provider to use
    max_rounds: int = DEFAULT_MAX_ROUNDS
    round_extension: int = DEFAULT_ROUND_EXTENSION
    hard_max_rounds: int = DEFAULT_HARD_MAX_ROUNDS
    round_limit_action: str = DEFAULT_ROUND_LIMIT_ACTION
    security_level: str = DEFAULT_SECURITY_LEVEL
    notes_enabled: bool = DEFAULT_NOTES_ENABLED
    tracing: TracingConfig = Field(default_factory=TracingConfig)
