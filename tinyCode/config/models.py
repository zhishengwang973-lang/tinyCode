"""Configuration data models."""

from pydantic import BaseModel

from tinyCode.config.constants import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOTES_ENABLED,
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


class AppConfig(BaseModel):
    """Top-level application configuration."""

    providers: list[ProviderConfig]
    active_provider: str  # name of the provider to use
    max_rounds: int = DEFAULT_MAX_ROUNDS
    security_level: str = DEFAULT_SECURITY_LEVEL
    notes_enabled: bool = DEFAULT_NOTES_ENABLED
