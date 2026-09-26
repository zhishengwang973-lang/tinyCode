"""Configuration data models."""

from pydantic import BaseModel, Field

from tinyCode.config.constants import (
    DEFAULT_HARD_MAX_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NOTE_ROUTING_CONFIDENCE,
    DEFAULT_NOTE_ROUTING_ENABLED,
    DEFAULT_NOTE_ROUTING_MODEL,
    DEFAULT_NOTE_ROUTING_TIMEOUT,
    DEFAULT_NOTES_ENABLED,
    DEFAULT_ROUND_EXTENSION,
    DEFAULT_ROUND_LIMIT_ACTION,
    DEFAULT_SECURITY_LEVEL,
    DEFAULT_TASK_MODE_ROUTING_CONFIDENCE,
    DEFAULT_TASK_MODE_ROUTING_ENABLED,
    DEFAULT_TASK_MODE_ROUTING_MODEL,
    DEFAULT_TASK_MODE_ROUTING_LLM_TIMEOUT,
    DEFAULT_TASK_MODE_ROUTING_TIMEOUT,
    DEFAULT_UI_MODE,
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


class TaskModeRoutingConfig(BaseModel):
    """Optional hybrid router for ambiguous direct/inspect/modify turns."""

    enabled: bool = DEFAULT_TASK_MODE_ROUTING_ENABLED
    api_key: str | None = None
    base_url: str = "https://api.typesafe.ai"
    model: str = DEFAULT_TASK_MODE_ROUTING_MODEL
    confidence_threshold: float = DEFAULT_TASK_MODE_ROUTING_CONFIDENCE
    timeout_seconds: float = DEFAULT_TASK_MODE_ROUTING_TIMEOUT
    llm_timeout_seconds: float = DEFAULT_TASK_MODE_ROUTING_LLM_TIMEOUT
    llm_fallback: bool = True


class NoteRoutingConfig(BaseModel):
    """Optional Jev gate for selecting auto-note categories to update."""

    enabled: bool = DEFAULT_NOTE_ROUTING_ENABLED
    api_key: str | None = None
    base_url: str = "https://api.typesafe.ai"
    model: str = DEFAULT_NOTE_ROUTING_MODEL
    confidence_threshold: float = DEFAULT_NOTE_ROUTING_CONFIDENCE
    timeout_seconds: float = DEFAULT_NOTE_ROUTING_TIMEOUT


class TeamAutomationConfig(BaseModel):
    """User-level policy for automatic multi-agent orchestration."""

    mode: str = "auto"
    max_members: int = 3
    isolation: str = "worktree"
    worktree_creation: str = "automatic"
    merge_policy: str = "review"
    require_plan_approval: bool = True
    cleanup_after_apply: bool = True
    timeout_seconds: float = 1800.0
    validation_commands: list[str] = Field(default_factory=list)
    allow_llm_conflict_resolution: bool = False


class AppConfig(BaseModel):
    """Top-level application configuration."""

    providers: list[ProviderConfig]
    active_provider: str  # name of the provider to use
    max_rounds: int = DEFAULT_MAX_ROUNDS
    round_extension: int = DEFAULT_ROUND_EXTENSION
    hard_max_rounds: int = DEFAULT_HARD_MAX_ROUNDS
    round_limit_action: str = DEFAULT_ROUND_LIMIT_ACTION
    security_level: str = DEFAULT_SECURITY_LEVEL
    ui_mode: str = DEFAULT_UI_MODE
    notes_enabled: bool = DEFAULT_NOTES_ENABLED
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    note_routing: NoteRoutingConfig = Field(default_factory=NoteRoutingConfig)
    task_mode_routing: TaskModeRoutingConfig = Field(
        default_factory=TaskModeRoutingConfig,
    )
    team: TeamAutomationConfig = Field(default_factory=TeamAutomationConfig)
