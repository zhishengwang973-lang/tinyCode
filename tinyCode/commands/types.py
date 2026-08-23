"""Command types — metadata, parameter hints, UI control interface."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum


class CommandType(Enum):
    LOCAL = "local"              # 纯本地执行，不影响 UI 或 AI
    UI = "ui"                    # 影响 UI 状态（清屏、切换模式等）
    PROMPT_INJECT = "inject"     # 将预设提示词注入对话流，让 AI 处理


@dataclass
class ParamHint:
    name: str
    description: str = ""
    required: bool = False


@dataclass
class CommandMeta:
    name: str
    description: str
    usage: str
    cmd_type: CommandType
    aliases: list[str] = field(default_factory=list)
    params: list[ParamHint] = field(default_factory=list)
    hidden: bool = False
    handler: Callable[[list[str]], Awaitable[str]] | None = None


class UIControl(ABC):
    """Commands interact with the UI through this interface, avoiding direct
    coupling to the TUI rendering framework."""

    @abstractmethod
    def show_system_message(self, text: str) -> None:
        """Display a system-level message in the conversation area."""
        ...

    @abstractmethod
    def send_to_conversation(self, text: str) -> None:
        """Inject text into the conversation as a user message (triggers AI)."""
        ...

    @abstractmethod
    def toggle_plan_mode(self) -> bool:
        """Toggle plan-only mode; return new state."""
        ...

    @abstractmethod
    def set_security_level(self, level_name: str) -> str:
        """Set security level; return current level label."""
        ...

    @abstractmethod
    def get_token_count(self) -> int:
        """Return estimated token count of current history."""
        ...

    @abstractmethod
    def clear_conversation(self) -> None:
        """Clear the conversation history and display."""
        ...

    @abstractmethod
    async def trigger_compress(self) -> str:
        """Manually trigger compression; return result message."""
        ...

    @abstractmethod
    def get_session_list(self) -> list[dict]:
        """Return list of available sessions from store."""
        ...

    @abstractmethod
    def load_session(self, session_id: str) -> str:
        """Load a saved session to replace current history."""
        ...

    def new_session(self) -> str:
        """Start a new empty session."""
        raise NotImplementedError

    def delete_session(self, session_id: str) -> str:
        """Delete a saved session from disk."""
        raise NotImplementedError

    @abstractmethod
    def get_plan_only(self) -> bool:
        ...

    @abstractmethod
    def get_security_level(self) -> str:
        ...

    def get_runtime_status(self) -> dict:
        """Return foreground turn status when the UI has a turn runtime."""
        return {"state": "unknown", "turn_id": 0, "last_outcome": "none"}

    def get_max_rounds(self) -> int:
        """Return the current process-level maximum for subsequent turns."""
        raise NotImplementedError

    def set_max_rounds(self, value: int) -> int:
        """Set the process-level maximum for subsequent turns."""
        raise NotImplementedError

    def get_round_extension(self) -> int:
        raise NotImplementedError

    def set_round_extension(self, value: int) -> int:
        raise NotImplementedError

    def get_hard_max_rounds(self) -> int:
        raise NotImplementedError

    def set_hard_max_rounds(self, value: int) -> int:
        raise NotImplementedError

    def get_round_limit_action(self) -> str:
        raise NotImplementedError

    def set_round_limit_action(self, value: str) -> str:
        raise NotImplementedError

    def request_exit(self) -> None:
        """Ask the UI input loop to finish through its normal cleanup path."""
        raise NotImplementedError

    def get_system_prompt(self, section: str = "all") -> str:
        """Return a read-only snapshot of the effective model prompt context."""
        raise NotImplementedError

    async def confirm_action(self, prompt: str) -> bool:
        """Request explicit confirmation for a local high-impact command."""
        return False

    def workspace_changed(self) -> None:
        """Refresh components that bind state to the current directory."""
        return None

    def update_progress(self, text: str) -> None:
        """Update transient progress without appending conversation output."""
        return None
