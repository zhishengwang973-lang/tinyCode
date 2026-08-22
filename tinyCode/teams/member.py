"""Team member runner — coroutine backend implementation."""

import asyncio
import os
from pathlib import Path

from tinyCode.agent.loop import AgentLoop
from tinyCode.conversation.history import ConversationHistory
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.providers.base import BaseProvider, create_provider
from tinyCode.security import PathSandbox, SecurityGuard, SecurityLevel, SecurityPolicy
from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.models import MemberDef, MemberStatus, MessageType, TeamMessage
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.context import use_workspace
from tinyCode.tools.registry import ToolRegistry


class TeamMember:
    """Runs a team member in a coroutine (same process, independent history)."""

    def __init__(
        self,
        member_def: MemberDef,
        team_dir: Path,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        max_rounds: int = 10,
        workspace: Path | None = None,
        security_level: SecurityLevel = SecurityLevel.NORMAL,
        preapproved: bool = False,
    ) -> None:
        self.defn = member_def
        self.status = MemberStatus.IDLE
        self._history = ConversationHistory()
        self._mailbox = Mailbox(team_dir, member_def.name)
        self._provider = (
            create_provider(provider.config.model_copy(update={"model": member_def.model}))
            if member_def.model else provider
        )
        self._owns_provider = self._provider is not provider
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._max_rounds = max(1, max_rounds)
        self._workspace = (workspace or Path.cwd()).resolve()
        self._security_level = security_level
        self._preapproved = preapproved
        self._last_msg_id = ""
        self.last_turns = 0
        self.last_model_requests = 0
        self.last_tokens = 0
        self.last_tokens_available = False

    @property
    def workspace(self) -> Path:
        return self._workspace

    async def run(self, task: str) -> str:
        """Execute one task to completion. Returns result text."""
        self.status = MemberStatus.BUSY
        self.last_turns = 0
        self.last_model_requests = 0
        self.last_tokens = 0
        self.last_tokens_available = False
        self._history.add_user_message(task)

        prompt_builder = PromptBuilder()
        prompt_injector = PromptInjector()

        loop = AgentLoop(
            provider=self._provider,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
            prompt_builder=prompt_builder,
            prompt_injector=prompt_injector,
            security_guard=SecurityGuard(
                policy=SecurityPolicy(
                    level=self._security_level,
                    project_root=self._workspace,
                ),
                sandbox=PathSandbox(self._workspace),
                level=self._security_level,
                interactive=False,
                preapproved=self._preapproved and self._security_level != SecurityLevel.STRICT,
            ),
            environment_text=f"cwd: {self._workspace}",
            max_rounds=self._max_rounds,
        )

        result_parts: list[str] = []
        try:
            if not self._workspace.is_dir():
                raise RuntimeError(f"Team 工作目录不存在: {self._workspace}")
            try:
                # Background worktree cleanup uses directory mtime as its
                # conservative activity lease. Reused worktrees can be old;
                # refresh the lease before the first model/tool await so a
                # cleaner cannot remove a live member workspace mid-task.
                os.utime(self._workspace, None)
            except OSError as exc:
                raise RuntimeError(f"无法标记 Team 工作目录为活跃: {exc}") from exc
            with use_workspace(self._workspace):
                async for event in loop.run(self._history):
                    from tinyCode.agent.events import (
                        AgentDoneEvent,
                        ErrorEvent,
                        RoundStartEvent,
                        TextDeltaEvent,
                    )
                    if isinstance(event, TextDeltaEvent):
                        result_parts.append(event.text)
                    elif isinstance(event, ErrorEvent):
                        raise RuntimeError(event.message)
                    elif isinstance(event, AgentDoneEvent):
                        break
                    elif isinstance(event, RoundStartEvent):
                        self.last_turns = max(self.last_turns, event.round_number)

            result = "".join(result_parts)
            if not result.strip():
                raise RuntimeError("Team 成员未返回结果")
            self._notify_lead("done", result)
            return result
        except asyncio.CancelledError:
            self._notify_lead("cancelled")
            raise
        except Exception as exc:
            self._notify_lead("failed", str(exc))
            raise
        finally:
            self.last_model_requests = loop.turn_model_requests
            self.last_tokens = loop.turn_usage.total_tokens
            self.last_tokens_available = loop.turn_usage.available
            self.status = MemberStatus.IDLE

    async def resume(self, new_task: str) -> str:
        """Resume from idle with a new task (keeps context)."""
        self._check_mail()
        return await self.run(new_task)

    async def close(self) -> None:
        if self._owns_provider:
            await self._provider.close()

    def _notify_lead(self, event: str, detail: str = "") -> None:
        msg = TeamMessage(
            from_member=self.defn.name, to_member="lead",
            msg_type=MessageType.LIFECYCLE,
            content=f"[{event}] {detail}",
        )
        self._mailbox.send(msg)

    def _check_mail(self) -> list[TeamMessage]:
        msgs = self._mailbox.read_new(self._last_msg_id)
        if msgs:
            self._last_msg_id = msgs[-1].id
            for m in msgs:
                if m.msg_type == MessageType.TEXT and m.from_member == "lead":
                    self._history.add_context_message(f"[Lead → {self.defn.name}]: {m.content}")
        return msgs
