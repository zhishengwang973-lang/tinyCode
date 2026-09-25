"""Sub-agent runner — executes a sub-agent to completion (run-to-end mode)."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path

from tinyCode.agent.events import AgentDoneEvent, ErrorEvent, RoundStartEvent, TextDeltaEvent
from tinyCode.agent.loop import AgentLoop
from tinyCode.conversation.compression import ContextCompressor
from tinyCode.conversation.history import ConversationHistory
from tinyCode.conversation.summarizer import _find_safe_split
from tinyCode.conversation.truncator import ToolResultTruncator
from tinyCode.providers.base import BaseProvider, create_provider
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.security import PathSandbox, SecurityGuard, SecurityLevel, SecurityPolicy
from tinyCode.subagent.filter import ToolFilter
from tinyCode.subagent.models import SubAgentRole, SubAgentTask, TaskStatus
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.base import ToolCategory
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tracing.recorder import TraceRecorder
from tinyCode.storage.journal import atomic_write_text

_FORK_INSTRUCTION = """\
[Fork 模式] 你是一个子工作器。遵守以下规则：
- 不要再创建子工作器（sub_agent 不可用）
- 不要主动对话、不要请求确认、不要问用户问题
- 直接使用工具完成任务，不需要征求许可
- 完成后输出结构化报告，控制在 500 字以内
- 报告格式：## 结果摘要 / ## 关键发现 / ## 文件与代码 / ## 建议"""

_FORK_MAX_MESSAGES = 24
# The delegated task is appended explicitly, so a concise recent, protocol-
# safe parent suffix is enough for most forks.  Never cut at an unsafe tool
# boundary; correctness wins when no valid suffix exists.
_FORK_MAX_CHARS = 40_000
_MAX_RESULT_CHARS = 16_000
_FORK_TIMEOUT_SECONDS = 300.0
_FORK_INITIAL_ROUNDS = 6
_FORK_HARD_ROUNDS = 16
_FORK_ROUND_EXTENSION = 4
_SUBAGENT_EXTENSION_PROMPT = (
    "[Subagent 轮次预算已扩展] 不要扩大任务范围，不要重复读取已有证据。"
    "优先收敛已有发现；证据足够时立即输出最终报告。"
)


class SubAgentRunner:
    """Run a sub-agent to completion in a single call."""

    def __init__(
        self,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        roles: dict[str, SubAgentRole],
        trace_recorder: TraceRecorder | None = None,
        instructions_text: str = "",
        note_manager=None,
        skill_registry=None,
        truncator: ToolResultTruncator | None = None,
        current_time_text=None,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._roles = roles
        self._trace_recorder = trace_recorder
        self._instructions_text = instructions_text
        self._note_manager = note_manager
        self._skill_registry = skill_registry
        self._truncator = truncator
        self._current_time_text = current_time_text

    def capability_manifest(
        self,
        role_name: str | None,
        *,
        background: bool,
    ) -> dict[str, object]:
        """Describe the effective worker limits shown before parent approval."""
        role = self._roles.get(role_name) if role_name else None
        registered_tools = self._tool_registry.list_tools()
        all_tools = [tool.name for tool in registered_tools]
        read_tools = {
            tool.name for tool in self._tool_registry.list_tools()
            if tool.category == ToolCategory.READ
        }
        effective_background = background or role_name is None
        unknown_role = role_name is not None and role is None
        allowed = (
            []
            if unknown_role
            else ToolFilter(
                role,
                background=effective_background,
                parent_tools=all_tools,
                read_tools=read_tools,
            ).filter(all_tools)
        )
        tools_by_name = {tool.name: tool for tool in registered_tools}
        may_modify_workspace = unknown_role
        if not may_modify_workspace:
            for name in allowed:
                tool = tools_by_name.get(name)
                if tool is None:
                    continue
                try:
                    if tool.may_modify({}):
                        may_modify_workspace = True
                        break
                except Exception:
                    # An extension that cannot prove it is read-only remains
                    # conservatively unavailable to inspect-mode delegation.
                    may_modify_workspace = True
                    break
        return {
            "mode": "fork" if role_name is None else "role",
            "role": role_name or "fork",
            "background": effective_background,
            "permission": role.permission if role is not None else "strict",
            "max_rounds": role.max_rounds if role is not None else _FORK_HARD_ROUNDS,
            "initial_rounds": (
                min(role.initial_rounds, role.max_rounds)
                if role is not None else _FORK_INITIAL_ROUNDS
            ),
            "round_extension": (
                min(role.round_extension, role.max_rounds)
                if role is not None else _FORK_ROUND_EXTENSION
            ),
            "finalization_rounds": (
                min(role.finalization_rounds, role.max_rounds)
                if role is not None else 2
            ),
            "timeout_seconds": (
                role.timeout_seconds if role is not None else _FORK_TIMEOUT_SECONDS
            ),
            "allowed_tools": allowed,
            "may_modify_workspace": may_modify_workspace,
        }

    async def run(self, task: SubAgentTask, parent_history: ConversationHistory) -> str:
        """Execute a sub-agent task to completion.

        Returns the final result text.
        """
        role = self._roles.get(task.role) if task.role else None
        if task.role and role is None:
            error = f"未知 Subagent 角色: {task.role}"
            task.fail(error)
            raise RuntimeError(error)
        is_fork = role is None
        provider = self._provider
        if role is not None and role.model:
            try:
                provider = create_provider(
                    self._provider.config.model_copy(update={"model": role.model})
                )
            except Exception as exc:
                error = f"无法创建角色模型 {role.model}: {exc}"
                task.fail(error)
                raise RuntimeError(error) from exc

        # Filter tools
        all_tool_names = [t.name for t in self._tool_registry.list_tools()]
        read_tool_names = {
            tool.name for tool in self._tool_registry.list_tools()
            if tool.category == ToolCategory.READ
        }
        tool_filter = ToolFilter(role, background=task.background,
                                 parent_tools=all_tool_names,
                                 read_tools=read_tool_names)
        allowed_tools = tool_filter.filter(all_tool_names)

        # Build tool definitions (only allowed tools)
        # We need a filtered registry
        filtered_registry = ToolRegistry()
        for t in self._tool_registry.list_tools():
            if t.name in allowed_tools:
                filtered_registry.register(t)

        # Build sub-history
        sub_history = ConversationHistory()

        if is_fork:
            # Fork while the parent is executing ``sub_agent``: the parent's
            # latest assistant message can legitimately contain this very
            # tool call before its result has been appended.  Passing that
            # incomplete protocol pair to another provider causes OpenAI-style
            # APIs to reject the entire fork request with HTTP 400.  Preserve
            # the useful conversational text, but exclude only unresolved
            # tool-call blocks from the fork snapshot.
            sub_history = self._fork_history(parent_history)
            # Append fork instruction as user message
            sub_history.add_user_message(f"{_FORK_INSTRUCTION}\n\n任务: {task.task}")
        else:
            # Defined role: blank conversation
            assert role is not None
            sub_history.add_user_message(f"{role.system_prompt}\n\n任务: {task.task}")

        # Build sub prompt components
        prompt_builder = PromptBuilder()
        prompt_injector = PromptInjector()

        # Sub agent loop
        workspace = get_workspace_root()
        # Fork workers are independently restricted to the background read
        # whitelist, so NORMAL lets those reads proceed without granting any
        # mutation capability.
        permission = SecurityLevel(role.permission) if role else SecurityLevel.NORMAL
        security_guard = SecurityGuard(
            policy=SecurityPolicy(
                level=permission,
                project_root=workspace,
            ),
            sandbox=PathSandbox(workspace),
            level=permission,
            interactive=False,
            # Starting a foreground role is itself a write-class tool call and
            # therefore passes through the parent HITL gate.  NORMAL roles may
            # carry that approval into their bounded workspace; STRICT roles
            # remain read-only.  Background/fork tasks are STRICT and filtered
            # to read tools independently.
            preapproved=role is not None and permission != SecurityLevel.STRICT,
        )
        hard_rounds = role.max_rounds if role else _FORK_HARD_ROUNDS
        initial_rounds = min(
            role.initial_rounds if role else _FORK_INITIAL_ROUNDS,
            hard_rounds,
        )
        round_extension = min(
            role.round_extension if role else _FORK_ROUND_EXTENSION,
            hard_rounds,
        )
        finalization_rounds = min(
            role.finalization_rounds if role else 2,
            hard_rounds,
        )
        sub_loop = AgentLoop(
            provider=provider,
            tool_registry=filtered_registry,
            tool_executor=self._tool_executor,
            prompt_builder=prompt_builder,
            prompt_injector=prompt_injector,
            security_guard=security_guard,
            truncator=self._truncator,
            note_manager=self._note_manager,
            skill_registry=self._skill_registry,
            instructions_text=self._instructions_text,
            environment_text=f"cwd: {workspace}",
            current_time_text=self._current_time_text,
            max_rounds=initial_rounds,
            round_extension=round_extension,
            hard_max_rounds=hard_rounds,
            round_limit_action="auto",
            auto_extension_prompt=_SUBAGENT_EXTENSION_PROMPT,
            finalization_rounds=finalization_rounds,
            compressor=ContextCompressor(provider.config.model, provider),
            # Background workers use a detached child trace so they remain
            # observable without appending spans after the parent task ends.
            trace_recorder=self._trace_recorder,
        )

        child_trace_handle = None
        if task.background and self._trace_recorder is not None:
            child_trace_handle = self._trace_recorder.begin_task(
                task.task,
                model=provider.config.model,
                detached=True,
                parent_task_id=task.id,
                role=task.role or "fork",
            )

        current_round_text: list[str] = []
        round_count = 0
        done_event: AgentDoneEvent | None = None

        try:
            async def consume() -> None:
                nonlocal round_count, done_event, current_round_text
                async for event in sub_loop.run(sub_history):
                    if isinstance(event, RoundStartEvent):
                        round_count += 1
                        current_round_text = []
                    elif isinstance(event, TextDeltaEvent):
                        current_round_text.append(event.text)
                    elif isinstance(event, ErrorEvent):
                        raise RuntimeError(event.message)
                    elif isinstance(event, AgentDoneEvent):
                        done_event = event
                        break

            timeout = role.timeout_seconds if role else _FORK_TIMEOUT_SECONDS
            try:
                await asyncio.wait_for(consume(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                error = f"子 Agent 执行超过 {timeout:g} 秒"
                task.fail(
                    error,
                    tokens=sub_loop.turn_usage.total_tokens,
                    rounds=round_count,
                )
                raise RuntimeError(error) from exc
            except asyncio.CancelledError:
                task.cancel()
                raise
            except Exception as exc:
                task.fail(
                    str(exc),
                    tokens=sub_loop.turn_usage.total_tokens,
                    rounds=round_count,
                )
                raise

            if done_event is None or not done_event.is_normal:
                reason = done_event.reason if done_event is not None else "missing_done_event"
                error = f"子 Agent 未正常完成: {reason}"
                task.fail(
                    error,
                    tokens=sub_loop.turn_usage.total_tokens,
                    rounds=round_count,
                )
                raise RuntimeError(error)

            final_text = "".join(current_round_text).strip()
            if not final_text:
                error = "子 Agent 未返回结果"
                task.fail(
                    error,
                    tokens=sub_loop.turn_usage.total_tokens,
                    rounds=round_count,
                )
                raise RuntimeError(error)

            final_text, result_path = self._bound_result(
                task.id, final_text, workspace,
            )
            task.result_path = result_path

            task.complete(
                final_text,
                tokens=sub_loop.turn_usage.total_tokens,
                rounds=round_count,
            )
            return final_text
        finally:
            if child_trace_handle is not None and self._trace_recorder is not None:
                trace_status = {
                    TaskStatus.COMPLETED: "no_tool_call",
                    TaskStatus.CANCELLED: "cancelled",
                    TaskStatus.FAILED: "error",
                }.get(task.status, "interrupted")
                self._trace_recorder.finish_task(
                    child_trace_handle,
                    status=trace_status,
                    attributes={
                        "subagent_task_id": task.id,
                        "role": task.role or "fork",
                        "turns": round_count,
                        "model_requests": sub_loop.turn_model_requests,
                        "total_tokens": sub_loop.turn_usage.total_tokens,
                        "error": task.result if task.status == TaskStatus.FAILED else "",
                    },
                )
            if provider is not self._provider:
                await provider.close()

    @staticmethod
    def _bound_result(
        task_id: str, content: str, workspace: Path,
    ) -> tuple[str, str]:
        if len(content) <= _MAX_RESULT_CHARS:
            return content, ""
        path = workspace / ".tinyCode" / "subagent_results" / f"{task_id}.md"
        try:
            atomic_write_text(path, content + ("\n" if not content.endswith("\n") else ""))
            relative = str(path.relative_to(workspace))
            suffix = (
                f"\n\n[Subagent 结果已截断；完整内容保存在 {relative}，"
                "可使用 read_file 分段读取]"
            )
            return content[:_MAX_RESULT_CHARS] + suffix, relative
        except OSError as exc:
            suffix = (
                "\n\n[Subagent 结果已截断且保存失败: "
                f"{type(exc).__name__}: {exc}]"
            )
            return content[:_MAX_RESULT_CHARS] + suffix, ""

    @staticmethod
    def _fork_history(parent_history: ConversationHistory) -> ConversationHistory:
        """Copy parent context without unresolved provider tool-call pairs."""
        messages = parent_history.get_messages()
        pending = SubAgentRunner._unpaired_tool_call_ids(messages)
        fork_history = ConversationHistory()

        for message in messages:
            copied = dict(message)
            if copied.get("role") == "assistant" and pending:
                tool_calls = copied.get("tool_calls")
                if isinstance(tool_calls, list):
                    copied["tool_calls"] = [
                        call for call in tool_calls
                        if not (
                            isinstance(call, dict)
                            and call.get("id") in pending
                        )
                    ]
                    if not copied["tool_calls"]:
                        copied.pop("tool_calls", None)

                content = copied.get("content")
                if isinstance(content, list):
                    copied["content"] = [
                        block for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") == "tool_use"
                            and block.get("id") in pending
                        )
                    ]
            fork_history.add_raw_message(copied)
        compacted = SubAgentRunner._compact_fork_messages(
            fork_history.get_messages(),
        )
        fork_history.replace_messages(compacted)
        return fork_history

    @staticmethod
    def _compact_fork_messages(messages: list[dict]) -> list[dict]:
        """Keep a protocol-safe recent suffix for an unusually long fork."""
        if (
            len(messages) <= _FORK_MAX_MESSAGES
            and SubAgentRunner._fork_message_chars(messages) <= _FORK_MAX_CHARS
        ):
            return messages

        desired = max(0, len(messages) - _FORK_MAX_MESSAGES)
        seen_splits: set[int] = set()
        fallback_suffix: list[dict] | None = None
        for candidate in range(desired, len(messages)):
            split = _find_safe_split(messages, candidate)
            if split in seen_splits:
                continue
            seen_splits.add(split)
            suffix = messages[split:]
            if not suffix:
                continue
            if len(suffix) <= _FORK_MAX_MESSAGES and fallback_suffix is None:
                fallback_suffix = suffix
            if (
                len(suffix) <= _FORK_MAX_MESSAGES
                and SubAgentRunner._fork_message_chars(suffix) <= _FORK_MAX_CHARS
            ):
                return suffix
        # Preserve message and tool-call structure, but never send an unbounded
        # snapshot. Oversized textual payloads are previews; the delegated task
        # is appended separately after this compaction step.
        return SubAgentRunner._truncate_fork_payloads(
            fallback_suffix or messages,
        )

    @staticmethod
    def _truncate_fork_payloads(messages: list[dict]) -> list[dict]:
        bounded = deepcopy(messages)
        limit = 1_200
        while SubAgentRunner._fork_message_chars(bounded) > _FORK_MAX_CHARS:
            for message in bounded:
                content = message.get("content")
                if isinstance(content, str) and len(content) > limit:
                    message["content"] = SubAgentRunner._text_preview(content, limit)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        for key in ("text", "content"):
                            value = block.get(key)
                            if isinstance(value, str) and len(value) > limit:
                                block[key] = SubAgentRunner._text_preview(value, limit)
                        if (
                            block.get("type") == "tool_use"
                            and len(json.dumps(block.get("input"), default=str)) > limit
                        ):
                            block["input"] = {"_truncated": True}
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        function = call.get("function") if isinstance(call, dict) else None
                        if not isinstance(function, dict):
                            continue
                        arguments = function.get("arguments")
                        if isinstance(arguments, str) and len(arguments) > limit:
                            function["arguments"] = '{"_truncated":true}'
            if limit <= 64:
                break
            limit //= 2
        return bounded

    @staticmethod
    def _text_preview(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = max(1, limit * 2 // 3)
        tail = max(1, limit - head)
        return text[:head] + "\n…[fork 内容已截断]…\n" + text[-tail:]

    @staticmethod
    def _fork_message_chars(messages: list[dict]) -> int:
        return sum(
            len(json.dumps(message, ensure_ascii=False, default=str))
            for message in messages
        )

    @staticmethod
    def _unpaired_tool_call_ids(messages: list[dict]) -> set[str]:
        """Return tool IDs without a following provider-native result."""
        pending: set[str] = set()
        for message in messages:
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    pending.update(
                        call_id
                        for call in tool_calls
                        if isinstance(call, dict)
                        and isinstance((call_id := call.get("id")), str)
                        and call_id
                    )
                content = message.get("content")
                if isinstance(content, list):
                    pending.update(
                        call_id
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and isinstance((call_id := block.get("id")), str)
                        and call_id
                    )
            elif message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if isinstance(call_id, str):
                    pending.discard(call_id)
            elif message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and isinstance(block.get("tool_use_id"), str)
                        ):
                            pending.discard(block["tool_use_id"])
        return pending
