"""Agent Loop — ReAct pattern with prompt assembly, injections, and cache tracking."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import httpx

from tinyCode.agent.events import (
    AgentDoneEvent,
    BackgroundResultsAppliedEvent,
    AgentEvent,
    ErrorEvent,
    HITLRequestEvent,
    RoundLimitDecision,
    RoundLimitDecisionAction,
    RoundLimitExtendedEvent,
    RoundLimitReachedEvent,
    RoundStartEvent,
    ProgressWarningEvent,
    SteeringAppliedEvent,
    TaskStalledDecision,
    TaskStalledDecisionAction,
    TaskStalledEvent,
    TextDeltaEvent,
    ThinkingEvent,
    ToolBlockedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ContextCompressionEvent,
)
from tinyCode.agent.context import PromptContextAssembler
from tinyCode.agent.progress import ProgressState, ProgressWatchdog
from tinyCode.agent.tool_routing import (
    TaskMode,
    classify_task_mode,
    task_mode_instruction,
)
from tinyCode.agent.task_mode_router import TaskModeRouteResult, TaskModeRouter
from tinyCode.conversation.history import ConversationHistory
from tinyCode.conversation.truncator import ToolResultTruncator
from tinyCode.config.constants import (
    DEFAULT_HARD_MAX_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_ROUND_EXTENSION,
    MAX_ALLOWED_ROUNDS,
    SUPPORTED_ROUND_LIMIT_ACTIONS,
)
from tinyCode.conversation.compression import ContextCompressor
from tinyCode.conversation.summarizer import StructuredSummarizer
from tinyCode.notes.manager import AutoNoteManager
from tinyCode.hooks.engine import HookEngine
from tinyCode.hooks.models import HookEvent
from tinyCode.skills.registry import SkillRegistry
from tinyCode.providers.base import (
    BaseProvider,
    CacheUsage,
    MAX_PARALLEL_TOOL_CALLS,
    Message,
    ProviderError,
    TokenUsage,
    ToolCall,
)
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.security.guard import SecurityGuard
from tinyCode.security.models import HITLDecision, SecurityLevel
from tinyCode.storage.recovery import TaskRecoveryStore
from tinyCode.tools.base import ToolCategory, ToolResult
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tracing.recorder import TraceRecorder

DEFAULT_FIRST_EVENT_TIMEOUT = 45.0
DEFAULT_IDLE_EVENT_TIMEOUT = 60.0
DEFAULT_PROVIDER_RETRIES = 2
DEFAULT_RETRY_DELAY = 0.5
DEFAULT_MAX_RESPONSE_CHARS = 1_000_000
_PLAN_MODE_ALLOWED = {"read_file", "glob", "grep", "request_user_input"}
_DEFERRED_TOOL_RESULT_TOOLS = {"tool_result_search", "tool_result_read"}


@dataclass
class _TurnRoundBudget:
    current_limit: int
    extension: int
    hard_limit: int
    action: str
    auto_extend: bool
    round_number: int = 0


class AgentLoop:
    """ReAct 循环 + Prompt 拼装 + 缓存感知。"""

    def __init__(
        self,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        prompt_builder: PromptBuilder,
        prompt_injector: PromptInjector,
        security_guard: SecurityGuard | None = None,
        truncator: ToolResultTruncator | None = None,
        note_manager: AutoNoteManager | None = None,
        skill_registry: SkillRegistry | None = None,
        hook_engine: HookEngine | None = None,
        instructions_text: str = "",
        environment_text: str | Callable[[], str] = "",
        current_time_text: Callable[[], str] | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        round_extension: int = DEFAULT_ROUND_EXTENSION,
        hard_max_rounds: int = DEFAULT_HARD_MAX_ROUNDS,
        # Headless sub-agents have no UI capable of resolving an ``ask``
        # event. The interactive main runtime passes its configured action.
        round_limit_action: str = "stop",
        first_event_timeout: float = DEFAULT_FIRST_EVENT_TIMEOUT,
        idle_event_timeout: float = DEFAULT_IDLE_EVENT_TIMEOUT,
        provider_retries: int = DEFAULT_PROVIDER_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
        compressor: ContextCompressor | None = None,
        trace_recorder: TraceRecorder | None = None,
        recovery_store: TaskRecoveryStore | None = None,
        task_mode_router: TaskModeRouter | None = None,
        background_context: Callable[[], list[str]] | None = None,
        auto_extension_prompt: str = "",
        finalization_rounds: int = 0,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._prompt_builder = prompt_builder
        self._prompt_injector = prompt_injector
        self._security_guard = security_guard
        self._truncator = truncator
        self._note_manager = note_manager
        self._skill_registry = skill_registry
        self._hook_engine = hook_engine
        self._instructions_text = instructions_text
        self._environment_text = environment_text
        self._current_time_text = current_time_text
        self._hard_max_rounds = self._validate_round_count(
            hard_max_rounds, "hard_max_rounds",
        )
        self._max_rounds = self._validate_round_count(
            max_rounds, "max_rounds", maximum=self._hard_max_rounds,
        )
        self._round_extension = self._validate_round_count(
            round_extension, "round_extension",
        )
        self._round_limit_action = self._validate_round_limit_action(
            round_limit_action,
        )
        self._first_event_timeout = first_event_timeout
        self._idle_event_timeout = idle_event_timeout
        self._provider_retries = max(0, provider_retries)
        self._retry_delay = max(0.0, retry_delay)
        self._max_response_chars = max(1, max_response_chars)
        self._compressor = compressor
        self._trace_recorder = trace_recorder
        self._recovery_store = recovery_store
        self._task_mode_router = task_mode_router
        self._background_context = background_context
        self._auto_extension_prompt = (
            auto_extension_prompt.strip()
            if isinstance(auto_extension_prompt, str)
            else ""
        )
        self._finalization_rounds = max(
            0,
            min(
                finalization_rounds
                if isinstance(finalization_rounds, int)
                and not isinstance(finalization_rounds, bool)
                else 0,
                self._hard_max_rounds,
            ),
        )
        self._recovery_task_id: str | None = None
        self._context_assembler = PromptContextAssembler(
            protocol=provider.config.protocol,
            prompt_builder=prompt_builder,
            prompt_injector=prompt_injector,
            instructions_text=instructions_text,
            environment_text=environment_text,
            notes_text=self._current_notes_text,
            skill_registry=skill_registry,
        )

        self._plan_only = False
        self._cancel_event = asyncio.Event()
        self.cache_hit = False
        self.turn_cache_usage = CacheUsage()
        self.turn_usage = TokenUsage()
        self.turn_model_requests = 0
        self._active_round = 0
        self._active_budget: _TurnRoundBudget | None = None
        self._task_environment_text: str | None = None
        self._task_notes_text: str | None = None
        self._task_skill_instructions: str | None = None
        self._task_injection: str | None = None
        self._task_tool_defs: list[dict] | None = None
        self._task_mode = TaskMode.DIRECT
        self._task_mode_source = "rule"
        self._task_mode_confidence: float | None = None
        self._task_snapshot_version = 0
        self._task_is_direct_answer = False
        self._task_needs_time = False
        self._previous_request_cache_shape: dict[str, object] | None = None

    # -- public API -----------------------------------------------------------

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    def set_recovery_task(self, task_id: str | None) -> None:
        """Bind tool WAL writes to the active foreground task."""
        self._recovery_task_id = task_id

    @property
    def plan_only(self) -> bool:
        return self._plan_only

    @property
    def max_rounds(self) -> int:
        """Return the maximum rounds used by subsequent turns."""
        return self._max_rounds

    def set_max_rounds(self, value: int) -> int:
        """Update the soft budget and an in-flight task's effective limit."""
        self._max_rounds = self._validate_round_count(
            value, "max_rounds", maximum=self._hard_max_rounds,
        )
        if self._active_budget is not None:
            self._active_budget.current_limit = max(
                self._active_budget.round_number,
                min(self._max_rounds, self._active_budget.hard_limit),
            )
        return self._max_rounds

    @property
    def round_extension(self) -> int:
        return self._round_extension

    def set_round_extension(self, value: int) -> int:
        self._round_extension = self._validate_round_count(
            value, "round_extension",
        )
        if self._active_budget is not None:
            self._active_budget.extension = self._round_extension
        return self._round_extension

    @property
    def hard_max_rounds(self) -> int:
        return self._hard_max_rounds

    def set_hard_max_rounds(self, value: int) -> int:
        validated = self._validate_round_count(value, "hard_max_rounds")
        if validated < self._max_rounds:
            raise ValueError(
                f"hard_max_rounds 不能小于 max_rounds（{self._max_rounds}）"
            )
        active_round = (
            self._active_budget.round_number if self._active_budget else 0
        )
        if active_round and validated < active_round:
            raise ValueError(
                f"hard_max_rounds 不能小于当前轮次（{active_round}）"
            )
        self._hard_max_rounds = validated
        if self._active_budget is not None:
            self._active_budget.hard_limit = validated
            self._active_budget.current_limit = min(
                self._active_budget.current_limit, validated,
            )
        return self._hard_max_rounds

    @property
    def round_limit_action(self) -> str:
        return self._round_limit_action

    def set_round_limit_action(self, value: str) -> str:
        self._round_limit_action = self._validate_round_limit_action(value)
        if self._active_budget is not None:
            self._active_budget.action = self._round_limit_action
            self._active_budget.auto_extend = self._round_limit_action == "auto"
        return self._round_limit_action

    @staticmethod
    def _validate_round_count(
        value: int, name: str, *, maximum: int = MAX_ALLOWED_ROUNDS,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"{name} 必须是 1 到 {maximum} 之间的整数")
        return value

    @staticmethod
    def _validate_round_limit_action(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("round_limit_action 必须是 ask、auto 或 stop")
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_ROUND_LIMIT_ACTIONS:
            raise ValueError("round_limit_action 必须是 ask、auto 或 stop")
        return normalized

    def get_system_prompt(self, section: str = "all") -> str:
        """Return a read-only snapshot of the prompt context sent to the model."""
        active_skills = ""
        if self._skill_registry:
            active_skills = self._skill_registry.get_active_instructions()

        sections = {
            "base": self._prompt_builder.build(),
            "instructions": self._instructions_text,
            "skills": active_skills,
            "environment": self._current_environment_text(),
            "notes": self._current_notes_text(),
            "injection": self._prompt_injector.preview_injection(1) or "",
        }
        labels = {
            "base": "Base System Prompt",
            "instructions": "Instructions",
            "skills": "Activated Skills",
            "environment": "Environment",
            "notes": "Notes",
            "injection": "Dynamic Injection（下一轮预览）",
        }

        normalized = section.strip().lower()
        if normalized != "all" and normalized not in sections:
            raise ValueError(
                "可用部分: all, base, instructions, skills, environment, notes, injection"
            )

        selected = sections if normalized == "all" else {normalized: sections[normalized]}
        protocol = self._provider.config.protocol
        output = [f"系统上下文快照 · protocol={protocol}"]
        for name, content in selected.items():
            output.append(f"\n--- {labels[name]} ---\n{content or '(空)'}")
        return "\n".join(output)

    def toggle_plan_only(self) -> bool:
        self._plan_only = not self._plan_only
        self._prompt_injector.set_plan_only(self._plan_only)
        return self._plan_only

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset_cancel(self) -> None:
        self._cancel_event.clear()

    async def run(self, history: ConversationHistory) -> AsyncIterator[AgentEvent]:
        """Run one user turn and keep history valid if the pipeline fails."""
        history.flush_deferred()
        self._apply_background_context(history)
        start_index = len(history.get_messages())
        self.cache_hit = False
        self.turn_cache_usage = CacheUsage()
        self.turn_usage = TokenUsage()
        self.turn_model_requests = 0
        self._previous_request_cache_shape = None
        await self._begin_task_prompt_snapshot(history)
        task_snapshot_version = self._task_snapshot_version
        try:
            async for event in self._run_impl(history):
                yield event
        except asyncio.CancelledError:
            self._repair_unpaired_tool_calls(history, start_index)
            if self._active_round:
                await self._fire_round_end(self._active_round, 0, "cancelled")
                self._active_round = 0
            raise
        except ProviderError as exc:
            self._repair_unpaired_tool_calls(history, start_index)
            await self._fire_error(str(exc), exc.code, self._active_round)
            self._active_round = 0
            yield ErrorEvent(
                message=str(exc),
                code=exc.code,
                retryable=exc.retryable,
            )
        except Exception as exc:
            self._repair_unpaired_tool_calls(history, start_index)
            await self._fire_error(
                str(exc), type(exc).__name__, self._active_round,
            )
            self._active_round = 0
            yield ErrorEvent(
                message=f"Agent 执行失败: {type(exc).__name__}: {exc}",
            )
        finally:
            self._active_budget = None
            self._clear_task_prompt_snapshot(task_snapshot_version)

    async def _run_impl(self, history: ConversationHistory) -> AsyncIterator[AgentEvent]:
        self.reset_cancel()
        response_chars = 0
        task_mode = self._task_mode
        tools_enabled = task_mode.tools_enabled
        # Skill activation and large-result persistence used to rebuild this
        # list between ReAct rounds.  Freeze the advertised schema for one
        # task; execution-time policy checks remain authoritative.
        self._task_tool_defs = (
            self._build_tool_defs(task_mode=task_mode) if tools_enabled else None
        )
        request_history = self._request_history(history, tools_enabled)
        budget = _TurnRoundBudget(
            current_limit=self._max_rounds,
            extension=self._round_extension,
            hard_limit=self._hard_max_rounds,
            action=self._round_limit_action,
            auto_extend=self._round_limit_action == "auto",
        )
        self._active_budget = budget
        progress_watchdog = ProgressWatchdog()
        round_num = 1

        while round_num <= budget.hard_limit:
            budget.round_number = round_num
            if self._cancel_event.is_set():
                yield AgentDoneEvent("cancelled")
                return

            self._active_round = round_num

            remaining_rounds = budget.hard_limit - round_num + 1
            if (
                self._finalization_rounds
                and remaining_rounds <= self._finalization_rounds
            ):
                if remaining_rounds == 1:
                    finalization_prompt = (
                        "[最终收敛阶段] 这是最后一个可用轮次。禁止继续调用工具；"
                        "请立即基于已有证据输出完整、可交付的最终结果。"
                    )
                else:
                    finalization_prompt = (
                        f"[最终收敛阶段] 仅剩 {remaining_rounds} 个轮次。停止扩大"
                        "调查范围；最多完成一批不可缺少的读取，然后输出最终结果。"
                    )
                self._prompt_injector.queue_injection(finalization_prompt)

            # Recovery instructions are real events in the conversation, not
            # a temporary prefix injected before all prior messages.  Keeping
            # them append-only preserves earlier cacheable history.
            for injection in self._prompt_injector.consume_pending_injections():
                request_history.add_context_message(injection)

            prepared_messages: list | None = None
            if self._compressor is not None:
                history_tokens = StructuredSummarizer._estimate_tokens(
                    request_history.get_messages()
                )
                assembled = self._assemble_messages(request_history, round_num)
                full_tokens = StructuredSummarizer._estimate_tokens(assembled)
                if tools_enabled:
                    full_tokens += StructuredSummarizer._estimate_tokens([{
                        "role": "system",
                        "content": json.dumps(
                            # Tool schemas are frozen for this task, so this
                            # mirrors the request payload exactly.
                            self._task_tool_defs or [],
                            ensure_ascii=False,
                        ),
                    }])
                compression_scope = (
                    self._trace_recorder.span(
                        "context_compression",
                        "context_compression",
                        {
                            "round": round_num,
                            "estimated_tokens": full_tokens,
                            "context_window": getattr(
                                self._compressor, "context_window", 0,
                            ),
                        },
                    )
                    if self._trace_recorder is not None
                    else nullcontext(None)
                )
                with compression_scope as compression_span:
                    comp = await self._compressor.check_and_compress(
                        request_history,
                        self._provider,
                        extra_tokens=max(0, full_tokens - history_tokens),
                    )
                    if compression_span is not None:
                        compression_raw_usage = getattr(
                            self._provider, "last_usage", None,
                        )
                        compression_usage = (
                            TokenUsage.from_raw(compression_raw_usage)
                            if comp.model_request_made
                            else TokenUsage()
                        )
                        compression_cache_usage = (
                            CacheUsage.from_raw(compression_raw_usage)
                            if comp.model_request_made
                            else CacheUsage()
                        )
                        compression_span.finish(
                            "error" if comp.error else "ok",
                            {
                                "round": round_num,
                                "warning_issued": comp.warning_issued,
                                "was_compressed": comp.was_compressed,
                                "model_request_made": comp.model_request_made,
                                "estimated_tokens_before": comp.estimated_tokens_before,
                                "estimated_tokens_after": comp.estimated_tokens_after,
                                "error": comp.error,
                                "input_tokens": compression_usage.input_tokens,
                                "output_tokens": compression_usage.output_tokens,
                                "total_tokens": compression_usage.total_tokens,
                                "cache_read_tokens": compression_cache_usage.read_tokens,
                                "cache_write_tokens": compression_cache_usage.write_tokens,
                                "cache_miss_tokens": compression_cache_usage.miss_tokens,
                                "cache_usage_available": compression_cache_usage.available,
                                "cache_hit": compression_cache_usage.read_tokens > 0,
                            },
                        )
                if comp.model_request_made:
                    self.turn_model_requests += 1
                    raw_usage = getattr(self._provider, "last_usage", None)
                    self.turn_usage = self.turn_usage + TokenUsage.from_raw(raw_usage)
                    self.turn_cache_usage = self.turn_cache_usage + CacheUsage.from_raw(
                        raw_usage
                    )
                    self.cache_hit = self.turn_cache_usage.read_tokens > 0
                if comp.warning_issued or comp.was_compressed or comp.error:
                    yield ContextCompressionEvent(
                        warning_issued=comp.warning_issued,
                        was_compressed=comp.was_compressed,
                        estimated_tokens_before=comp.estimated_tokens_before,
                        estimated_tokens_after=comp.estimated_tokens_after,
                        error=comp.error,
                    )
                if comp.was_compressed and self._hook_engine:
                    await self._hook_engine.fire(HookEvent.SYSTEM_COMPRESS, {
                        "tokens_before": comp.estimated_tokens_before,
                        "tokens_after": comp.estimated_tokens_after,
                    })
                if (
                    comp.error
                    and comp.estimated_tokens_after >= self._compressor.context_window
                ):
                    message = f"上下文已超过窗口且压缩失败：{comp.error}"
                    await self._fire_error(
                        message, "context_compression_failed", round_num,
                    )
                    self._active_round = 0
                    yield ErrorEvent(
                        message=message,
                        code="context_compression_failed",
                    )
                    return
                # Reuse the assembled request when history is unchanged. On
                # long conversations this avoids a second full deepcopy and
                # prompt rebuild before every model request.
                if not comp.was_compressed:
                    prepared_messages = assembled

            if self._trace_recorder is not None:
                self._trace_recorder.record("round_start", attributes={
                    "round": round_num,
                    "max_rounds": budget.current_limit,
                    "hard_limit": budget.hard_limit,
                })
            yield RoundStartEvent(
                round_number=round_num,
                max_rounds=budget.current_limit,
            )

            # --- 1. 拼装本轮 messages ---
            messages = (
                prepared_messages
                if prepared_messages is not None
                else self._assemble_messages(request_history, round_num)
            )

            # --- 1.5. Layer 1 截断 ---
            if self._truncator is not None:
                messages, trunc_infos = self._truncator.process_round(messages)
                for info in trunc_infos:
                    from tinyCode.agent.events import TruncationEvent
                    if self._trace_recorder is not None:
                        self._trace_recorder.record("truncation", attributes={
                            "round": round_num,
                            "tool": info["tool_name"],
                            "original_chars": info["original_chars"],
                            "file_path": info["file_path"],
                        })
                    yield TruncationEvent(
                        tool_name=info["tool_name"],
                        original_chars=info["original_chars"],
                        file_path=info["file_path"],
                    )

            # --- Hook: ROUND_START ---
            if self._hook_engine:
                await self._hook_engine.fire(HookEvent.ROUND_START, {
                    "round_number": round_num,
                    "max_rounds": budget.current_limit,
                })

            # --- Hook: MESSAGE_PRE_SEND ---
            if self._hook_engine:
                hook_prompt = await self._hook_engine.fire(HookEvent.MESSAGE_PRE_SEND, {
                    "message_count": len(messages),
                })
                if hook_prompt:
                    messages.append({
                        "role": "user",
                        "content": f"[Hook Injection]\n{hook_prompt}",
                    })

            # --- 2. 调 LLM ---
            tool_calls: list[ToolCall] = []
            tool_call_ids: set[str] = set()
            text_parts: list[str] = []
            tool_defs = (
                self._task_tool_defs
                if tools_enabled else None
            )

            async for raw in self._stream_provider(
                messages=messages,
                tools=tool_defs,
                system_blocks=self._build_system_blocks(),
            ):
                if self._cancel_event.is_set():
                    await self._fire_round_end(
                        round_num, len(tool_calls), "cancelled",
                    )
                    self._active_round = 0
                    yield AgentDoneEvent("cancelled")
                    return

                if isinstance(raw, str):
                    response_chars += len(raw)
                    if response_chars > self._max_response_chars:
                        raise ProviderError(
                            f"模型单次任务响应超过 {self._max_response_chars} 字符限制",
                            code="response_too_large",
                        )
                    if raw.startswith("<<THINKING:") or raw.startswith("<<REASONING:"):
                        label = "Thinking" if raw.startswith("<<THINKING:") else "Reasoning"
                        text = raw[len("<<THINKING:"):-2] if raw.startswith("<<THINKING:") else raw[len("<<REASONING:"):-2]
                        yield ThinkingEvent(text=text, label=label)
                    elif raw.startswith("<<ERROR:"):
                        message = raw[len("<<ERROR:"):-2]
                        await self._fire_error(message, "provider_event", round_num)
                        self._active_round = 0
                        yield ErrorEvent(message=message)
                        return
                    else:
                        text_parts.append(raw)
                        yield TextDeltaEvent(text=raw)
                elif isinstance(raw, ToolCall):
                    if not tools_enabled:
                        message = "模型请求了本轮未开放的工作区工具，已拒绝执行"
                        await self._fire_error(
                            message, "unadvertised_tool_call", round_num,
                        )
                        self._active_round = 0
                        yield ErrorEvent(
                            message=message,
                            code="unadvertised_tool_call",
                        )
                        return
                    identity_error = self._validate_tool_call_identity(raw)
                    if identity_error is not None:
                        await self._fire_error(
                            identity_error, "invalid_tool_call", round_num,
                        )
                        self._active_round = 0
                        yield ErrorEvent(message=identity_error)
                        return
                    if raw.id in tool_call_ids:
                        message = f"模型返回了重复的工具调用 ID: {raw.id}"
                        await self._fire_error(
                            message, "duplicate_tool_call_id", round_num,
                        )
                        self._active_round = 0
                        yield ErrorEvent(
                            message=message,
                            code="duplicate_tool_call_id",
                        )
                        return
                    if len(tool_calls) >= MAX_PARALLEL_TOOL_CALLS:
                        message = (
                            "模型单次响应的工具调用数量超过 "
                            f"{MAX_PARALLEL_TOOL_CALLS} 个限制"
                        )
                        await self._fire_error(
                            message, "too_many_tool_calls", round_num,
                        )
                        self._active_round = 0
                        yield ErrorEvent(
                            message=message,
                            code="too_many_tool_calls",
                        )
                        return
                    if isinstance(raw.input, dict):
                        try:
                            response_chars += len(json.dumps(
                                raw.input, ensure_ascii=False,
                            ))
                        except (TypeError, ValueError):
                            message = f"工具 '{raw.name}' 的参数不是有效 JSON 对象"
                            await self._fire_error(
                                message, "malformed_tool_arguments", round_num,
                            )
                            self._active_round = 0
                            yield ErrorEvent(
                                message=message,
                                code="malformed_tool_arguments",
                            )
                            return
                        if response_chars > self._max_response_chars:
                            raise ProviderError(
                                "模型单次任务返回的文本和工具参数总量超过 "
                                f"{self._max_response_chars} 字符限制",
                                code="response_too_large",
                            )
                    tool_call_ids.add(raw.id)
                    tool_calls.append(raw)
                    if self._trace_recorder is not None:
                        attributes = self._trace_recorder.tool_attributes(
                            raw.name, raw.input,
                        )
                        attributes.update({
                            "round": round_num,
                            "call_id": raw.id,
                        })
                        self._trace_recorder.record(
                            "tool_call",
                            name=raw.name,
                            attributes=attributes,
                        )
                else:
                    message = (
                        "Provider 输出事件必须是字符串或 ToolCall，"
                        f"实际收到: {type(raw).__name__}"
                    )
                    await self._fire_error(message, "invalid_provider_event", round_num)
                    self._active_round = 0
                    yield ErrorEvent(message=message)
                    return

            self.turn_usage = self.turn_usage + TokenUsage.from_raw(
                getattr(self._provider, "last_usage", None)
            )

            # --- Hook: MESSAGE_POST_RECEIVE ---
            if self._hook_engine:
                await self._hook_engine.fire(HookEvent.MESSAGE_POST_RECEIVE, {
                    "text": "".join(text_parts)[:500],
                    "tool_calls_count": len(tool_calls),
                })

            # --- 3. 检测缓存命中 ---
            cache_usage = CacheUsage.from_raw(
                getattr(self._provider, "last_usage", None)
            )
            self.turn_cache_usage = self.turn_cache_usage + cache_usage
            self.cache_hit = self.turn_cache_usage.read_tokens > 0

            if not text_parts and not tool_calls:
                message = "模型返回空响应，请重试本轮对话"
                await self._fire_error(message, "empty_response", round_num)
                self._active_round = 0
                yield ErrorEvent(message=message)
                return

            # --- 4. 无工具调用 → 终止 ---
            if not tool_calls:
                if text_parts:
                    response_text = "".join(text_parts)
                    history.add_assistant_message(response_text)
                    if request_history is not history:
                        request_history.add_assistant_message(response_text)
                history_size_before_steering = len(history.get_messages())
                background_count = self._apply_background_context(history)
                deferred_count = history.flush_steering()
                pending_count = deferred_count + background_count
                if pending_count:
                    if request_history is not history:
                        # Copy only newly flushed input into the isolated current
                        # turn, keeping unrelated previous answers out.
                        for message in history.get_messages()[
                            history_size_before_steering:
                        ]:
                            request_history.add_raw_message(message)
                    can_continue = round_num < budget.hard_limit
                    if deferred_count:
                        yield SteeringAppliedEvent(
                            message_count=deferred_count,
                            continued=can_continue,
                        )
                    if background_count:
                        yield BackgroundResultsAppliedEvent(
                            result_count=background_count,
                            continued=can_continue,
                        )
                    if can_continue:
                        previous_limit = budget.current_limit
                        budget.current_limit = min(
                            budget.hard_limit,
                            max(
                                budget.current_limit,
                                round_num + budget.extension,
                            ),
                        )
                        await self._fire_round_end(
                            round_num, 0, "continued_by_user_steering",
                        )
                        self._active_round = 0
                        if budget.current_limit > previous_limit:
                            yield RoundLimitExtendedEvent(
                                previous_limit=previous_limit,
                                new_limit=budget.current_limit,
                                hard_limit=budget.hard_limit,
                                automatic=False,
                            )
                        round_num += 1
                        continue
                    await self._fire_round_end(
                        round_num, 0, "hard_max_rounds",
                    )
                    self._active_round = 0
                    yield AgentDoneEvent("hard_max_rounds")
                    return
                await self._fire_round_end(round_num, 0, "completed")
                self._active_round = 0
                yield AgentDoneEvent("no_tool_call")
                return

            # --- 5. 合并文本 + 工具调用为单条 assistant 消息 ---
            text_prefix = "".join(text_parts)
            tc_msg = self._provider.make_tool_calls_message(tool_calls, text_prefix=text_prefix)
            history.add_raw_message(tc_msg)
            # Publish calls only after the complete assistant tool-call batch
            # exists in history. The TUI checkpoints on these events before
            # execution, so a hard crash can never leave an invisible write.
            for tool_call in tool_calls:
                yield ToolCallEvent(tool_call=tool_call)

            # --- 6. 工具分批执行（含安全检查） ---
            reads, writes = self._partition_tools(tool_calls)
            round_observations: list[tuple[ToolCall, ToolResult]] = []

            # 读类 — 并发（安全检查前置）
            valid_reads: list[ToolCall] = []
            for tc in reads:
                invalid_result = self._validate_tool_call_input(tc)
                if invalid_result is not None:
                    self._append_tool_result(history, tc, invalid_result)
                    round_observations.append((tc, invalid_result))
                    yield ToolResultEvent(
                        tool_name=tc.name, call_id=tc.id, result=invalid_result,
                    )
                    continue

                if not self._tool_allowed_for_task(tc):
                    blocked = await self._block_tool(tc)
                    blocked_result = ToolResult(
                        success=False, content="", error=blocked.reason,
                    )
                    self._append_tool_result(history, tc, blocked_result)
                    round_observations.append((tc, blocked_result))
                    yield blocked
                    continue

                allowed, reason, hitl_future = self._precheck_tool(tc)
                if hitl_future is not None:
                    guard = self._security_guard
                    if guard is None:
                        raise RuntimeError("安全确认 Future 存在但 SecurityGuard 未配置")
                    approval_params = self._approval_parameters(tc)
                    prompt = guard.build_hitl_prompt(tc.name, approval_params)
                    yield HITLRequestEvent(
                        tool_name=tc.name, params=approval_params,
                        prompt=prompt, future=hitl_future,
                    )
                    decision = await hitl_future
                    if decision == HITLDecision.DENY:
                        blocked_result = ToolResult(success=False, content="", error="用户拒绝了该操作")
                        self._append_tool_result(history, tc, blocked_result)
                        round_observations.append((tc, blocked_result))
                        yield ToolResultEvent(
                            tool_name=tc.name, call_id=tc.id, result=blocked_result,
                        )
                        continue
                    guard.apply_hitl(
                        decision, tc.name, self._security_parameters(tc),
                    )
                elif not allowed:
                    blocked_result = ToolResult(success=False, content="", error=reason)
                    self._append_tool_result(history, tc, blocked_result)
                    round_observations.append((tc, blocked_result))
                    yield ToolResultEvent(
                        tool_name=tc.name, call_id=tc.id, result=blocked_result,
                    )
                    continue
                valid_reads.append(tc)

            if valid_reads:
                results = await self._execute_concurrent(valid_reads)
                for tc, result in zip(valid_reads, results):
                    self._append_tool_result(history, tc, result)
                    round_observations.append((tc, result))
                    yield ToolResultEvent(
                        tool_name=tc.name, call_id=tc.id, result=result,
                    )

            # 写类 — 串行（每个执行前检查）
            for tc in writes:
                if self._cancel_event.is_set():
                    await self._fire_round_end(
                        round_num, len(tool_calls), "cancelled",
                    )
                    self._active_round = 0
                    yield AgentDoneEvent("cancelled")
                    return

                invalid_result = self._validate_tool_call_input(tc)
                if invalid_result is not None:
                    self._append_tool_result(history, tc, invalid_result)
                    round_observations.append((tc, invalid_result))
                    yield ToolResultEvent(
                        tool_name=tc.name, call_id=tc.id, result=invalid_result,
                    )
                    continue

                if not self._tool_allowed_for_task(tc):
                    blocked = await self._block_tool(tc)
                    blocked_result = ToolResult(
                        success=False, content="", error=blocked.reason,
                    )
                    self._append_tool_result(history, tc, blocked_result)
                    round_observations.append((tc, blocked_result))
                    yield blocked
                    continue

                allowed, reason, hitl_future = self._precheck_tool(tc)
                if hitl_future is not None:
                    guard = self._security_guard
                    if guard is None:
                        raise RuntimeError("安全确认 Future 存在但 SecurityGuard 未配置")
                    approval_params = self._approval_parameters(tc)
                    prompt = guard.build_hitl_prompt(tc.name, approval_params)
                    yield HITLRequestEvent(
                        tool_name=tc.name, params=approval_params,
                        prompt=prompt, future=hitl_future,
                    )
                    decision = await hitl_future
                    if decision == HITLDecision.DENY:
                        blocked_result = ToolResult(success=False, content="", error="用户拒绝了该操作")
                        self._append_tool_result(history, tc, blocked_result)
                        round_observations.append((tc, blocked_result))
                        yield ToolResultEvent(
                            tool_name=tc.name, call_id=tc.id, result=blocked_result,
                        )
                        continue
                    guard.apply_hitl(
                        decision, tc.name, self._security_parameters(tc),
                    )
                elif not allowed:
                    blocked_result = ToolResult(success=False, content="", error=reason)
                    self._append_tool_result(history, tc, blocked_result)
                    round_observations.append((tc, blocked_result))
                    yield ToolResultEvent(
                        tool_name=tc.name, call_id=tc.id, result=blocked_result,
                    )
                    continue

                result = await self._execute_tool_with_hooks(tc)

                self._append_tool_result(history, tc, result)
                round_observations.append((tc, result))
                yield ToolResultEvent(
                    tool_name=tc.name, call_id=tc.id, result=result,
                )

            background_count = self._apply_background_context(history)
            deferred_count = history.flush_steering()
            pending_count = deferred_count + background_count
            if pending_count:
                await self._refresh_task_mode_after_steering(history)
                tools_enabled = self._task_mode.tools_enabled
                if request_history is not history:
                    request_history = history
                can_continue = round_num < budget.hard_limit
                if deferred_count:
                    yield SteeringAppliedEvent(
                        message_count=deferred_count,
                        continued=can_continue,
                    )
                if background_count:
                    yield BackgroundResultsAppliedEvent(
                        result_count=background_count,
                        continued=can_continue,
                    )
                if can_continue:
                    previous_limit = budget.current_limit
                    budget.current_limit = min(
                        budget.hard_limit,
                        max(
                            budget.current_limit,
                            round_num + budget.extension,
                        ),
                    )
                    if budget.current_limit > previous_limit:
                        yield RoundLimitExtendedEvent(
                            previous_limit=previous_limit,
                            new_limit=budget.current_limit,
                            hard_limit=budget.hard_limit,
                            automatic=False,
                        )

            progress = progress_watchdog.observe(round_num, round_observations)
            if pending_count:
                # A user steering message supersedes the strategy that produced
                # this round. Give the new direction a fresh observation window.
                progress_watchdog.reset_strategy()
            elif progress.state == ProgressState.SLOW:
                self._prompt_injector.queue_injection(progress.recovery_prompt)
                yield ProgressWarningEvent(
                    state=progress.state.value,
                    reasons=progress.reasons,
                    recovery_prompt=progress.recovery_prompt,
                )

            if round_num >= budget.hard_limit:
                if not pending_count and progress.requires_intervention:
                    yield ProgressWarningEvent(
                        state=progress.state.value,
                        reasons=progress.reasons,
                        recovery_prompt=progress.recovery_prompt,
                    )
                await self._fire_round_end(
                    round_num, len(tool_calls), "hard_max_rounds",
                )
                self._active_round = 0
                yield AgentDoneEvent("hard_max_rounds")
                return

            if not pending_count and progress.requires_intervention:
                await self._fire_round_end(
                    round_num, len(tool_calls), "awaiting_progress_decision",
                )
                self._active_round = 0
                default_continue = min(5, budget.hard_limit - round_num)
                if budget.action == "stop":
                    yield ProgressWarningEvent(
                        state=progress.state.value,
                        reasons=progress.reasons,
                        recovery_prompt=progress.recovery_prompt,
                    )
                    yield AgentDoneEvent("stalled")
                    return
                if budget.auto_extend:
                    yield ProgressWarningEvent(
                        state=progress.state.value,
                        reasons=progress.reasons,
                        recovery_prompt=progress.recovery_prompt,
                    )
                    stall_decision = TaskStalledDecision(
                        TaskStalledDecisionAction.STRATEGY,
                        max(1, default_continue),
                    )
                else:
                    future = asyncio.get_running_loop().create_future()
                    yield TaskStalledEvent(
                        state=progress.state.value,
                        reasons=progress.reasons,
                        recovery_prompt=progress.recovery_prompt,
                        round_number=round_num,
                        continue_rounds=max(1, default_continue),
                        hard_limit=budget.hard_limit,
                        future=future,
                    )
                    stall_decision = await future
                if not isinstance(stall_decision, TaskStalledDecision):
                    stall_decision = TaskStalledDecision(
                        TaskStalledDecisionAction.STOP,
                    )
                if stall_decision.action == TaskStalledDecisionAction.STOP:
                    yield AgentDoneEvent("stalled")
                    return

                if (
                    isinstance(stall_decision.continue_rounds, int)
                    and not isinstance(stall_decision.continue_rounds, bool)
                    and stall_decision.continue_rounds > 0
                ):
                    requested_rounds = stall_decision.continue_rounds
                elif stall_decision.action == TaskStalledDecisionAction.STRATEGY:
                    requested_rounds = budget.extension
                else:
                    requested_rounds = max(1, default_continue)
                previous_limit = budget.current_limit
                budget.current_limit = min(
                    budget.hard_limit,
                    max(budget.current_limit, round_num + requested_rounds),
                )
                if stall_decision.action == TaskStalledDecisionAction.STRATEGY:
                    self._prompt_injector.queue_injection(progress.recovery_prompt)
                progress_watchdog.reset_strategy()
                if budget.current_limit > previous_limit:
                    yield RoundLimitExtendedEvent(
                        previous_limit=previous_limit,
                        new_limit=budget.current_limit,
                        hard_limit=budget.hard_limit,
                        automatic=budget.auto_extend,
                    )
                round_num += 1
                continue

            if round_num >= budget.current_limit:
                decision: RoundLimitDecision
                if budget.action == "stop":
                    decision = RoundLimitDecision(RoundLimitDecisionAction.STOP)
                elif budget.auto_extend:
                    decision = RoundLimitDecision(RoundLimitDecisionAction.AUTO)
                else:
                    await self._fire_round_end(
                        round_num, len(tool_calls), "awaiting_round_extension",
                    )
                    self._active_round = 0
                    future = asyncio.get_running_loop().create_future()
                    yield RoundLimitReachedEvent(
                        round_number=round_num,
                        current_limit=budget.current_limit,
                        extension=budget.extension,
                        hard_limit=budget.hard_limit,
                        stalled=False,
                        future=future,
                    )
                    decision = await future

                if decision.action == RoundLimitDecisionAction.STOP:
                    if self._active_round:
                        await self._fire_round_end(
                            round_num, len(tool_calls), "round_budget_stopped",
                        )
                    self._active_round = 0
                    yield AgentDoneEvent("round_budget_stopped")
                    return

                previous_limit = budget.current_limit
                requested_limit = decision.requested_limit
                if requested_limit is None:
                    requested_limit = round_num + budget.extension
                budget.current_limit = min(
                    budget.hard_limit,
                    max(previous_limit, round_num + 1, requested_limit),
                )
                if decision.action == RoundLimitDecisionAction.AUTO:
                    budget.auto_extend = True
                    if self._auto_extension_prompt:
                        self._prompt_injector.queue_injection(
                            self._auto_extension_prompt,
                        )
                if self._active_round:
                    await self._fire_round_end(
                        round_num, len(tool_calls), "continued",
                    )
                self._active_round = 0
                yield RoundLimitExtendedEvent(
                    previous_limit=previous_limit,
                    new_limit=budget.current_limit,
                    hard_limit=budget.hard_limit,
                    automatic=decision.action == RoundLimitDecisionAction.AUTO,
                )
            else:
                await self._fire_round_end(
                    round_num, len(tool_calls), "continued",
                )
                self._active_round = 0

            round_num += 1

    @staticmethod
    def _request_history(
        history: ConversationHistory,
        tools_enabled: bool,
    ) -> ConversationHistory:
        """Return the history visible to this provider request.

        High-confidence self-contained turns intentionally receive only the
        latest user request. This prevents a fresh merge-sort request from
        being biased into replaying the preceding quicksort answer. Workspace
        and continuation turns keep the original complete history.
        """
        if tools_enabled:
            return history
        trailing_users: list[Message] = []
        for message in reversed(history.get_messages()):
            if message.get("role") != "user":
                break
            if isinstance(message.get("content"), str):
                trailing_users.append(message)
        if trailing_users:
            isolated = ConversationHistory()
            for message in reversed(trailing_users):
                isolated.add_raw_message(message)
            return isolated
        return history

    async def _fire_round_end(
        self, round_number: int, tool_calls_count: int, outcome: str,
    ) -> None:
        if self._trace_recorder is not None:
            self._trace_recorder.record("round_end", status=outcome, attributes={
                "round": round_number,
                "tool_calls": tool_calls_count,
                "outcome": outcome,
            })
        if self._hook_engine:
            await self._hook_engine.fire(HookEvent.ROUND_END, {
                "round_number": round_number,
                "tool_calls_count": tool_calls_count,
                "outcome": outcome,
            })

    async def _fire_error(
        self, message: str, code: str, round_number: int = 0,
    ) -> None:
        if self._trace_recorder is not None:
            self._trace_recorder.record("error", status="error", attributes={
                "error": message,
                "code": code,
                "round": round_number,
            })
        if self._hook_engine:
            await self._hook_engine.fire(HookEvent.SYSTEM_ERROR, {
                "error": message, "code": code, "round_number": round_number,
            })
        if round_number:
            await self._fire_round_end(round_number, 0, "error")

    # -- message assembly -----------------------------------------------------

    def _apply_background_context(self, history: ConversationHistory) -> int:
        """Append completed worker reports as bounded internal context."""
        if self._background_context is None:
            return 0
        try:
            messages = self._background_context()
        except Exception as exc:
            if self._trace_recorder is not None:
                self._trace_recorder.record(
                    "background_context_error",
                    status="error",
                    attributes={"error_type": type(exc).__name__, "error": str(exc)},
                )
            return 0
        count = 0
        for content in messages:
            if not isinstance(content, str) or not content.strip():
                continue
            history.add_context_message(content)
            count += 1
        if count and self._trace_recorder is not None:
            self._trace_recorder.record(
                "background_results_applied",
                attributes={"result_count": count},
            )
        return count

    def _assemble_messages(
        self, history: ConversationHistory, round_num: int,
    ) -> list:
        """Build the messages array for this round.

        Structure (for Anthropic): system is sent separately via _build_system_blocks.
        For OpenAI/DeepSeek: system prompt + env + injections all go in messages.
        """
        return self._context_assembler.assemble(
            history,
            round_num,
            environment_text=self._task_environment_text,
            notes_text=self._task_notes_text,
            skill_instructions=self._task_skill_instructions,
            task_injection=self._task_injection,
            direct_answer=self._task_is_direct_answer,
            include_environment=(
                not self._task_is_direct_answer or self._task_needs_time
            ),
        )

    def _build_system_blocks(self) -> list[dict] | None:
        """Build Anthropic system blocks (None for other providers)."""
        return self._context_assembler.system_blocks(
            direct_answer=self._task_is_direct_answer,
        )

    async def _begin_task_prompt_snapshot(self, history: ConversationHistory) -> None:
        """Freeze dynamic prefix inputs for the lifetime of one user task."""
        self._task_snapshot_version += 1
        route = await self._route_task_mode(history.get_messages())
        self._apply_task_mode_route(route)
        if self._plan_only and self._task_mode is TaskMode.MODIFY:
            self._task_mode = TaskMode.INSPECT
        self._task_is_direct_answer = self._task_mode is TaskMode.DIRECT
        self._task_needs_time = self._task_needs_current_time(history)
        self._task_environment_text = self._current_environment_text()
        if self._task_needs_time and self._current_time_text:
            current_time = self._current_time_text().strip()
            if current_time:
                self._task_environment_text = "\n".join(
                    value for value in (self._task_environment_text, current_time) if value
                )
        self._task_notes_text = (
            "" if self._task_is_direct_answer else self._current_notes_text(
                self._latest_user_text(history),
            )
        )
        self._task_skill_instructions = (
            self._skill_registry.get_active_instructions()
            if self._skill_registry is not None
            else ""
        )
        self._task_injection = "\n\n".join(filter(None, (
            self._prompt_injector.build_task_injection() or "",
            task_mode_instruction(self._task_mode),
        )))

    def _clear_task_prompt_snapshot(self, task_snapshot_version: int) -> None:
        if task_snapshot_version != self._task_snapshot_version:
            return
        self._task_environment_text = None
        self._task_notes_text = None
        self._task_skill_instructions = None
        self._task_injection = None
        self._task_tool_defs = None
        self._task_mode = TaskMode.DIRECT
        self._task_mode_source = "rule"
        self._task_mode_confidence = None
        self._task_is_direct_answer = False
        self._task_needs_time = False
        self._previous_request_cache_shape = None

    async def _refresh_task_mode_after_steering(
        self, history: ConversationHistory,
    ) -> None:
        """Apply an explicit mid-task capability escalation or restriction."""
        route = await self._route_task_mode(history.get_messages())
        candidate = route.mode
        if self._plan_only and candidate is TaskMode.MODIFY:
            candidate = TaskMode.INSPECT
        # A direct-answer steering message such as “also explain why” should
        # not discard the active workspace context. Explicit read-only wording
        # is classified as INSPECT and still restricts an active modify task.
        if candidate is TaskMode.DIRECT or candidate is self._task_mode:
            return

        self._task_mode = candidate
        self._task_mode_source = route.source
        self._task_mode_confidence = route.confidence
        self._task_is_direct_answer = False
        self._task_tool_defs = self._build_tool_defs(task_mode=candidate)
        if not self._task_notes_text:
            self._task_notes_text = self._current_notes_text(
                self._latest_user_text(history),
            )
        self._task_injection = "\n\n".join(filter(None, (
            self._prompt_injector.build_task_injection() or "",
            task_mode_instruction(candidate),
        )))

    async def _route_task_mode(
        self, messages: list[Message],
    ) -> TaskModeRouteResult:
        if self._task_mode_router is None:
            return TaskModeRouteResult(
                mode=classify_task_mode(messages),
                source="rule",
                rule_decisive=True,
            )
        trace_scope = (
            self._trace_recorder.span(
                "task_mode_routing",
                "routing",
                {"message_count": len(messages)},
            )
            if self._trace_recorder is not None
            else nullcontext(None)
        )
        with trace_scope as trace_span:
            route = await self._task_mode_router.route(messages)
            if trace_span is not None:
                trace_span.finish(
                    "ok" if not route.error else "fallback",
                    {
                        "mode": route.mode.value,
                        "source": route.source,
                        "rule_decisive": route.rule_decisive,
                        "confidence": route.confidence,
                        "probabilities": route.probabilities,
                        "model_requests": route.model_requests,
                        "error": route.error,
                    },
                )
        self.turn_model_requests += route.model_requests
        self.turn_usage = self.turn_usage + route.usage
        self.turn_cache_usage = self.turn_cache_usage + route.cache_usage
        self.cache_hit = self.turn_cache_usage.read_tokens > 0
        return route

    def _apply_task_mode_route(self, route: TaskModeRouteResult) -> None:
        self._task_mode = route.mode
        self._task_mode_source = route.source
        self._task_mode_confidence = route.confidence

    @staticmethod
    def _cache_fingerprint(value: object) -> str:
        serialized = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _cache_shape_attributes(
        self,
        messages: list,
        tools: list[dict] | None,
        system_blocks: list[dict] | None,
    ) -> dict[str, object]:
        """Describe request-shape changes without recording prompt content."""
        message_parts = [self._cache_fingerprint(message) for message in messages]
        current = {
            "system": self._cache_fingerprint(system_blocks or []),
            "tools": self._cache_fingerprint(tools or []),
            "messages": message_parts,
        }
        previous = self._previous_request_cache_shape
        first_changed: int | None = 0
        common_prefix = 0
        if previous is not None:
            previous_messages = previous.get("messages", [])
            if not isinstance(previous_messages, list):
                previous_messages = []
            for before, after in zip(previous_messages, message_parts):
                if before != after:
                    break
                common_prefix += 1
            if common_prefix == len(previous_messages) and common_prefix == len(message_parts):
                first_changed = None
            else:
                first_changed = common_prefix

        self._previous_request_cache_shape = current
        return {
            "prompt_fingerprint": self._cache_fingerprint(current),
            "system_fingerprint": current["system"],
            "tools_fingerprint": current["tools"],
            "message_fingerprint": self._cache_fingerprint(message_parts),
            "first_changed_message_index": first_changed,
            "common_message_prefix_count": common_prefix,
            "system_changed": previous is not None and previous.get("system") != current["system"],
            "tools_changed": previous is not None and previous.get("tools") != current["tools"],
        }

    # -- internals ------------------------------------------------------------

    async def _stream_provider(
        self,
        messages: list,
        tools: list[dict] | None,
        system_blocks: list[dict] | None,
    ) -> AsyncIterator[str | ToolCall]:
        """Apply first-event/idle timeouts and retry before any output."""
        attempt = 0
        while True:
            emitted = False
            first_token_ms: float | None = None
            # Count actual provider attempts, including retries that fail before
            # producing their first event.
            self.turn_model_requests += 1
            request_number = self.turn_model_requests
            request_started = monotonic()
            cache_shape = self._cache_shape_attributes(messages, tools, system_blocks)
            token_budget = (
                self._request_token_budget_attributes(messages, tools, system_blocks)
                if self._trace_recorder is not None
                else {}
            )
            trace_scope = (
                self._trace_recorder.span(
                    f"request #{request_number}",
                    "model_request",
                    {
                        "round": self._active_round,
                        "request": request_number,
                        "retry": attempt,
                        "message_count": len(messages),
                        "tool_schema_count": len(tools or []),
                        "model": self._provider.config.model,
                        "task_mode": self._task_mode.value,
                        "task_mode_source": self._task_mode_source,
                        "task_mode_confidence": self._task_mode_confidence,
                        **token_budget,
                        **cache_shape,
                    },
                    # This scope intentionally survives async-generator
                    # yields. Cancellation may finalize the generator from a
                    # different Context, so it must not own a ContextVar token.
                    activate=False,
                )
                if self._trace_recorder is not None
                else nullcontext(None)
            )
            with trace_scope as trace_span:
                self._provider.begin_request()
                stream = self._provider.chat_stream(
                    messages=messages,
                    tools=tools,
                    system_blocks=system_blocks,
                ).__aiter__()
                try:
                    while True:
                        timeout = (
                            self._idle_event_timeout
                            if emitted
                            else self._first_event_timeout
                        )
                        try:
                            item = await asyncio.wait_for(stream.__anext__(), timeout)
                        except StopAsyncIteration:
                            raw_usage = getattr(self._provider, "last_usage", None)
                            usage = TokenUsage.from_raw(
                                raw_usage
                            )
                            cache_usage = CacheUsage.from_raw(raw_usage)
                            if trace_span is not None:
                                trace_span.finish("ok", {
                                    "round": self._active_round,
                                    "request": request_number,
                                    "retry": attempt,
                                    "first_token_ms": first_token_ms,
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens,
                                    "total_tokens": usage.total_tokens,
                                    "usage_available": usage.available,
                                    "cache_read_tokens": cache_usage.read_tokens,
                                    "cache_write_tokens": cache_usage.write_tokens,
                                    "cache_miss_tokens": cache_usage.miss_tokens,
                                    "cache_usage_available": cache_usage.available,
                                    "cache_hit": cache_usage.read_tokens > 0,
                                })
                            return
                        if not emitted:
                            first_token_ms = max(
                                0.0, (monotonic() - request_started) * 1_000,
                            )
                            if trace_span is not None:
                                trace_span.event("model_first_token", {
                                    "round": self._active_round,
                                    "request": request_number,
                                    "first_token_ms": first_token_ms,
                                })
                        emitted = True
                        yield item
                except asyncio.CancelledError:
                    await self._close_provider_stream(stream)
                    if trace_span is not None:
                        trace_span.finish("cancelled", {
                            "round": self._active_round,
                            "request": request_number,
                            "first_token_ms": first_token_ms,
                        })
                    raise

                except Exception as exc:
                    await self._close_provider_stream(stream)
                    error_text = str(exc).strip() or repr(exc)
                    retryable = (
                        isinstance(
                            exc,
                            (
                                asyncio.TimeoutError,
                                httpx.RequestError,
                                ConnectionError,
                                OSError,
                            ),
                        )
                        or isinstance(exc, ProviderError) and exc.retryable
                    )
                    will_retry = (
                        not emitted and retryable and attempt < self._provider_retries
                    )
                    if trace_span is not None:
                        trace_span.finish("retry" if will_retry else "error", {
                            "round": self._active_round,
                            "request": request_number,
                            "retry": attempt,
                            "first_token_ms": first_token_ms,
                            "error_type": type(exc).__name__,
                            "error": error_text,
                            "retryable": retryable,
                        })
                    if will_retry:
                        attempt += 1
                        if self._trace_recorder is not None:
                            self._trace_recorder.record("retry", status="retry", attributes={
                                "round": self._active_round,
                                "request": request_number,
                                "next_attempt": attempt,
                                "error_type": type(exc).__name__,
                                "error": error_text,
                            })
                        if self._retry_delay:
                            await asyncio.sleep(self._retry_delay * attempt)
                        continue

                    if isinstance(exc, asyncio.TimeoutError):
                        stage = "首个响应" if not emitted else "流式响应"
                        timeout = (
                            self._first_event_timeout
                            if not emitted
                            else self._idle_event_timeout
                        )
                        raise TimeoutError(
                            f"模型{stage}超时（{timeout:g} 秒）"
                        ) from exc
                    raise

    @staticmethod
    def _request_token_budget_attributes(
        messages: list[Message],
        tools: list[dict] | None,
        system_blocks: list[dict] | None,
    ) -> dict[str, int]:
        """Estimate input-token contributors for local cost diagnosis.

        Provider usage remains the billing source of truth.  These stable,
        tokenizer-independent estimates only identify which prompt component
        is worth optimizing without recording prompt payloads in Trace.
        """
        components = {
            "system": 0,
            "instructions": 0,
            "skills": 0,
            "notes": 0,
            "environment": 0,
            "conversation": 0,
            "tool_schema": 0,
        }

        def estimate_message(message: Message) -> int:
            return StructuredSummarizer._estimate_tokens([message])

        for block in system_blocks or []:
            if isinstance(block, dict):
                components["system"] += estimate_message({
                    "role": "system", "content": block.get("text", ""),
                })

        labels = {
            "[Instructions]": "instructions",
            "[Activated Skills]": "skills",
            "[Notes]": "notes",
            "[Environment]": "environment",
        }
        for message in messages:
            content = message.get("content")
            target = (
                next(
                    (name for prefix, name in labels.items() if content.startswith(prefix)),
                    None,
                )
                if isinstance(content, str) else None
            )
            if target is not None:
                components[target] += estimate_message(message)
            elif message.get("role") == "system":
                components["system"] += estimate_message(message)
            else:
                components["conversation"] += estimate_message(message)

        if tools:
            components["tool_schema"] = StructuredSummarizer._estimate_tokens([{
                "role": "system",
                "content": json.dumps(tools, ensure_ascii=False),
            }])

        return {
            "estimated_input_tokens": sum(components.values()),
            **{
                f"estimated_{name}_tokens": value
                for name, value in components.items()
            },
        }

    @staticmethod
    async def _close_provider_stream(stream) -> None:
        close = getattr(stream, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except (RuntimeError, asyncio.CancelledError):
            pass

    def _build_tool_defs(
        self,
        *,
        include_tool_result_tools: bool = True,
        task_mode: TaskMode = TaskMode.MODIFY,
    ) -> list[dict]:
        """Return a stable tool schema for every tool-enabled request.

        ``tool_result_read`` and ``tool_result_search`` validate their storage
        at execution time.  Keeping their schemas present avoids invalidating
        the provider cache the first time a large tool result is persisted.
        ``include_tool_result_tools`` remains accepted for compatibility.
        """
        del include_tool_result_tools
        definitions = self._context_assembler.tool_definitions(self._tool_registry)
        if task_mode is TaskMode.MODIFY and not self._plan_only:
            return definitions
        allowed_names = {
            tool.name
            for tool in self._tool_registry.list_tools()
            if self._tool_allowed_in_mode(tool.name, task_mode)
        }
        return [
            definition
            for definition in definitions
            if self._tool_definition_name(definition) in allowed_names
        ]

    @classmethod
    def _tool_definition_names(cls, definitions: list[dict] | None) -> set[str]:
        return {
            cls._tool_definition_name(definition)
            for definition in definitions or []
        }

    @staticmethod
    def _tool_definition_name(definition: dict) -> str:
        function = definition.get("function")
        if isinstance(function, dict):
            name = function.get("name", "")
        else:
            name = definition.get("name", "")
        return name if isinstance(name, str) else ""

    def _partition_tools(
        self, tool_calls: list[ToolCall],
    ) -> tuple[list[ToolCall], list[ToolCall]]:
        reads: list[ToolCall] = []
        writes: list[ToolCall] = []
        for tc in tool_calls:
            tool = self._tool_registry.get(tc.name)
            if tool is not None and tool.category == ToolCategory.READ:
                reads.append(tc)
            else:
                writes.append(tc)
        return reads, writes

    def _tool_allowed_in_mode(self, tool_name: str, mode: TaskMode) -> bool:
        if self._plan_only:
            return tool_name in _PLAN_MODE_ALLOWED
        if mode is TaskMode.MODIFY:
            return True
        if mode is TaskMode.DIRECT:
            return False
        if tool_name == "request_user_input":
            return True
        tool = self._tool_registry.get(tool_name)
        return tool is not None and tool.available_in_inspect

    def _tool_allowed_for_task(self, tc: ToolCall) -> bool:
        if not self._tool_allowed_in_mode(tc.name, self._task_mode):
            return False
        if self._task_mode is TaskMode.MODIFY and not self._plan_only:
            return True
        if tc.name == "request_user_input":
            return True
        tool = self._tool_registry.get(tc.name)
        return tool is not None and not tool.may_modify(tc.input)

    def _validate_tool_call_identity(self, tc: ToolCall) -> str | None:
        if not isinstance(tc.id, str) or not tc.id:
            return "ToolCall.id 必须是非空字符串"
        if not isinstance(tc.name, str) or not tc.name:
            return "ToolCall.name 必须是非空字符串"
        return None

    def _validate_tool_call_input(self, tc: ToolCall) -> ToolResult | None:
        if isinstance(tc.input, dict):
            return None
        return ToolResult(
            success=False,
            content="",
            error="工具调用参数必须是对象",
        )

    def _repair_unpaired_tool_calls(
        self,
        history: ConversationHistory,
        start_index: int,
    ) -> None:
        """Keep protocol history valid without pretending side effects rolled back."""
        pending: dict[str, str] = {}
        messages = history.get_messages()[start_index:]
        for message in messages:
            if message.get("role") == "assistant":
                tool_calls = message.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        call_id = call.get("id")
                        function = call.get("function", {})
                        name = function.get("name", "") if isinstance(function, dict) else ""
                        if isinstance(call_id, str) and call_id:
                            pending[call_id] = str(name)
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        call_id = block.get("id")
                        if isinstance(call_id, str) and call_id:
                            pending[call_id] = str(block.get("name", ""))
            elif message.get("role") == "tool":
                pending.pop(str(message.get("tool_call_id", "")), None)
            elif message.get("role") == "user" and isinstance(message.get("content"), list):
                for block in message["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        pending.pop(str(block.get("tool_use_id", "")), None)

        recovery = (
            "[运行中断] 此工具调用的最终状态未知；磁盘或外部系统可能已经发生变化。"
            "继续前必须先检查实际状态，不要盲目重试。"
        )
        for call_id, name in pending.items():
            try:
                history.add_raw_message(
                    self._provider.make_tool_result_message(call_id, name, recovery)
                )
            except Exception:
                # Never mask the original provider/runtime error.
                return

    async def _execute_concurrent(self, tool_calls: list[ToolCall]) -> list:
        async def _one(tc: ToolCall):
            return await self._execute_tool_with_hooks(tc)
        return await asyncio.gather(*[_one(tc) for tc in tool_calls])

    async def _execute_tool_with_hooks(self, tc: ToolCall) -> ToolResult:
        trace_attributes = (
            self._trace_recorder.tool_attributes(tc.name, tc.input)
            if self._trace_recorder is not None
            else {}
        )
        trace_attributes.update({
            "round": self._active_round,
            "call_id": tc.id,
        })
        trace_scope = (
            self._trace_recorder.span(tc.name, "tool", trace_attributes)
            if self._trace_recorder is not None
            else nullcontext(None)
        )
        with trace_scope as trace_span:
            tool = self._tool_registry.get(tc.name)
            may_modify = tool is None or tool.may_modify(tc.input)
            if self._recovery_store is not None and self._recovery_task_id:
                try:
                    self._recovery_store.record_tool_intent(
                        self._recovery_task_id,
                        call_id=tc.id,
                        tool_name=tc.name,
                        arguments=tc.input,
                        round_number=self._active_round,
                        may_modify=may_modify,
                    )
                except Exception as exc:
                    # A write without a durable intent cannot be reconciled
                    # safely after a power loss, so fail closed for write tools.
                    if may_modify:
                        return ToolResult(
                            success=False,
                            content="",
                            error=(
                                "工具恢复日志写入失败，已阻止可能产生副作用的执行: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )

            intercept_reason: str | None = None
            if self._hook_engine:
                intercept_reason = await self._hook_engine.fire(
                    HookEvent.TOOL_PRE_EXEC,
                    {"tool_name": tc.name, "params": tc.input},
                )

            if intercept_reason:
                result = ToolResult(success=False, content="", error=intercept_reason)
            else:
                if tool is None:
                    result = ToolResult(
                        success=False, content="", error=f"未知工具: {tc.name}",
                    )
                else:
                    result = await self._tool_executor.execute(tool, tc.input)

            if self._hook_engine:
                await self._hook_engine.fire(HookEvent.TOOL_POST_EXEC, {
                    "tool_name": tc.name, "params": tc.input,
                    "success": result.success,
                })

            if self._recovery_store is not None and self._recovery_task_id:
                self._recovery_store.record_tool_result(
                    self._recovery_task_id,
                    call_id=tc.id,
                    tool_name=tc.name,
                    success=result.success,
                    error=result.error,
                    content=result.content,
                )

            if trace_span is not None:
                trace_span.finish("ok" if result.success else "error", {
                    "round": self._active_round,
                    "call_id": tc.id,
                    "success": result.success,
                    "content_chars": len(result.content),
                    "error": result.error,
                })

        return result

    async def _block_tool(self, tc: ToolCall) -> ToolBlockedEvent:
        if self._plan_only:
            reason = (
                f"Plan-only 模式已开启，工具 '{tc.name}' 不在只读白名单中，已被拦截。"
                "请先关闭 plan-only 开关再执行修改操作。"
            )
        elif tc.name == "sub_agent":
            reason = (
                "当前任务处于 inspect 只读模式，但该 Subagent 调用包含写入能力，"
                "已被运行时拦截。请改用 background=true，或选择只开放读取工具的角色。"
            )
        else:
            reason = (
                f"当前任务处于 inspect 只读模式，工具 '{tc.name}' 可能产生副作用，"
                "已被运行时拦截。只有用户明确要求实施修改或执行命令后才能使用该工具。"
            )
        if self._trace_recorder is not None:
            self._trace_recorder.record("tool_blocked", status="blocked", attributes={
                "round": self._active_round,
                "tool": tc.name,
                "call_id": tc.id,
                "reason": reason,
            })
        return ToolBlockedEvent(
            tool_name=tc.name, reason=reason,
        )

    def _precheck_tool(self, tc: ToolCall) -> tuple[bool, str, "asyncio.Future | None"]:
        """Run security check. Returns ``(allowed, reason, hitl_future)``.

        If ``hitl_future`` is not None, the caller must yield a HITLRequestEvent
        and await the future before proceeding.
        """
        if self._security_guard is None:
            return True, "ok", None

        tool = self._tool_registry.get(tc.name)
        security_params = self._security_parameters(tc)
        read_only = None if tool is None else not tool.may_modify(tc.input)
        allowed, reason = self._security_guard.check(
            tc.name,
            security_params,
            read_only=read_only,
        )
        if allowed and reason == "ask":
            loop = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            return True, "ask", future
        return allowed, reason, None

    def _approval_parameters(self, tc: ToolCall) -> dict:
        """Let a tool disclose its effective capability envelope for HITL."""
        tool = self._tool_registry.get(tc.name)
        if tool is None:
            return dict(tc.input)
        try:
            details = tool.approval_parameters(tc.input)
        except Exception:
            # Approval rendering must never make an otherwise valid tool call
            # fail. Security evaluation and execution still use the original
            # provider arguments.
            return dict(tc.input)
        return details if isinstance(details, dict) else dict(tc.input)

    def _security_parameters(self, tc: ToolCall) -> dict:
        """Build rule-matching arguments without changing execution input."""
        tool = self._tool_registry.get(tc.name)
        if tool is None:
            return dict(tc.input)
        try:
            details = tool.security_parameters(tc.input)
        except Exception:
            return dict(tc.input)
        return details if isinstance(details, dict) else dict(tc.input)

    def record_round(self, user_msg: str, assistant_msg: str) -> None:
        """Record a completed round for auto-note purposes."""
        if self._note_manager:
            self._note_manager.record_round(user_msg, assistant_msg)

    async def update_notes_if_needed(self) -> int:
        """Trigger note update if interval reached. Returns number of files changed."""
        if self._note_manager and self._note_manager.should_update():
            results = await self._note_manager.update_all()
            return len(results)
        return 0

    def set_security_level(self, level: SecurityLevel) -> None:
        """Switch the global security level."""
        if self._security_guard:
            self._security_guard.set_level(level)

    def tool_may_modify_workspace(
        self, tool_name: str, params: dict | None = None,
    ) -> bool:
        """Whether a call needs a before/after workspace snapshot.

        Read tools are declared concurrency-safe and must not mutate the
        project. Unknown tools retain the conservative answer so a newly added
        extension cannot silently disappear from change reporting.
        """
        tool = self._tool_registry.get(tool_name)
        return tool is None or tool.may_modify(params or {})

    def set_workspace(self, workspace: Path) -> None:
        if self._security_guard:
            self._security_guard.set_project_root(workspace)
        if self._truncator:
            self._truncator.set_project_root(workspace)
            for name in _DEFERRED_TOOL_RESULT_TOOLS:
                tool = self._tool_registry.get(name)
                set_storage_dir = getattr(tool, "set_storage_dir", None)
                if callable(set_storage_dir):
                    set_storage_dir(self._truncator.storage_dir)

    def _current_environment_text(self) -> str:
        return self._context_assembler.environment_text()

    def _current_notes_text(self, query: str = "") -> str:
        if self._note_manager is None:
            return ""
        # Third-party note managers from earlier versions may not yet accept
        # a query.  Keep that extension point compatible.
        try:
            return self._note_manager.context_text(query=query)
        except TypeError:
            return self._note_manager.context_text()

    @staticmethod
    def _latest_user_text(history: ConversationHistory) -> str:
        for message in reversed(history.get_messages()):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    @classmethod
    def _task_needs_current_time(cls, history: ConversationHistory) -> bool:
        text = cls._latest_user_text(history).lower()
        if not text:
            return False
        markers = (
            "当前时间", "现在几点", "几点了", "今天几号", "今天日期", "北京时间",
            "current time", "what time", "today's date", "todays date",
        )
        return any(marker in text for marker in markers)

    def _append_tool_result(
        self, history: ConversationHistory, tool_call: ToolCall, result: ToolResult,
    ) -> None:
        tr_msg = self._provider.make_tool_result_message(
            tool_call.id, tool_call.name, result.to_message(),
        )
        history.add_raw_message(tr_msg)
