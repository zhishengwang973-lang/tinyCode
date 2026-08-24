"""Sub-agent runner — executes a sub-agent to completion (run-to-end mode)."""

from tinyCode.agent.loop import AgentLoop
from tinyCode.conversation.history import ConversationHistory
from tinyCode.providers.base import BaseProvider, create_provider
from tinyCode.prompts.builder import PromptBuilder
from tinyCode.prompts.injector import PromptInjector
from tinyCode.security import PathSandbox, SecurityGuard, SecurityLevel, SecurityPolicy
from tinyCode.subagent.filter import ToolFilter
from tinyCode.subagent.models import SubAgentRole, SubAgentTask
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.base import ToolCategory
from tinyCode.tools.context import get_workspace_root
from tinyCode.tools.registry import ToolRegistry
from tinyCode.tracing.recorder import TraceRecorder

_FORK_INSTRUCTION = """\
[Fork 模式] 你是一个子工作器。遵守以下规则：
- 不要再创建子工作器（sub_agent 不可用）
- 不要主动对话、不要请求确认、不要问用户问题
- 直接使用工具完成任务，不需要征求许可
- 完成后输出结构化报告，控制在 500 字以内
- 报告格式：## 结果摘要 / ## 关键发现 / ## 文件与代码 / ## 建议"""


class SubAgentRunner:
    """Run a sub-agent to completion in a single call."""

    def __init__(
        self,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        roles: dict[str, SubAgentRole],
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._roles = roles
        self._trace_recorder = trace_recorder

    async def run(self, task: SubAgentTask, parent_history: ConversationHistory) -> str:
        """Execute a sub-agent task to completion.

        Returns the final result text.
        """
        role = self._roles.get(task.role) if task.role else None
        is_fork = role is None
        provider = self._provider
        if role is not None and role.model:
            provider = create_provider(
                self._provider.config.model_copy(update={"model": role.model})
            )

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
            # Fork: inherit parent history
            sub_history.replace_messages(parent_history.get_messages())
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
        sub_loop = AgentLoop(
            provider=provider,
            tool_registry=filtered_registry,
            tool_executor=self._tool_executor,
            prompt_builder=prompt_builder,
            prompt_injector=prompt_injector,
            security_guard=security_guard,
            environment_text=f"cwd: {workspace}",
            max_rounds=role.max_rounds if role else 3,
            trace_recorder=self._trace_recorder,
        )

        final_text = ""
        round_count = 0

        try:
            async for event in sub_loop.run(sub_history):
                from tinyCode.agent.events import AgentDoneEvent, ErrorEvent, RoundStartEvent, TextDeltaEvent
                if isinstance(event, TextDeltaEvent):
                    final_text += event.text
                elif isinstance(event, ErrorEvent):
                    task.fail(event.message)
                    raise RuntimeError(event.message)
                elif isinstance(event, AgentDoneEvent):
                    break
                elif isinstance(event, RoundStartEvent):
                    round_count += 1

            if not final_text.strip():
                error = "子 Agent 未返回结果"
                task.fail(error)
                raise RuntimeError(error)

            task.complete(
                final_text,
                tokens=sub_loop.turn_usage.total_tokens,
                rounds=round_count,
            )
            return final_text
        finally:
            if provider is not self._provider:
                await provider.close()
