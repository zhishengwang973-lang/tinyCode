"""Interactive user-input tool bound to the active TUI."""

from collections.abc import Awaitable, Callable
from typing import Any

from tinyCode.tools.base import BaseTool, ToolParameter, ToolResult
from tinyCode.tools.validation import require_string


MAX_QUESTION_CHARS = 4_000
MAX_OPTIONS = 8
MAX_OPTION_CHARS = 500
MAX_ANSWER_CHARS = 20_000

UserInputHandler = Callable[[str, list[str]], Awaitable[str | None]]


class RequestUserInputTool(BaseTool):
    """Pause the foreground turn and collect one answer from the user."""

    def __init__(self) -> None:
        self._handler: UserInputHandler | None = None

    @property
    def name(self) -> str:
        return "request_user_input"

    @property
    def description(self) -> str:
        return (
            "当缺少会显著改变实现结果的关键信息时，暂停当前任务并询问用户。"
            "不要用它询问可以从项目中查到的信息，也不要用于安全权限确认。"
            "options 可提供 2–8 个简短候选项；留空则允许自由输入。"
        )

    @property
    def timeout_exempt(self) -> bool:
        # A user decision is not a network/tool stall. The foreground turn
        # remains cancellable via TurnRuntime while the prompt is open.
        return True

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("question", "string", "要向用户询问的单个明确问题。"),
            ToolParameter(
                "options",
                "array",
                "可选的简短候选项列表；无候选项时传空列表。",
                required=False,
                default=[],
                item_type="string",
            ),
        ]

    def set_handler(self, handler: UserInputHandler | None) -> None:
        self._handler = handler

    async def execute(
        self, question: str, options: list[str] | None = None, **_: Any,
    ) -> ToolResult:
        try:
            question = require_string(question, "question").strip()
        except ValueError as exc:
            return ToolResult(False, "", str(exc))
        if not question:
            return ToolResult(False, "", "question 不能为空")
        if len(question) > MAX_QUESTION_CHARS:
            return ToolResult(False, "", f"question 不能超过 {MAX_QUESTION_CHARS} 字符")

        normalized: list[str] = []
        if options is not None:
            if not isinstance(options, list):
                return ToolResult(False, "", "options 必须是字符串列表")
            if options and not 2 <= len(options) <= MAX_OPTIONS:
                return ToolResult(False, "", f"options 非空时必须只包含 2–{MAX_OPTIONS} 项")
            for index, option in enumerate(options):
                if not isinstance(option, str) or not option.strip():
                    return ToolResult(False, "", f"options[{index}] 必须是非空字符串")
                clean = option.strip()
                if len(clean) > MAX_OPTION_CHARS:
                    return ToolResult(
                        False, "", f"options[{index}] 不能超过 {MAX_OPTION_CHARS} 字符",
                    )
                normalized.append(clean)
        if len(set(normalized)) != len(normalized):
            return ToolResult(False, "", "options 不能包含重复项")
        if self._handler is None:
            return ToolResult(False, "", "当前运行环境不支持交互式用户输入")

        try:
            answer = await self._handler(question, normalized)
        except Exception as exc:
            return ToolResult(
                False, "", f"获取用户输入失败: {type(exc).__name__}: {exc}",
            )
        if answer is None:
            return ToolResult(False, "", "用户取消了输入")
        answer = answer.strip()
        if not answer:
            return ToolResult(False, "", "用户未提供答案")
        if len(answer) > MAX_ANSWER_CHARS:
            return ToolResult(False, "", f"用户回答不能超过 {MAX_ANSWER_CHARS} 字符")
        return ToolResult(True, f"用户回答: {answer}")
