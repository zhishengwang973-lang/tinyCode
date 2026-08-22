"""Decide whether a user turn actually needs access to the local workspace."""

from __future__ import annotations

import re

from tinyCode.providers.base import Message


_CHINESE_WORKSPACE_TERMS = (
    "这个项目", "当前项目", "项目里", "项目中", "项目文件", "仓库", "代码库",
    "当前目录", "工作目录", "文件", "目录", "路径", "源码", "代码变更",
    "修改", "修复", "排查", "调试", "重构", "构建", "实现", "优化", "改成",
    "运行测试", "执行测试", "启动项目", "执行命令",
    "安装", "提交", "删除", "保存", "写入", "创建文件", "新增文件", "添加到",
    "这个应用", "当前应用",
)

_ENGLISH_WORKSPACE_RE = re.compile(
    r"\b(?:project|repo|repository|codebase|workspace|file|directory|path|"
    r"source\s+code|modify|change|fix|debug|refactor|implement|optimi[sz]e|build|"
    r"run\s+(?:tests?|the\s+project|the\s+app)|execute\s+(?:tests?|commands?)|"
    r"install|commit|delete|save|review|add\s+to)\b",
    re.IGNORECASE,
)

_SELF_CONTAINED_RE = re.compile(
    r"^\s*(?:请|麻烦)?(?:写|输出|给我)(?:一份|一个)?[^\n]{0,40}"
    r"(?:排序|查找|算法|代码片段|示例)"
    r"|^\s*(?:请|麻烦)?(?:解释|介绍|什么是|是什么意思)[^\n]{0,80}"
    r"|^\s*(?:(?:please|can you)\s+)?(?:write|show|give me|provide)\b"
    r"[^\n]{0,100}(?:algorithm|code snippet|example|concept|quick\s*sort|"
    r"merge\s*sort|heap\s*sort|binary\s+search)\b"
    r"|^\s*(?:(?:please|can you)\s+)?(?:explain|describe|what is)\b[^\n]{0,100}",
    re.IGNORECASE,
)

_SMALL_TALK = frozenset({"你好", "您好", "嗨", "hello", "hi", "hey", "谢谢", "thanks"})

_CONTINUATION_TERMS = frozenset({
    "继续", "接着", "继续做", "接着做", "开始吧", "照做", "执行吧",
    "continue", "go on", "proceed", "do it",
})

_FILE_REFERENCE_RE = re.compile(
    r"(?:^|[\s`'\"])(?:[^\s`'\"]+/)?[^\s`'\"]+\."
    r"(?:py|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|rb|php|swift|kt|kts|"
    r"md|txt|json|ya?ml|toml|ini|cfg|html|css|scss|sql|sh)(?:$|[\s`'\",，。])",
    re.IGNORECASE,
)


def should_enable_tools(messages: list[Message]) -> bool:
    """Enable workspace tools only for turns that express workspace intent.

    TinyCode previously exposed all tools for every prompt.  Tool-capable
    models then tended to inspect the repository even for self-contained
    requests such as "write a quicksort algorithm".  Direct answering is the
    safer and faster default; explicit workspace language opts into tools.
    """
    latest = _latest_user_text(messages)
    if not latest:
        return False

    normalized = " ".join(latest.casefold().split())
    if normalized in _SMALL_TALK:
        return False
    if any(term in normalized for term in _CHINESE_WORKSPACE_TERMS):
        return True
    if _ENGLISH_WORKSPACE_RE.search(latest):
        return True
    if _FILE_REFERENCE_RE.search(latest):
        return True
    if normalized in _CONTINUATION_TERMS:
        return _has_prior_tool_exchange(messages)
    if _SELF_CONTAINED_RE.search(latest):
        return False
    # TinyCode is a workspace coding agent.  Unknown actionable requests are
    # safer to route with tools available than to silently make implementation
    # impossible.  Only high-confidence self-contained questions opt out.
    return True


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _has_prior_tool_exchange(messages: list[Message]) -> bool:
    for message in messages[:-1]:
        if message.get("role") == "tool" or message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict)
            and block.get("type") in {"tool_use", "tool_result"}
            for block in content
        ):
            return True
    return False
