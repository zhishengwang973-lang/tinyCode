"""Fetch one bounded public web page as model-readable text."""

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.web_common import (
    UNTRUSTED_WEB_PREFIX,
    fetch_public_url,
    response_to_text,
)


class WebFetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "读取一个公网 HTTP(S) 页面并返回有大小上限的文本。"
            "用于查看已知 URL 的官方文档或资料；不支持本机、内网、非 80/443 端口或二进制文件。"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter("url", "string", "要读取的完整 http:// 或 https:// URL。")]

    async def execute(self, url: str) -> ToolResult:
        try:
            response = await fetch_public_url(url)
            content = response_to_text(response)
        except Exception as exc:
            return ToolResult(False, "", f"web_fetch 失败: {type(exc).__name__}: {exc}")
        return ToolResult(
            True,
            f"{UNTRUSTED_WEB_PREFIX}\n最终 URL: {response.url}\n\n{content}",
        )
