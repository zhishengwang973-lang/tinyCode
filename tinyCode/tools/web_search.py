"""Bounded public-web search with retry and source failover."""

from __future__ import annotations

import asyncio
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

from tinyCode.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult
from tinyCode.tools.validation import require_string
from tinyCode.tools.web_common import UNTRUSTED_WEB_PREFIX, fetch_public_url


MAX_QUERY_CHARS = 1_000
MAX_SEARCH_RESULTS = 10
_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
_BRAVE_URL = "https://search.brave.com/search"
# ToolExecutor has a 30-second deadline. Keep the complete failover path below
# it: 11s primary + 7s retry + 9s fallback + 0.25s backoff.
_PRIMARY_TIMEOUT_SECONDS = 11.0
_RETRY_TIMEOUT_SECONDS = 7.0
_FALLBACK_TIMEOUT_SECONDS = 9.0
_RETRY_DELAY_SECONDS = 0.25


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._collect: str | None = None
        self._collect_tag = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._finish_current()
            self._current = {"title": "", "url": attributes.get("href", ""), "snippet": ""}
            self._collect = "title"
            self._collect_tag = tag
        elif self._current is not None and "result__snippet" in classes:
            self._collect = "snippet"
            self._collect_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._collect is not None and tag == self._collect_tag:
            self._collect = None
            self._collect_tag = ""

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._collect is not None:
            self._current[self._collect] += data

    def close(self) -> None:
        super().close()
        self._finish_current()

    def _finish_current(self) -> None:
        if self._current is not None:
            self._current = {
                key: " ".join(value.split()) for key, value in self._current.items()
            }
            if self._current["title"] and self._current["url"]:
                self.results.append(self._current)
        self._current = None
        self._collect = None
        self._collect_tag = ""


class _BraveParser(HTMLParser):
    """Extract Brave's ordinary web-result title, URL, and snippet."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._anchor_stack: list[str] = []
        self._current: dict[str, str] | None = None
        self._collect: str | None = None
        self._collect_tag = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "a":
            self._anchor_stack.append(attributes.get("href", ""))
        classes = set(attributes.get("class", "").split())
        is_title = (
            "search-snippet-title" in classes
            or any(name.endswith("snippet-title") for name in classes)
        )
        if is_title and self._anchor_stack and self._anchor_stack[-1]:
            self._finish_current()
            self._current = {
                "title": "", "url": self._anchor_stack[-1], "snippet": "",
            }
            self._collect = "title"
            self._collect_tag = tag
        elif self._current is not None and (
            "snippet-description" in classes
            or any(name.endswith("snippet-description") for name in classes)
        ):
            self._collect = "snippet"
            self._collect_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._collect is not None and tag == self._collect_tag:
            self._collect = None
            self._collect_tag = ""
        if tag == "a" and self._anchor_stack:
            self._anchor_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._collect is not None:
            self._current[self._collect] += data

    def close(self) -> None:
        super().close()
        self._finish_current()

    def _finish_current(self) -> None:
        if self._current is not None:
            item = {
                key: " ".join(value.split())
                for key, value in self._current.items()
            }
            if item["title"] and item["url"]:
                self.results.append(item)
        self._current = None
        self._collect = None
        self._collect_tag = ""


def _direct_result_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlsplit(url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        encoded = parse_qs(parsed.query).get("uddg", [])
        if encoded:
            return unquote(encoded[0])
    return url


class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "搜索公网并返回标题、URL 和简短摘要；主搜索源不可用时会自动重试并切换备用源。"
            "需要最新资料、官方文档或尚未知道目标 URL 时使用；"
            "找到结果后应使用 web_fetch 核实关键内容。"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.READ

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("query", "string", "精简的搜索词。"),
            ToolParameter(
                "max_results", "integer", f"返回结果数，1–{MAX_SEARCH_RESULTS}。",
                required=False, default=5,
            ),
        ]

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            query = require_string(query, "query").strip()
        except ValueError as exc:
            return ToolResult(False, "", str(exc))
        if not query:
            return ToolResult(False, "", "query 不能为空")
        if len(query) > MAX_QUERY_CHARS:
            return ToolResult(False, "", f"query 不能超过 {MAX_QUERY_CHARS} 字符")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= MAX_SEARCH_RESULTS
        ):
            return ToolResult(False, "", f"max_results 必须是 1–{MAX_SEARCH_RESULTS} 的整数")

        duckduckgo_url = _DUCKDUCKGO_URL + "?" + urlencode({"q": query})
        brave_url = _BRAVE_URL + "?" + urlencode({"q": query, "source": "web"})
        attempts = (
            ("DuckDuckGo", duckduckgo_url, _DuckDuckGoParser, _PRIMARY_TIMEOUT_SECONDS),
            ("DuckDuckGo 重试", duckduckgo_url, _DuckDuckGoParser, _RETRY_TIMEOUT_SECONDS),
            ("Brave Search 备用源", brave_url, _BraveParser, _FALLBACK_TIMEOUT_SECONDS),
        )
        errors: list[str] = []
        for index, (source, url, parser_type, timeout) in enumerate(attempts):
            if index == 1:
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
            try:
                results = await self._search_source(
                    url, parser_type, max_results, timeout,
                )
            except Exception as exc:
                errors.append(f"{source}: {type(exc).__name__}: {exc}")
                continue
            source_note = f"搜索源: {source}\n"
            return ToolResult(
                True,
                f"{UNTRUSTED_WEB_PREFIX}\n{source_note}\n" + "\n\n".join(results),
            )
        return ToolResult(
            False,
            "",
            "web_search 所有搜索源均失败；" + "；".join(errors),
        )

    @staticmethod
    async def _search_source(
        url: str,
        parser_type: type[_DuckDuckGoParser] | type[_BraveParser],
        max_results: int,
        timeout: float,
    ) -> list[str]:
        # The source hosts are fixed by the implementation. DNS validation
        # remains enabled so compromised local resolution cannot redirect a
        # search to the local or private network.
        try:
            response = await asyncio.wait_for(
                fetch_public_url(url, max_bytes=400_000),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"请求超时（{timeout:g} 秒）") from exc
        parser = parser_type()
        parser.feed(response.body.decode("utf-8", errors="replace"))
        parser.close()
        results: list[str] = []
        for item in parser.results:
            direct_url = _direct_result_url(item["url"])
            parsed = urlsplit(direct_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            block = f"{len(results) + 1}. {item['title']}\n   URL: {direct_url}"
            if item["snippet"]:
                block += f"\n   摘要: {item['snippet']}"
            results.append(block)
            if len(results) >= max_results:
                break
        if not results:
            raise RuntimeError("搜索服务未返回可解析的结果")
        return results
