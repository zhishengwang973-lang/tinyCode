import unittest
import asyncio
from unittest.mock import AsyncMock, patch

from tinyCode.tools.request_user_input import RequestUserInputTool
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.web_common import WebResponse, validate_public_url
from tinyCode.tools.web_common import fetch_public_url
from tinyCode.tools.web_fetch import WebFetchTool
from tinyCode.tools.web_search import WebSearchTool, _direct_result_url


class RequestUserInputToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_selected_user_answer(self):
        handler = AsyncMock(return_value="PostgreSQL")
        tool = RequestUserInputTool()
        tool.set_handler(handler)

        result = await tool.execute(
            "选择数据库", ["PostgreSQL", "SQLite"],
        )

        self.assertTrue(result.success)
        self.assertEqual("用户回答: PostgreSQL", result.content)
        handler.assert_awaited_once_with("选择数据库", ["PostgreSQL", "SQLite"])

    async def test_rejects_invalid_options_before_prompting(self):
        handler = AsyncMock()
        tool = RequestUserInputTool()
        tool.set_handler(handler)

        result = await tool.execute("choose", ["only-one"])

        self.assertFalse(result.success)
        self.assertIn("2–8", result.error)
        handler.assert_not_awaited()

    async def test_reports_noninteractive_environment(self):
        result = await RequestUserInputTool().execute("question", [])

        self.assertFalse(result.success)
        self.assertIn("不支持交互", result.error)

    async def test_user_input_is_not_killed_by_ordinary_tool_timeout(self):
        async def delayed_answer(question, options):
            await asyncio.sleep(0.02)
            return "answer"

        tool = RequestUserInputTool()
        tool.set_handler(delayed_answer)

        result = await ToolExecutor(default_timeout=0.001).execute(tool, {
            "question": "question",
            "options": [],
        })

        self.assertTrue(result.success, result.error)

    async def test_explicit_timeout_still_applies_to_user_input(self):
        async def delayed_answer(question, options):
            await asyncio.sleep(0.02)
            return "answer"

        tool = RequestUserInputTool()
        tool.set_handler(delayed_answer)

        result = await ToolExecutor(default_timeout=1).execute(
            tool,
            {"question": "question", "options": []},
            timeout=0.001,
        )

        self.assertFalse(result.success)
        self.assertIn("执行超时", result.error)

    async def test_rejects_oversized_answer(self):
        tool = RequestUserInputTool()
        tool.set_handler(AsyncMock(return_value="x" * 20_001))

        result = await tool.execute("question", [])

        self.assertFalse(result.success)
        self.assertIn("20000", result.error)


class WebToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_url_guard_has_bounded_dns_lookup(self):
        async def timeout_wait_for(awaitable, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        with patch(
            "tinyCode.tools.web_common.asyncio.wait_for",
            side_effect=timeout_wait_for,
        ):
            with self.assertRaisesRegex(TimeoutError, "DNS 解析超时"):
                await validate_public_url("https://example.com/docs")

    async def test_public_url_guard_rejects_local_targets_and_credentials(self):
        for url in (
            "http://127.0.0.1/",
            "http://[::1]/",
            "http://localhost/",
            "https://user:pass@example.com/",
            "https://example.com:8443/",
            "file:///tmp/data",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    await validate_public_url(url)

    async def test_web_fetch_marks_page_as_untrusted_and_extracts_text(self):
        response = WebResponse(
            url="https://example.com/docs",
            status_code=200,
            content_type="text/html",
            body=b"<html><script>ignore()</script><h1>Official Docs</h1><p>Hello</p></html>",
            truncated=False,
        )
        with patch(
            "tinyCode.tools.web_fetch.fetch_public_url",
            AsyncMock(return_value=response),
        ):
            result = await WebFetchTool().execute("https://example.com/docs")

        self.assertTrue(result.success)
        self.assertIn("外部网页内容", result.content)
        self.assertIn("Official Docs Hello", result.content)
        self.assertNotIn("ignore()", result.content)

    async def test_fetch_revalidates_redirect_target_before_following(self):
        class FakeResponse:
            status_code = 302
            headers = {"location": "http://127.0.0.1/private"}
            url = "https://example.com/start"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def aiter_bytes(self):
                if False:
                    yield b""

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                return FakeResponse()

        validator = AsyncMock(side_effect=[
            "https://example.com/start",
            ValueError("拒绝访问本机或内网主机"),
        ])
        with patch("tinyCode.tools.web_common.httpx.AsyncClient", FakeClient), patch(
            "tinyCode.tools.web_common.validate_public_url", validator,
        ):
            with self.assertRaisesRegex(ValueError, "本机"):
                await fetch_public_url("https://example.com/start")

        self.assertEqual(2, validator.await_count)

    async def test_web_search_parses_and_limits_results(self):
        html = b"""
        <div class="result">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.com%2Fa">First</a>
          <a class="result__snippet">First snippet</a>
        </div>
        <div class="result">
          <a class="result__a" href="https://example.org/b">Second</a>
          <a class="result__snippet">Second snippet</a>
        </div>
        """
        response = WebResponse(
            url="https://html.duckduckgo.com/html/?q=test",
            status_code=200,
            content_type="text/html",
            body=html,
            truncated=False,
        )
        with patch(
            "tinyCode.tools.web_search.fetch_public_url",
            AsyncMock(return_value=response),
        ):
            result = await WebSearchTool().execute("test", max_results=1)

        self.assertTrue(result.success, result.error)
        self.assertIn("First", result.content)
        self.assertIn("https://docs.example.com/a", result.content)
        self.assertNotIn("Second", result.content)

    async def test_web_search_retries_then_uses_brave_fallback(self):
        brave_html = b"""
        <div class="snippet" data-type="web">
          <a href="https://www.bosch.com.cn/careers/">
            <div class="title search-snippet-title">Join Bosch</div>
          </a>
          <div class="snippet-description">Official careers site</div>
        </div>
        """
        brave_response = WebResponse(
            url="https://search.brave.com/search?q=bosch",
            status_code=200,
            content_type="text/html",
            body=brave_html,
            truncated=False,
        )
        fetch = AsyncMock(side_effect=[
            asyncio.TimeoutError(),
            RuntimeError("temporary connection failure"),
            brave_response,
        ])
        with patch(
            "tinyCode.tools.web_search.fetch_public_url", fetch,
        ), patch(
            "tinyCode.tools.web_search.asyncio.sleep", AsyncMock(),
        ):
            result = await WebSearchTool().execute("Bosch careers", max_results=3)

        self.assertTrue(result.success, result.error)
        self.assertEqual(3, fetch.await_count)
        self.assertIn("Brave Search 备用源", result.content)
        self.assertIn("https://www.bosch.com.cn/careers/", result.content)
        self.assertIn("Official careers site", result.content)

    async def test_web_search_reports_every_failed_source(self):
        fetch = AsyncMock(side_effect=[
            RuntimeError("primary unavailable"),
            RuntimeError("retry unavailable"),
            RuntimeError("fallback unavailable"),
        ])
        with patch(
            "tinyCode.tools.web_search.fetch_public_url", fetch,
        ), patch(
            "tinyCode.tools.web_search.asyncio.sleep", AsyncMock(),
        ):
            result = await WebSearchTool().execute("test")

        self.assertFalse(result.success)
        self.assertIn("DuckDuckGo: RuntimeError: primary unavailable", result.error)
        self.assertIn("DuckDuckGo 重试", result.error)
        self.assertIn("Brave Search 备用源", result.error)

    def test_duckduckgo_redirect_is_unwrapped(self):
        self.assertEqual(
            "https://example.com/docs?a=1",
            _direct_result_url(
                "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fa%3D1"
            ),
        )


if __name__ == "__main__":
    unittest.main()
