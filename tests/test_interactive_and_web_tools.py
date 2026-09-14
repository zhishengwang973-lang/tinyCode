import unittest
import asyncio
from unittest.mock import AsyncMock, patch

from tinyCode.tools.request_user_input import RequestUserInputTool
from tinyCode.tools.executor import ToolExecutor
from tinyCode.tools.web_common import WebResponse, validate_public_url
from tinyCode.tools.web_common import fetch_public_url
from tinyCode.tools.web_fetch import WebFetchTool
from tinyCode.tools.web_search import WebSearchTool, _direct_result_url
from tinyCode.network import ProxyRouteDecision


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
    async def test_web_fetch_bypasses_unavailable_local_proxy(self):
        captured: dict = {}

        class FakeResponse:
            status_code = 200
            headers = {"content-type": "text/plain"}
            url = "https://example.com/docs"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def aiter_bytes(self):
                yield b"ok"

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, method, url):
                return FakeResponse()

        route = ProxyRouteDecision(
            trust_env=False,
            proxy_url="http://127.0.0.1:12334",
            proxy_address="127.0.0.1:12334",
            bypassed_unavailable_proxy=True,
        )
        with patch(
            "tinyCode.tools.web_common.detect_proxy_route", return_value=route,
        ), patch(
            "tinyCode.tools.web_common.httpx.AsyncClient", FakeClient,
        ):
            response = await fetch_public_url(
                "https://example.com/docs", validate_dns=False,
            )

        self.assertEqual(b"ok", response.body)
        self.assertFalse(captured["trust_env"])

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

    async def test_web_search_prefers_baidu_and_limits_results(self):
        html = b"""
        <div class="result c-container">
          <h3 class="t c-title-en">
            <a href="/link?url=first" data-landurl="https://docs.example.com/a">First</a>
          </h3>
          <div class="c-abstract">First snippet</div>
        </div>
        <div class="result c-container">
          <h3 class="t">
            <a href="https://example.org/b">Second</a>
          </h3>
          <div class="c-abstract">Second snippet</div>
        </div>
        """
        response = WebResponse(
            url="https://www.baidu.com/s?wd=test&amp;ie=utf-8",
            status_code=200,
            content_type="text/html",
            body=html,
            truncated=False,
        )
        with patch(
            "tinyCode.tools.web_search.fetch_public_url",
            AsyncMock(return_value=response),
        ) as fetch_mock:
            result = await WebSearchTool().execute("test", max_results=1)

        self.assertTrue(result.success, result.error)
        self.assertIn("搜索源: 百度", result.content)
        self.assertIn("First", result.content)
        self.assertIn("https://docs.example.com/a", result.content)
        self.assertNotIn("Second", result.content)
        requested_url = fetch_mock.await_args.args[0]
        self.assertTrue(requested_url.startswith("https://www.baidu.com/s?"))
        self.assertIn("tn=baidurt", requested_url)
        self.assertIn("wd=test", requested_url)
        self.assertIn(
            "Mozilla/5.0",
            fetch_mock.await_args.kwargs["request_headers"]["User-Agent"],
        )

    async def test_baidu_verification_skips_retry_and_uses_fallback(self):
        blocked = WebResponse(
            url="https://wappass.baidu.com/static/captcha/tuxing.html",
            status_code=200,
            content_type="text/html",
            body="<title>百度安全验证</title>".encode(),
            truncated=False,
        )
        duckduckgo = WebResponse(
            url="https://html.duckduckgo.com/html/?q=test",
            status_code=200,
            content_type="text/html",
            body=b'<a class="result__a" href="https://example.com">Fallback</a>',
            truncated=False,
        )
        fetch = AsyncMock(side_effect=[blocked, duckduckgo])
        with patch(
            "tinyCode.tools.web_search.fetch_public_url", fetch,
        ), patch(
            "tinyCode.tools.web_search.asyncio.sleep", AsyncMock(),
        ) as sleep:
            result = await WebSearchTool().execute("test")

        self.assertTrue(result.success, result.error)
        self.assertEqual(2, fetch.await_count)
        sleep.assert_not_awaited()
        self.assertIn("DuckDuckGo 备用源", result.content)

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
            RuntimeError("duckduckgo unavailable"),
            brave_response,
        ])
        with patch(
            "tinyCode.tools.web_search.fetch_public_url", fetch,
        ), patch(
            "tinyCode.tools.web_search.asyncio.sleep", AsyncMock(),
        ):
            result = await WebSearchTool().execute("Bosch careers", max_results=3)

        self.assertTrue(result.success, result.error)
        self.assertEqual(4, fetch.await_count)
        self.assertIn("Brave Search 备用源", result.content)
        self.assertIn("https://www.bosch.com.cn/careers/", result.content)
        self.assertIn("Official careers site", result.content)

    async def test_web_search_reports_every_failed_source(self):
        fetch = AsyncMock(side_effect=[
            RuntimeError("primary unavailable"),
            RuntimeError("retry unavailable"),
            RuntimeError("duckduckgo unavailable"),
            RuntimeError("brave unavailable"),
        ])
        with patch(
            "tinyCode.tools.web_search.fetch_public_url", fetch,
        ), patch(
            "tinyCode.tools.web_search.asyncio.sleep", AsyncMock(),
        ):
            result = await WebSearchTool().execute("test")

        self.assertFalse(result.success)
        self.assertIn("百度: RuntimeError: primary unavailable", result.error)
        self.assertIn("百度重试", result.error)
        self.assertIn("DuckDuckGo 备用源", result.error)
        self.assertIn("Brave Search 备用源", result.error)

    def test_duckduckgo_redirect_is_unwrapped(self):
        self.assertEqual(
            "https://example.com/docs?a=1",
            _direct_result_url(
                "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fa%3D1"
            ),
        )

    def test_relative_baidu_redirect_is_made_absolute(self):
        self.assertEqual(
            "https://www.baidu.com/link?url=result",
            _direct_result_url("/link?url=result"),
        )


if __name__ == "__main__":
    unittest.main()
