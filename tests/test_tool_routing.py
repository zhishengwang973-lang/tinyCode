import unittest

from tinyCode.agent.tool_routing import should_enable_tools


class ToolRoutingTests(unittest.TestCase):
    def test_self_contained_requests_do_not_receive_workspace_tools(self):
        prompts = [
            "写一个快速排序",
            "输出一个快速排序算法",
            "解释什么是二分查找",
            "give me a Python code snippet for quicksort",
            "write merge sort in Python",
            "show me quicksort",
            "explain the latest sorting algorithm research",
            "what is a user profile",
            "你好",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(should_enable_tools([
                    {"role": "user", "content": prompt},
                ]))

    def test_workspace_requests_receive_tools(self):
        prompts = [
            "帮我总结这个项目",
            "修改 app.py 里的排序实现",
            "排查当前目录为什么启动失败",
            "运行测试并修复失败用例",
            "review the repository changes",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertTrue(should_enable_tools([
                    {"role": "user", "content": prompt},
                ]))

    def test_continuation_reuses_existing_project_tool_context(self):
        messages = [
            {"role": "user", "content": "检查这个项目"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {"role": "user", "content": "继续"},
        ]

        self.assertTrue(should_enable_tools(messages))

    def test_new_direct_request_does_not_inherit_old_project_tools(self):
        messages = [
            {"role": "user", "content": "检查这个项目"},
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {"role": "user", "content": "写一个快速排序"},
        ]

        self.assertFalse(should_enable_tools(messages))

    def test_natural_implementation_requests_default_to_workspace(self):
        for prompt in (
            "实现登录功能", "把按钮改成红色", "优化一下启动速度",
            "write then explain", "try to write",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(should_enable_tools([
                    {"role": "user", "content": prompt},
                ]))

    def test_conceptual_runtime_complexity_question_stays_self_contained(self):
        self.assertFalse(should_enable_tools([
            {"role": "user", "content": "解释快速排序的运行时间复杂度"},
        ]))


if __name__ == "__main__":
    unittest.main()
