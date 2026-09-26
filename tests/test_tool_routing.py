import unittest

from tinyCode.agent.tool_routing import (
    TaskMode,
    classify_task_mode,
    should_enable_tools,
    task_mode_instruction,
)


class ToolRoutingTests(unittest.TestCase):
    def test_self_contained_requests_do_not_receive_workspace_tools(self):
        prompts = [
            "写一个快速排序",
            "输出一个快速排序算法",
            "给我一个归并排序的 Python 实现",
            "解释什么是二分查找",
            "give me a Python code snippet for quicksort",
            "write merge sort in Python",
            "show me quicksort",
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

    def test_multimodal_turn_is_classified_from_its_text_block(self):
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "根据这个截图修复当前项目的界面问题"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/screen.png"},
                },
            ],
        }

        self.assertIs(TaskMode.MODIFY, classify_task_mode([message]))

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

    def test_direct_mode_covers_self_contained_answers_and_generated_content(self):
        prompts = (
            "给我一个测试 prompt",
            "给出一段 Markdown 格式的文本",
            "apply_patch 和 write_file 有什么区别",
            "如何学习 Python",
            "写一封邮件通知团队延期",
            "写一个 Python 程序计算斐波那契",
            "什么是项目管理",
            "解释应用程序生命周期",
            "给我一个目录树示例",
            "给我一个 config.yaml 示例",
            "帮我写一段代码",
            "把上面的代码改成 TypeScript",
            "解释这段代码的时间复杂度：for i in range(n): pass",
            "帮我分析下面这段日志",
            "检查这段代码有没有 bug",
            "provide a regex example for an email address",
            "what is dependency injection",
            "read this paragraph and summarize it",
            "这个报错是什么意思",
        )
        self._assert_modes(TaskMode.DIRECT, prompts)

    def test_inspect_mode_covers_workspace_web_and_explicit_read_only_tasks(self):
        prompts = (
            "帮我总结这个项目",
            "看一下 mergesort.py 是干嘛的",
            "输出 app.py 当前的内容",
            "审查这个项目有哪些不合理设计",
            "只分析 app.py，不要修改",
            "告诉我怎么重构这个项目，先别动代码",
            "如何修改 app.py",
            "刚才修改了哪些文件",
            "检查测试为什么失败，不要执行命令",
            "这个项目是否需要重构",
            "芯原股份校招官网",
            "今天北京天气",
            "最新 Textual 版本是什么",
            "explain the latest sorting algorithm research",
            "review the repository changes without editing anything",
            "总结项目结构",
            "检查仓库",
            "项目有哪些问题",
            "当前程序为什么崩溃",
            "web_search 工具为什么搜不到",
            "最终回答一直没有输出",
            "任务总是莫名其妙崩溃",
            "斜杠命令接入了吗",
            "模型请求有没有设置超时",
        )
        self._assert_modes(TaskMode.INSPECT, prompts)

    def test_modify_mode_requires_actionable_workspace_intent(self):
        prompts = (
            "修改 app.py 里的排序实现",
            "运行测试并修复失败用例",
            "实现登录功能",
            "把按钮改成红色",
            "优化一下启动速度",
            "修复这个崩溃",
            "创建 config.yaml",
            "删除 old.py",
            "安装依赖并运行测试",
            "提交并推送主分支",
            "启动项目",
            "不要只分析，直接修复",
            "请按照方案重构当前项目",
            "go ahead and implement the authentication feature",
            "fix the failing tests and commit the changes",
            "补一下 DeepSeek 的配置",
            "调整当前配置",
            "统一项目里的时间处理",
            "验证当前项目测试是否通过",
            "我希望全局 YAML 能配置 security_level",
            "当前全屏 TUI 输入框需要支持多行输入",
            "详情应该展示在每个事件下面",
            "按钮能不能改成黑色圆形",
            "颜色对调一下",
            "统计区第二行和第一行对齐",
            "文件变更、统计和错误等系统信息使用框",
            "全量完成评测集用例",
            "第 2 点加下换行",
            "补1、4、5",
            "运行测试，但不要修改代码",
            "run the test suite without editing files",
        )
        self._assert_modes(TaskMode.MODIFY, prompts)

    def test_explicit_restrictions_override_mutating_words(self):
        cases = {
            "修复这个问题，但先不要修改代码，只分析原因": TaskMode.INSPECT,
            "分析如何优化当前项目，不要动文件": TaskMode.INSPECT,
            "不要修改任何东西，直接回答什么是缓存": TaskMode.DIRECT,
            "不用查看项目，直接给我一个快速排序算法": TaskMode.DIRECT,
            "do not modify app.py; only explain the bug": TaskMode.INSPECT,
            "不要只分析，直接修改 app.py": TaskMode.MODIFY,
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertIs(expected, classify_task_mode([
                    {"role": "user", "content": prompt},
                ]))

    def test_followups_inherit_or_escalate_the_previous_workspace_mode(self):
        inspect_history = [
            {"role": "user", "content": "只检查当前项目，不要修改"},
            {"role": "assistant", "content": "检查完成，未发现需要处理的问题。"},
        ]
        self.assertIs(TaskMode.INSPECT, classify_task_mode([
            *inspect_history, {"role": "user", "content": "继续"},
        ]))
        self.assertIs(TaskMode.MODIFY, classify_task_mode([
            *inspect_history, {"role": "user", "content": "按这个方案实现"},
        ]))
        self.assertIs(TaskMode.DIRECT, classify_task_mode([
            *inspect_history, {"role": "user", "content": "可以"},
        ]))
        proposal_history = [
            {"role": "user", "content": "只检查当前项目，不要修改"},
            {
                "role": "assistant",
                "content": "建议修改配置加载逻辑，我可以按这个方案实现。",
            },
        ]
        self.assertIs(TaskMode.MODIFY, classify_task_mode([
            *proposal_history, {"role": "user", "content": "可以"},
        ]))

        modify_history = [
            {"role": "user", "content": "修复当前项目"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]
        self.assertIs(TaskMode.MODIFY, classify_task_mode([
            *modify_history, {"role": "user", "content": "继续"},
        ]))
        self.assertIs(TaskMode.MODIFY, classify_task_mode([
            *modify_history, {"role": "user", "content": "还是没用"},
        ]))
        self.assertIs(TaskMode.DIRECT, classify_task_mode([
            *modify_history, {"role": "user", "content": "写一个归并排序"},
        ]))
        self.assertIs(TaskMode.DIRECT, classify_task_mode([
            *modify_history, {"role": "user", "content": "好的"},
        ]))

    def test_task_mode_instructions_match_runtime_permissions(self):
        self.assertEqual("", task_mode_instruction(TaskMode.DIRECT))
        self.assertIn("只读审查", task_mode_instruction(TaskMode.INSPECT))
        self.assertIn("禁止修改文件", task_mode_instruction(TaskMode.INSPECT))
        self.assertIn("实施后", task_mode_instruction(TaskMode.MODIFY))

    def _assert_modes(self, expected: TaskMode, prompts) -> None:
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIs(expected, classify_task_mode([
                    {"role": "user", "content": prompt},
                ]))


if __name__ == "__main__":
    unittest.main()
