import unittest

from tinyCode.prompts.builder import PromptBuilder


class OutputPromptTests(unittest.TestCase):
    def test_direct_code_contract_requests_one_implementation_only(self):
        prompt = PromptBuilder().build()

        self.assertIn("默认只给一个完整实现", prompt)

    def test_latest_user_request_must_replace_previous_answer(self):
        prompt = PromptBuilder().build()

        self.assertIn("每轮以最新一条用户消息为当前任务", prompt)
        self.assertIn("不要重放", prompt)
        self.assertIn("不要主动追加第二种写法", prompt)
        self.assertIn("代码之外最多补充一句", prompt)

    def test_blocking_user_question_must_use_interactive_tool(self):
        prompt = PromptBuilder().build()

        self.assertIn("等待用户回答的强制协议", prompt)
        self.assertIn("本轮必须调用 request_user_input", prompt)
        self.assertIn("禁止只在普通文本中输出问题", prompt)
        self.assertIn("普通文本会被视为任务已经完成", prompt)
        self.assertIn("安全权限确认由系统的 HITL 流程处理", prompt)

    def test_subagent_policy_defines_positive_and_negative_triggers(self):
        prompt = PromptBuilder().build()

        self.assertIn("### Subagent 委派策略", prompt)
        self.assertIn("两个及以上互不依赖的模块", prompt)
        self.assertIn("不要委派：通用问答或独立代码片段", prompt)
        self.assertIn("不要按文件机械拆分", prompt)
        self.assertIn("多个可写任务的文件范围不得重叠", prompt)
        self.assertIn("只有任务确实依赖当前对话中的大量上下文时才使用 fork", prompt)

    def test_direct_answer_prompt_omits_only_tool_operations_policy(self):
        prompt = PromptBuilder().build(include_tool_instructions=False)

        self.assertNotIn("## 工具使用", prompt)
        self.assertIn("## 安全边界", prompt)
        self.assertIn("## 输出风格", prompt)


if __name__ == "__main__":
    unittest.main()
