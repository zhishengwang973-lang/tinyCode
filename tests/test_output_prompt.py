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


if __name__ == "__main__":
    unittest.main()
