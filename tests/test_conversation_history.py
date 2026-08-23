import unittest

from tinyCode.conversation.history import ConversationHistory


class ConversationHistoryTests(unittest.TestCase):
    def test_deferred_context_only_appears_after_safe_boundary_flush(self):
        history = ConversationHistory()
        history.add_raw_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1"}],
        })
        history.defer_user_message("background complete")

        self.assertEqual(1, len(history.get_messages()))
        self.assertEqual(1, history.flush_deferred())
        self.assertEqual(
            ["assistant", "user"],
            [message["role"] for message in history.get_messages()],
        )

    def test_steering_is_flushed_at_safe_boundary(self):
        history = ConversationHistory()
        history.add_assistant_message("working")
        history.queue_steering_message("先不要修改文件")

        self.assertEqual(1, history.steering_count)
        self.assertEqual(1, history.flush_steering())

        self.assertEqual(0, history.steering_count)
        self.assertEqual(
            ["assistant", "user"],
            [message["role"] for message in history.get_messages()],
        )

    def test_discarding_steering_preserves_background_context(self):
        history = ConversationHistory()
        history.defer_user_message("background complete")
        history.queue_steering_message("change direction")

        self.assertEqual(1, history.discard_steering_messages())
        self.assertEqual(1, history.flush_deferred())

        messages = history.get_messages()
        self.assertEqual("background complete", messages[0]["content"])

    def test_multiple_steering_messages_keep_order_in_one_user_message(self):
        history = ConversationHistory()
        history.queue_steering_message("先检查根因")
        history.queue_steering_message("不要修改测试")

        self.assertEqual(2, history.flush_steering())

        messages = history.get_messages()
        self.assertEqual(1, len(messages))
        content = messages[0]["content"]
        self.assertLess(content.index("先检查根因"), content.index("不要修改测试"))

    def test_add_raw_message_copies_input_message(self):
        history = ConversationHistory()
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "original",
                },
            ],
        }

        history.add_raw_message(message)
        message["content"][0]["content"] = "mutated"

        self.assertEqual("original", history.get_messages()[0]["content"][0]["content"])

    def test_get_messages_does_not_expose_internal_message_objects(self):
        history = ConversationHistory()
        history.add_raw_message({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "original",
                },
            ],
        })

        messages = history.get_messages()
        messages[0]["content"][0]["content"] = "mutated"

        self.assertEqual("original", history.get_messages()[0]["content"][0]["content"])

    def test_replace_messages_copies_input_messages(self):
        history = ConversationHistory()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "original",
                    },
                ],
            }
        ]

        history.replace_messages(messages)
        messages[0]["content"][0]["content"] = "mutated"

        self.assertEqual("original", history.get_messages()[0]["content"][0]["content"])


if __name__ == "__main__":
    unittest.main()
