import unittest

from tinyCode.teams.scheduler import DispatchScheduler


class DispatchSchedulerTests(unittest.TestCase):
    def test_dispatch_mode_keeps_allowed_anthropic_tools(self):
        scheduler = DispatchScheduler()
        scheduler.set_lock_1(True)
        scheduler.set_lock_2(True)
        tools = [
            {"name": "read_file", "description": "read"},
            {"name": "team_create_task", "description": "create"},
            {"name": "sub_agent", "description": "delegate"},
        ]

        filtered = scheduler.filter_tools(tools)

        self.assertEqual(["team_create_task", "sub_agent"], [tool["name"] for tool in filtered])

    def test_dispatch_mode_keeps_allowed_openai_tools(self):
        scheduler = DispatchScheduler()
        scheduler.set_lock_1(True)
        scheduler.set_lock_2(True)
        tools = [
            {"type": "function", "function": {"name": "run_command"}},
            {"type": "function", "function": {"name": "delete_file"}},
            {"type": "function", "function": {"name": "grep"}},
            {"type": "function", "function": {"name": "team_list_tasks"}},
        ]

        filtered = scheduler.filter_tools(tools)

        self.assertEqual(["team_list_tasks"], [tool["function"]["name"] for tool in filtered])


if __name__ == "__main__":
    unittest.main()
