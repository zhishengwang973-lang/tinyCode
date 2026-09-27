import unittest

from tinyCode.agent.intent_constraints import (
    ModePolicy,
    PlanningPolicy,
    SubagentPolicy,
    TeamPolicy,
    extract_intent_constraints,
)
from tinyCode.agent.task_mode import TaskMode, classify_task_mode


class IntentConstraintTests(unittest.TestCase):
    def test_clause_scoped_team_and_subagent_constraints_do_not_downgrade_main_task(self):
        prompt = (
            "先不要使用 Agent Team；如存在可独立的只读分析任务，可以使用 Subagent。"
            "新增批处理模块并编写测试、文档和示例，实际运行测试并修复失败。"
        )

        constraints = extract_intent_constraints(prompt)

        self.assertIs(TeamPolicy.DENY, constraints.team)
        self.assertIs(SubagentPolicy.READ_ONLY_ALLOWED, constraints.subagent)
        self.assertIs(ModePolicy.AUTO, constraints.mode)
        self.assertFalse(constraints.uncertain)
        self.assertIs(TaskMode.MODIFY, classify_task_mode([
            {"role": "user", "content": prompt},
        ]))

    def test_explicit_global_read_only_remains_a_hard_inspection_constraint(self):
        constraints = extract_intent_constraints("不要修改项目，只分析导致测试失败的原因")

        self.assertIs(ModePolicy.INSPECT, constraints.mode)
        self.assertIs(TaskMode.INSPECT, classify_task_mode([
            {"role": "user", "content": "不要修改项目，只分析导致测试失败的原因"},
        ]))

    def test_conflicting_read_only_and_modify_is_marked_uncertain(self):
        constraints = extract_intent_constraints("不要修改文件，但请实现登录功能")

        self.assertTrue(constraints.uncertain)
        self.assertIs(ModePolicy.AUTO, constraints.mode)

    def test_planning_opt_out_is_independent_from_modify_intent(self):
        constraints = extract_intent_constraints("不要任务规划，直接重构当前项目并运行测试")

        self.assertIs(PlanningPolicy.DENY, constraints.planning)
        self.assertIs(ModePolicy.AUTO, constraints.mode)
