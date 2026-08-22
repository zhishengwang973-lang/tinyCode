"""Condition evaluator — exact, reverse, regex, glob with ALL/ANY logic."""

import fnmatch
import re
from typing import Any

from tinyCode.hooks.models import Condition, ConditionRule, MatchMode, Operator


class ConditionEvaluator:
    """Evaluate a Condition against a context dict."""

    def evaluate(self, condition: Condition | None, context: dict[str, Any]) -> bool:
        """Return True if *condition* matches *context*.

        - ``None`` condition → always True (unconditional trigger).
        - ``ALL`` → every rule must match.
        - ``ANY`` → at least one rule must match.
        """
        if condition is None or not condition.rules:
            return True

        results = [self._eval_rule(r, context) for r in condition.rules]

        if condition.match == MatchMode.ALL:
            return all(results)
        return any(results)

    def _eval_rule(self, rule: ConditionRule, context: dict[str, Any]) -> bool:
        actual = self._resolve_field(rule.field, context)
        target = rule.value

        if rule.operator == Operator.EXACT:
            return str(actual) == target
        elif rule.operator == Operator.NOT:
            return str(actual) != target
        elif rule.operator == Operator.GLOB:
            return fnmatch.fnmatch(str(actual), target)
        elif rule.operator == Operator.REGEX:
            try:
                return bool(re.search(target, str(actual)))
            except re.error:
                return False
        return False

    def _resolve_field(self, field: str, context: dict[str, Any]) -> Any:
        """Resolve dot-path field against nested context dicts."""
        parts = field.split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, "")
            else:
                return ""
        return current if current is not None else ""
