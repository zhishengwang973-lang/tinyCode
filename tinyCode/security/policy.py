"""Security policy — evaluates rules with session > project > global priority."""

import fnmatch
from pathlib import Path
from typing import Any

import yaml

from tinyCode.security.command_classifier import is_read_only_command
from tinyCode.storage.journal import atomic_write_text
from tinyCode.security.models import (
    HITLDecision,
    RuleAction,
    RuleScope,
    SecurityLevel,
    SecurityRule,
)


# Default path globs allowed in STRICT mode (in addition to configured rules)
STRICT_DEFAULT_ALLOWED = [
    "*.py", "*.txt", "*.md", "*.yaml", "*.yml", "*.json", "*.toml", "*.cfg",
    "*.js", "*.ts", "*.tsx", "*.jsx", "*.css", "*.html",
    "src/**", "lib/**", "tests/**", "test/**", "docs/**",
]


class SecurityPolicy:
    """Evaluates security rules with three-tier priority."""

    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.NORMAL,
        project_root: Path | None = None,
    ) -> None:
        self.level = level
        self._project_root = (project_root or Path.cwd()).resolve()
        self._session_rules: list[SecurityRule] = []
        self._project_rules: list[SecurityRule] = []
        self._global_rules: list[SecurityRule] = []
        self.load_errors: list[str] = []

        # Load persistent rules
        self._load_project_rules()
        self._load_global_rules()

    # -- rule management -----------------------------------------------------

    def add_session_rule(self, rule: SecurityRule) -> None:
        rule.scope = RuleScope.SESSION
        self._session_rules.insert(0, rule)

    def add_permanent_rule(self, rule: SecurityRule) -> None:
        """Save a rule permanently to the project-level security file."""
        rule.scope = RuleScope.PROJECT
        self._project_rules.insert(0, rule)
        try:
            self._save_project_rules()
        except BaseException:
            # A failed persistence attempt must not leave an in-memory rule
            # that looks permanent for the rest of this process.
            if self._project_rules and self._project_rules[0] is rule:
                self._project_rules.pop(0)
            else:
                try:
                    self._project_rules.remove(rule)
                except ValueError:
                    pass
            raise

    def set_level(self, level: SecurityLevel) -> None:
        self.level = level

    def set_project_root(self, project_root: Path) -> None:
        """Move project-scoped rule persistence to the active workspace."""
        self._project_root = project_root.resolve()
        self._load_project_rules()

    # -- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        path: str | None = None,
        command: str | None = None,
    ) -> RuleAction:
        """Evaluate rules and return the effective action.

        Priority: session > project > global > mode default.
        """
        # 1. Check rules in priority order
        for rules in [self._session_rules, self._project_rules, self._global_rules]:
            for rule in rules:
                if rule.matches(tool_name, path=path, command=command):
                    return rule.action

        # 2. Mode-based default
        return self._mode_default(tool_name, path, command)

    def to_hitl_prompt(self, tool_name: str, params: dict[str, Any]) -> str:
        """Build the HITL prompt text."""
        args = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return (
            f"⚠ 安全确认: {tool_name}({args})\n"
            f"  当前模式: {self.level.value}\n"
            f"  [A]llow once  [S]ession allow  [P]ermanent allow  [D]eny\n"
            f"  直接按 A/S/P/D，或输入选项后按 Enter"
        )

    def hitl_to_rule(
        self, decision: HITLDecision, tool_name: str,
        path: str | None, command: str | None,
    ) -> SecurityRule | None:
        """Convert a HITL decision into a new security rule (if permanent)."""
        rules = self.hitl_to_rules(
            decision, tool_name, paths=[path], command=command,
        )
        return rules[0] if rules else None

    def hitl_to_rules(
        self,
        decision: HITLDecision,
        tool_name: str,
        *,
        paths: list[str | None],
        command: str | None,
    ) -> list[SecurityRule]:
        """Persist a multi-path approval as one atomic policy update."""
        if decision not in {
            HITLDecision.ALLOW_PERMANENT, HITLDecision.ALLOW_SESSION,
        }:
            return []
        scope = (
            RuleScope.PROJECT
            if decision == HITLDecision.ALLOW_PERMANENT
            else RuleScope.SESSION
        )
        rules = [
            SecurityRule(
                tool=tool_name,
                action=RuleAction.ALLOW,
                path_pattern=path,
                command_pattern=command,
                scope=scope,
            )
            for path in paths
        ]
        target = self._project_rules if scope == RuleScope.PROJECT else self._session_rules
        target[0:0] = rules
        if scope == RuleScope.PROJECT:
            try:
                self._save_project_rules()
            except BaseException:
                for rule in rules:
                    try:
                        target.remove(rule)
                    except ValueError:
                        pass
                raise
        return rules

    # -- internals -----------------------------------------------------------

    def _mode_default(
        self,
        tool_name: str,
        path: str | None,
        command: str | None,
    ) -> RuleAction:
        # Asking the foreground user is not a side effect and must not itself
        # trigger the separate security-approval prompt.
        if tool_name == "request_user_input":
            return RuleAction.ALLOW
        # Public web reads are allowed in normal mode; strict mode explicitly
        # asks because the query/URL is transmitted to an external service.
        if tool_name in {"web_search", "web_fetch"}:
            return (
                RuleAction.ASK
                if self.level == SecurityLevel.STRICT
                else RuleAction.ALLOW
            )

        # Determine if tool is read-only
        is_read = self._is_read_tool(tool_name, command)

        if self.level == SecurityLevel.STRICT:
            # A path allow-list is only a read boundary.  Treating it as a
            # blanket allow-list used to make STRICT less restrictive than
            # NORMAL for edit/delete operations on ordinary source files.
            if is_read and path and self._is_path_allowed(path):
                return RuleAction.ALLOW
            return RuleAction.ASK

        elif self.level == SecurityLevel.NORMAL:
            # Shell commands are never silently trusted merely because their
            # first executable looks read-only.  `cat`, `env`, and similar
            # commands can disclose data outside the workspace, and a shell is
            # not a filesystem sandbox.
            if is_read and tool_name != "run_command":
                return RuleAction.ALLOW
            # Write tools in normal mode: ask
            return RuleAction.ASK

        elif self.level == SecurityLevel.PERMISSIVE:
            return RuleAction.ALLOW

        return RuleAction.ASK

    def _is_read_tool(self, tool_name: str, command: str | None) -> bool:
        if tool_name in {
            "read_file", "glob", "grep", "web_search", "web_fetch",
            "request_user_input",
        }:
            return True
        return tool_name == "run_command" and is_read_only_command(command or "")

    def _is_path_allowed(self, path: str) -> bool:
        """Check path against strict-mode allowed globs."""
        for pattern in STRICT_DEFAULT_ALLOWED:
            if fnmatch.fnmatch(path, pattern):
                return True
        # Also check project/global rules for allow rules with path_pattern
        for rules in [self._project_rules, self._global_rules]:
            for rule in rules:
                if rule.action == RuleAction.ALLOW and rule.path_pattern:
                    if fnmatch.fnmatch(path, rule.path_pattern):
                        return True
        return False

    # -- persistence ---------------------------------------------------------

    def _project_config_path(self) -> Path:
        return self._project_root / ".tinyCode-security.yaml"

    def _global_config_path(self) -> Path:
        return Path.home() / ".tinyCode" / "security.yaml"

    def _load_project_rules(self) -> None:
        self._project_rules = self._load_rules_file(
            self._project_config_path(), RuleScope.PROJECT,
        )

    def _load_global_rules(self) -> None:
        self._global_rules = self._load_rules_file(
            self._global_config_path(), RuleScope.GLOBAL,
        )

    def _load_rules_file(
        self, path: Path, scope: RuleScope = RuleScope.PROJECT,
    ) -> list[SecurityRule]:
        if not path.exists():
            return []
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            self.load_errors.append(f"{path}: 读取或解析失败 — {exc}")
            return []
        if not isinstance(raw, dict):
            self.load_errors.append(f"{path}: 顶层必须是 mapping")
            return []
        if "rules" not in raw or not isinstance(raw["rules"], list):
            self.load_errors.append(f"{path}: 'rules' 必须是列表")
            return []
        rules: list[SecurityRule] = []
        for index, entry in enumerate(raw["rules"]):
            if not isinstance(entry, dict):
                self.load_errors.append(f"{path}: 规则 #{index} 必须是 mapping，已跳过")
                continue
            action_str = entry.get("action", "ask")
            try:
                action = RuleAction(action_str)
            except (TypeError, ValueError):
                self.load_errors.append(
                    f"{path}: 规则 #{index} 的 action 无效，已跳过"
                )
                continue
            tool = entry.get("tool", "*")
            path_pattern = entry.get("path_pattern")
            command_pattern = entry.get("command_pattern")
            if not isinstance(tool, str) or not tool:
                self.load_errors.append(
                    f"{path}: 规则 #{index} 的 tool 必须是非空字符串，已跳过"
                )
                continue
            if path_pattern is not None and not isinstance(path_pattern, str):
                self.load_errors.append(
                    f"{path}: 规则 #{index} 的 path_pattern 必须是字符串，已跳过"
                )
                continue
            if command_pattern is not None and not isinstance(command_pattern, str):
                self.load_errors.append(
                    f"{path}: 规则 #{index} 的 command_pattern 必须是字符串，已跳过"
                )
                continue
            rule = SecurityRule(
                tool=tool,
                action=action,
                path_pattern=path_pattern,
                command_pattern=command_pattern,
                scope=scope,
            )
            rules.append(rule)
        return rules

    def _save_project_rules(self) -> None:
        path = self._project_config_path()
        entries: list[dict] = []
        for rule in self._project_rules:
            entry: dict = {"tool": rule.tool, "action": rule.action.value}
            if rule.path_pattern:
                entry["path_pattern"] = rule.path_pattern
            if rule.command_pattern:
                entry["command_pattern"] = rule.command_pattern
            entries.append(entry)
        atomic_write_text(
            path,
            yaml.dump({"rules": entries}, allow_unicode=True, default_flow_style=False),
        )
