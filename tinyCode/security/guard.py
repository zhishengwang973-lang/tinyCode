"""SecurityGuard — orchestrates the full security check pipeline."""

import asyncio
import re
from typing import Any

from tinyCode.security.blacklist import check_blacklist
from tinyCode.security.models import (
    HITLDecision,
    RuleAction,
    SecurityLevel,
)
from tinyCode.security.policy import SecurityPolicy
from tinyCode.security.sandbox import PathSandbox
from tinyCode.security.sensitive_paths import (
    command_references_sensitive_path,
    is_sensitive_path,
)


class SecurityGuard:
    """Orchestrates blacklist → sandbox → policy → HITL for each tool call."""

    def __init__(
        self,
        policy: SecurityPolicy,
        sandbox: PathSandbox,
        level: SecurityLevel = SecurityLevel.NORMAL,
        interactive: bool = True,
        preapproved: bool = False,
    ) -> None:
        self.policy = policy
        self.sandbox = sandbox
        self.level = level
        self.interactive = interactive
        self.preapproved = preapproved

    # -- main pipeline --------------------------------------------------------

    def check(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> tuple[bool, str]:
        """Run the full security pipeline.

        Returns:
            ``(allowed, reason)`` — *allowed* is True if the tool call may
            proceed.  *reason* explains why it was blocked (or "ok").
        """
        # Extract paths and command from params for rule matching. apply_patch
        # can touch multiple files and every declared target must pass.
        paths = self._path_params(tool_name, params)
        path = paths[0] if len(paths) == 1 else None
        command = self._command_param(params)

        if tool_name == "apply_patch" and not isinstance(params.get("patch"), str):
            return False, "patch 参数必须是字符串"
        if tool_name == "apply_patch" and not paths:
            return False, "patch 中没有可校验的文件路径"
        if any(not isinstance(candidate, str) for candidate in paths):
            return False, "路径参数必须是字符串"
        if command is not None and not isinstance(command, str):
            return False, "命令参数必须是字符串"

        # Provider configuration may contain live API credentials.  Never put
        # it into model-visible tool results, even in permissive mode.
        if any(is_sensitive_path(candidate) for candidate in paths):
            return False, "拒绝读取或修改包含模型凭据的本地配置文件"
        if tool_name == "run_command" and command_references_sensitive_path(command):
            return False, "拒绝通过命令访问包含模型凭据的本地配置文件"

        # ---- 1. Blacklist (always active) ----
        if command and tool_name == "run_command":
            blocked = check_blacklist(command)
            if blocked:
                return False, blocked

        # ---- 2. Path sandbox ----
        if tool_name in {
            "read_file", "write_file", "edit_file", "apply_patch",
            "delete_file", "glob", "grep",
        }:
            for candidate in paths:
                safe, msg = self.sandbox.validate(candidate)
                if not safe:
                    return False, msg

        # ---- 3. Policy evaluation ----
        actions = [
            self.policy.evaluate(tool_name, path=candidate, command=command)
            for candidate in (paths or [None])
        ]
        action = (
            RuleAction.DENY if RuleAction.DENY in actions
            else RuleAction.ASK if RuleAction.ASK in actions
            else RuleAction.ALLOW
        )

        if action == RuleAction.ALLOW:
            return True, "ok"
        elif action == RuleAction.DENY:
            reason = f"安全策略拒绝: {tool_name}"
            if paths:
                reason += f" (paths={', '.join(str(item) for item in paths)})"
            if command:
                reason += f" (command={command})"
            return False, reason
        else:
            # ASK — handled by caller (AgentLoop via HITL)
            if not self.interactive:
                if self.preapproved:
                    return True, "ok"
                return False, f"非交互任务无法确认操作: {tool_name}"
            return True, "ask"

    def needs_hitl(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Check whether this tool call requires human-in-the-loop."""
        paths = self._path_params(tool_name, params)
        command = self._command_param(params)
        actions = [
            self.policy.evaluate(tool_name, path=path, command=command)
            for path in (paths or [None])
        ]
        return RuleAction.DENY not in actions and RuleAction.ASK in actions

    def build_hitl_prompt(self, tool_name: str, params: dict[str, Any]) -> str:
        if tool_name == "apply_patch":
            paths = self._path_params(tool_name, params)
            patch = params.get("patch", "")
            return self.policy.to_hitl_prompt(tool_name, {
                "paths": paths,
                "patch_chars": len(patch) if isinstance(patch, str) else 0,
            })
        return self.policy.to_hitl_prompt(tool_name, params)

    def apply_hitl(
        self,
        decision: HITLDecision,
        tool_name: str,
        params: dict[str, Any],
    ) -> None:
        """Apply the HITL decision (create session/permanent rules)."""
        paths = self._path_params(tool_name, params)
        command = self._command_param(params)
        self.policy.hitl_to_rules(
            decision, tool_name, paths=paths or [None], command=command,
        )

    def set_level(self, level: SecurityLevel) -> None:
        self.level = level
        self.policy.set_level(level)

    def set_project_root(self, project_root) -> None:
        self.sandbox.set_project_root(project_root)
        self.policy.set_project_root(project_root)

    def _path_params(self, tool_name: str, params: dict[str, Any]) -> list[Any]:
        if tool_name == "apply_patch":
            patch = params.get("patch")
            if not isinstance(patch, str):
                return []
            return [
                match.group(1).strip()
                for match in re.finditer(
                    r"^\*\*\* (?:Add|Update|Delete) File: (.+)$",
                    patch,
                    flags=re.MULTILINE,
                )
            ]
        path = params.get("path") or params.get("file_path")
        if tool_name == "glob":
            path = path or params.get("pattern")
        return [path] if path is not None else []

    def _command_param(self, params: dict[str, Any]) -> Any:
        return params.get("command") or params.get("cmd")
