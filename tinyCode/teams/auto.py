"""Automatic Team routing, isolated execution, and review-branch lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tinyCode.config.models import TeamAutomationConfig
from tinyCode.teams.merger import GitMerger
from tinyCode.teams.models import MemberDef, TeamDef
from tinyCode.teams.orchestrator import run_team
from tinyCode.time_utils import beijing_now_iso


Progress = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True)
class ProposedMember:
    name: str
    role: str
    scope: str


@dataclass(frozen=True)
class TeamProposal:
    goal: str
    reason: str
    members: tuple[ProposedMember, ...]

    def render(self) -> str:
        lines = [
            "检测到适合并行处理的复杂任务，建议启用 Agent Team。",
            f"原因：{self.reason}",
            "执行方案：",
        ]
        lines.extend(
            f"  {index}. {member.name}（{member.role}）— {member.scope}"
            for index, member in enumerate(self.members, start=1)
        )
        lines.append("每个成员使用独立 worktree；结果先进入审核分支，不直接修改当前分支。")
        return "\n".join(lines)


@dataclass
class TeamRunResult:
    summary: str
    run_id: str = ""
    review_ready: bool = False
    applied: bool = False


@dataclass
class ReviewRecord:
    run_id: str
    goal: str
    status: str
    target_branch: str
    target_head: str
    integration_name: str = ""
    integration_branch: str = ""
    integration_path: str = ""
    member_names: list[str] = field(default_factory=list)
    member_branches: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=beijing_now_iso)
    summary: str = ""
    error: str = ""


class AutoTeamService:
    """Conservative task router plus durable review/apply workflow."""

    _EXPLICIT_TEAM = re.compile(
        r"(?:agent\s*team|multi[- ]?agent|多个\s*(?:agent|智能体)|"
        r"多智能体|团队协作|并行(?:处理|开发|实现|排查))",
        re.IGNORECASE,
    )
    _OPT_OUT = re.compile(
        r"(?:不要|不使用|禁用|别用).{0,8}(?:team|多智能体|多个\s*agent)|"
        r"(?:单\s*(?:agent|智能体)|single[- ]?agent)",
        re.IGNORECASE,
    )
    _COMPLEXITY = (
        "全量", "整个项目", "跨模块", "多模块", "架构", "重构", "迁移",
        "端到端", "完整实现", "系统性", "全面审查", "并发", "恢复机制",
    )
    _DELIVERABLES = (
        "测试", "文档", "验证", "审查", "实现", "修复", "优化", "重构",
        "benchmark", "评测", "集成",
    )
    _SIMPLE = re.compile(
        r"(?:输出|写一个|给我|解释|是什么|怎么用).{0,18}"
        r"(?:算法|函数|代码片段|示例|命令)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        config: TeamAutomationConfig,
        *,
        repo_root: Path,
        worktree_manager: Any,
        provider: Any,
        tool_registry: Any,
        tool_executor: Any,
        roles: dict[str, Any] | None = None,
        progress: Progress | None = None,
        before_run: Callable[[], Awaitable[None]] | None = None,
        after_run: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self.config = config
        self._root = repo_root.resolve()
        self._manager = worktree_manager
        self._provider = provider
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._roles = roles or {}
        self._progress = progress
        self._before_run = before_run
        self._after_run = after_run
        self._review_dir = self._root / ".tinyCode" / "team_reviews"

    def propose(self, text: str) -> TeamProposal | None:
        goal = text.strip()
        if not goal or self.config.mode == "single" or self._OPT_OUT.search(goal):
            return None
        explicit = bool(self._EXPLICIT_TEAM.search(goal))
        if self.config.mode == "auto" and not explicit:
            if self._SIMPLE.search(goal) or len(goal) < 20:
                return None
            complexity = sum(token in goal for token in self._COMPLEXITY)
            deliverables = sum(token in goal for token in self._DELIVERABLES)
            if complexity < 1 or deliverables < 2:
                return None

        candidates = (
            ProposedMember("implementer", "general", "核心实现与必要的代码修改"),
            ProposedMember("verifier", "general", "测试、回归验证与边界场景修复"),
            ProposedMember("reviewer", "explorer", "只读审查、风险识别与验收核对"),
            ProposedMember("integrator", "general", "跨模块集成与文档收尾"),
        )
        members = candidates[: self.config.max_members]
        reason = (
            "你明确要求了多 Agent/并行协作"
            if explicit or self.config.mode == "team"
            else "任务同时涉及多个模块和独立的实现、验证工作流"
        )
        return TeamProposal(goal=goal, reason=reason, members=tuple(members))

    async def preflight(self) -> tuple[bool, str]:
        """Check whether an isolated Team can start without changing state."""
        ok, _branch, _head, error = await self._safe_target()
        return ok, error

    async def run(self, proposal: TeamProposal) -> TeamRunResult:
        if self._before_run is not None:
            await self._before_run()
        try:
            return await self._run_isolated(proposal)
        finally:
            if self._after_run is not None:
                result = self._after_run()
                if asyncio.iscoroutine(result):
                    await result

    async def _run_isolated(self, proposal: TeamProposal) -> TeamRunResult:
        ok, target_branch, target_head, error = await self._safe_target()
        if not ok:
            return TeamRunResult(f"## Team 未启动\n\n{error}")

        run_id = uuid.uuid4().hex[:12]
        created_names: list[str] = []
        members: list[MemberDef] = []
        running_record: ReviewRecord | None = None
        try:
            await self._emit("正在创建 Team 隔离工作区")
            for member in proposal.members:
                name = f"auto/{run_id}/{member.name}"
                info, create_error = await self._manager.create(name, recover=False)
                if info is None:
                    raise RuntimeError(
                        f"创建 {member.name} worktree 失败: {create_error}"
                    )
                created_names.append(name)
                members.append(MemberDef(
                    name=member.name,
                    role=member.role,
                    worktree=name,
                    backend="coro",
                ))

            running_record = ReviewRecord(
                run_id=run_id,
                goal=proposal.goal,
                status="running",
                target_branch=target_branch,
                target_head=target_head,
                member_names=list(created_names),
                member_branches=[f"tinyCode/{name}" for name in created_names],
                summary="Team 正在隔离工作区中执行",
            )
            self._save(running_record)

            definition = TeamDef(
                name=f"auto-{run_id}",
                description="TinyCode 自动编排 Team",
                members=members,
                max_rounds_per_member=32,
                timeout_seconds=self.config.timeout_seconds,
                validation_commands=list(self.config.validation_commands),
                allow_llm_conflict_resolution=(
                    self.config.allow_llm_conflict_resolution
                ),
                merge_policy="review",
            )
            result = await run_team(
                definition.name,
                proposal.goal,
                provider=self._provider,
                tool_registry=self._tool_registry,
                tool_executor=self._tool_executor,
                repo_root=self._root,
                roles=self._roles,
                preapproved=True,
                progress=self._progress,
                definition=definition,
                state_dir=self._root / ".tinyCode" / "teams" / definition.name,
            )
            if self.config.merge_policy == "none":
                assert running_record is not None
                running_record.status = "preserved"
                running_record.summary = (
                    "成员分支已保留，配置 merge_policy=none，未创建审核分支：\n"
                    + "\n".join(
                        f"- {branch}" for branch in running_record.member_branches
                    )
                )
                self._save(running_record)
                return TeamRunResult(
                    result + "\n\n## 成员分支已保留\n\n" + running_record.summary,
                    run_id=run_id,
                )
            await self._emit("正在汇总成员变更到审核分支")
            review = await self._build_review(
                run_id,
                proposal.goal,
                target_branch,
                target_head,
                created_names,
                result,
            )
            if review.status != "ready":
                return TeamRunResult(
                    result + "\n\n## 审核分支创建失败\n\n" + review.error,
                    run_id=run_id,
                )
            summary = (
                result
                + "\n\n## 变更待审核\n\n"
                + review.summary
                + f"\n\n审核编号：`{run_id}`。当前分支尚未改变。"
            )
            if self.config.merge_policy == "auto":
                applied, message = await self.apply(run_id)
                return TeamRunResult(
                    summary + "\n\n" + message,
                    run_id=run_id,
                    review_ready=not applied,
                    applied=applied,
                )
            return TeamRunResult(summary, run_id=run_id, review_ready=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._remove_many(created_names)
            if running_record is not None:
                running_record.status = "failed"
                running_record.error = f"{type(exc).__name__}: {exc}"
                self._save(running_record)
            return TeamRunResult(
                "## Team 启动失败\n\n"
                f"{type(exc).__name__}: {exc}。已清理本次创建的隔离工作区。"
            )

    async def apply(self, run_id: str) -> tuple[bool, str]:
        record = self._load(run_id)
        if record is None:
            return False, f"审核记录 '{run_id}' 不存在"
        if record.status == "applied":
            return True, f"Team 变更 {run_id} 已应用，无需重复操作"
        if record.status != "ready" or not record.integration_branch:
            return False, f"审核记录 {run_id} 当前状态为 {record.status}，不可应用"
        ok, branch, head, error = await self._safe_target()
        if not ok:
            return False, error
        if branch != record.target_branch or head != record.target_head:
            return False, (
                "当前分支或 HEAD 已变化，无法安全快进应用；"
                f"期望 {record.target_branch}@{record.target_head[:8]}，"
                f"实际 {branch}@{head[:8]}"
            )
        code, _, error = await self._git(
            "merge", "--ff-only", record.integration_branch,
        )
        if code != 0:
            return False, f"应用审核分支失败: {error}"
        record.status = "applied"
        self._save(record)
        cleanup = ""
        if self.config.cleanup_after_apply:
            cleanup_errors = await self._remove_many(
                [record.integration_name, *record.member_names]
            )
            if cleanup_errors:
                cleanup = "；部分隔离工作区保留: " + "; ".join(cleanup_errors)
        return True, f"已将 Team 变更 {run_id} 安全应用到 {branch}{cleanup}"

    async def discard(self, run_id: str) -> tuple[bool, str]:
        record = self._load(run_id)
        if record is None:
            return False, f"审核记录 '{run_id}' 不存在"
        if record.status == "applied":
            return False, "变更已经应用；丢弃审核记录不会撤销主分支提交"
        errors = await self._remove_many(
            [record.integration_name, *record.member_names]
        )
        record.status = "discarded"
        record.error = "; ".join(errors)
        self._save(record)
        if errors:
            return False, "审核已标记丢弃，但部分工作区清理失败: " + record.error
        return True, f"已丢弃 Team 审核 {run_id} 并清理隔离工作区"

    def list_reviews(self) -> list[ReviewRecord]:
        if not self._review_dir.is_dir():
            return []
        records = [self._load(path.stem) for path in self._review_dir.glob("*.json")]
        return sorted(
            (record for record in records if record is not None),
            key=lambda item: item.created_at,
            reverse=True,
        )

    def show_review(self, run_id: str) -> str:
        record = self._load(run_id)
        if record is None:
            return f"审核记录 '{run_id}' 不存在"
        return (
            f"Team 审核 {record.run_id}\n"
            f"状态: {record.status}\n目标: {record.goal}\n"
            f"目标分支: {record.target_branch}@{record.target_head[:8]}\n"
            f"审核分支: {record.integration_branch or '未创建'}\n"
            f"{record.summary or record.error or '暂无摘要'}"
        )

    async def _build_review(
        self,
        run_id: str,
        goal: str,
        target_branch: str,
        target_head: str,
        member_names: list[str],
        team_summary: str,
    ) -> ReviewRecord:
        integration_name = f"review/{run_id}"
        info, error = await self._manager.create(integration_name, recover=False)
        branches = [f"tinyCode/{name}" for name in member_names]
        record = ReviewRecord(
            run_id=run_id,
            goal=goal,
            status="building",
            target_branch=target_branch,
            target_head=target_head,
            integration_name=integration_name,
            integration_branch=info.branch if info else "",
            integration_path=info.path if info else "",
            member_names=list(member_names),
            member_branches=branches,
            summary=team_summary[-4_000:],
        )
        if info is None:
            record.status = "failed"
            record.error = error
            self._save(record)
            return record

        merger = GitMerger(
            self._provider,
            Path(info.path),
            allow_llm_conflicts=self.config.allow_llm_conflict_resolution,
        )
        changed = 0
        for branch in branches:
            code, _, diff_error = await self._git(
                "diff", "--quiet", f"{target_head}..{branch}",
            )
            if code == 0:
                continue
            if code not in {0, 1}:
                record.status = "failed"
                record.error = f"无法检查 {branch} 的变更: {diff_error}"
                self._save(record)
                return record
            merged, merge_message = await merger.merge(branch, info.branch)
            if not merged:
                record.status = "failed"
                record.error = f"{branch}: {merge_message}"
                self._save(record)
                return record
            changed += 1

        if changed:
            valid, validation_message = await merger.validate_worktree(
                Path(info.path), list(self.config.validation_commands),
            )
            if not valid:
                record.status = "failed"
                record.error = f"合并后的审核分支验证失败: {validation_message}"
                self._save(record)
                return record
            check_code, _, check_error = await self._git(
                "-C", info.path, "diff", "--check", f"{target_head}..HEAD",
            )
            if check_code != 0:
                record.status = "failed"
                record.error = f"审核分支变更完整性检查失败: {check_error}"
                self._save(record)
                return record

        code, stat, stat_error = await self._git(
            "diff", "--stat", f"{target_head}..{info.branch}",
        )
        if code != 0:
            record.status = "failed"
            record.error = f"生成审核摘要失败: {stat_error}"
        elif changed == 0:
            record.status = "empty"
            record.error = "Team 没有产生可应用的文件变更"
        else:
            record.status = "ready"
            record.summary = stat.strip() or f"{changed} 个成员分支包含变更"
        self._save(record)
        return record

    async def _safe_target(self) -> tuple[bool, str, str, str]:
        code, branch, error = await self._git("branch", "--show-current")
        branch = branch.strip()
        if code != 0 or not branch:
            return False, "", "", error or "当前处于 detached HEAD"
        code, status, error = await self._git(
            "status", "--porcelain", "--untracked-files=all",
        )
        if code != 0:
            return False, "", "", f"无法检查当前工作区: {error}"
        runtime_prefixes = (
            ".tinyCode/worktrees/",
            ".tinyCode/team_reviews/",
            ".tinyCode/teams/",
        )
        relevant_status = []
        for line in status.splitlines():
            path = line[3:].strip().strip('"') if len(line) >= 4 else line
            if any(path.startswith(prefix) for prefix in runtime_prefixes):
                continue
            relevant_status.append(line)
        if relevant_status:
            return False, "", "", "当前工作区存在未提交修改，无法安全启动自动 Team"
        code, head, error = await self._git("rev-parse", "HEAD")
        if code != 0:
            return False, "", "", f"无法读取当前 HEAD: {error}"
        return True, branch, head.strip(), ""

    async def _remove_many(self, names: list[str]) -> list[str]:
        errors: list[str] = []
        for name in reversed([name for name in names if name]):
            ok, message = await self._manager.exit(name, force=True)
            if not ok and "不存在" not in message:
                errors.append(f"{name}: {message}")
        return errors

    async def _emit(self, message: str) -> None:
        if self._progress is None:
            return
        result = self._progress(message)
        if asyncio.iscoroutine(result):
            await result

    def _save(self, record: ReviewRecord) -> None:
        self._review_dir.mkdir(parents=True, exist_ok=True)
        target = self._review_dir / f"{record.run_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _load(self, run_id: str) -> ReviewRecord | None:
        if not re.fullmatch(r"[a-f0-9]{12}", run_id or ""):
            return None
        path = self._review_dir / f"{run_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ReviewRecord(**data) if isinstance(data, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return None

    async def _git(self, *args: str) -> tuple[int, str, str]:
        process: asyncio.subprocess.Process | None = None
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            process = await asyncio.create_subprocess_exec(
                "git", *args,
                cwd=str(self._root),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=60.0,
            )
            return (
                process.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                await process.wait()
            return -1, "", "git 命令执行超时"
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except Exception as exc:
            return -1, "", f"{type(exc).__name__}: {exc}"
