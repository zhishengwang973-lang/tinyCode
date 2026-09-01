"""Command-line interface for reproducible, model-separated evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tinyCode.config.loader import ConfigError, load_provider_config
from tinyCode.evals.loader import EvalConfigError, load_case
from tinyCode.evals.runner import EvalRunner
from tinyCode.time_utils import beijing_filename_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyCode eval",
        description="Run a TinyCode evaluation with separate executor and judge models.",
    )
    parser.add_argument("target", type=Path, help="评测用例 YAML 文件或用例目录")
    parser.add_argument("--executor", required=True, help="配置中的执行 Provider 名")
    parser.add_argument("--judge", required=True, help="配置中的评测 Provider 名")
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"), help="报告目录")
    parser.add_argument("--keep-workspace", action="store_true", help="保留隔离工作区以便排查")
    parser.add_argument("--tag", action="append", default=[], help="只运行带该标签的用例；可重复")
    return parser


async def run_from_args(args: argparse.Namespace) -> int:
    try:
        executor = load_provider_config(args.executor)
        judge = load_provider_config(args.judge)
        runner = EvalRunner(executor, judge)
    except (EvalConfigError, ConfigError, ValueError) as exc:
        print(f"评测配置错误: {exc}", file=sys.stderr)
        return 2

    try:
        cases = _load_cases(args.target, set(args.tag))
    except EvalConfigError as exc:
        print(f"评测配置错误: {exc}", file=sys.stderr)
        return 2
    if not cases:
        print("评测配置错误: 未找到匹配的 YAML 用例", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = beijing_filename_timestamp()
    reports = []
    progress = _TerminalProgress(sys.stdout, len(cases))
    for index, case in enumerate(cases, start=1):
        progress.start(index, case.name)
        report = await runner.run(
            case,
            keep_workspace=args.keep_workspace,
            progress=progress.update,
        )
        progress.clear()
        path = output_dir / f"{_safe_name(case.name)}_{stamp}.json"
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append((report, path))
        print(_summary(report, path))

    if len(reports) > 1:
        summary_path = output_dir / f"summary_{stamp}.json"
        summary_path.write_text(json.dumps({
            "executor": f"{executor.name}/{executor.model}",
            "judge": f"{judge.name}/{judge.model}",
            "cases": [
                {"name": report.case, "score": report.score, "status": report.status,
                 "report": str(path)}
                for report, path in reports
            ],
            "average_score": round(sum(report.score for report, _ in reports) / len(reports), 2),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"汇总报告：{summary_path}")
    return 0 if all(
        report.score >= 70 and report.status == "no_tool_call"
        for report, _ in reports
    ) else 1


def entry_point(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = asyncio.run(run_from_args(args))
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:80]


def _load_cases(target: Path, tags: set[str]):
    resolved = target.resolve()
    if resolved.is_file():
        paths = [resolved]
    elif resolved.is_dir():
        paths = [
            path
            for path in (
                sorted(resolved.rglob("*.yaml")) + sorted(resolved.rglob("*.yml"))
            )
            if "fixtures" not in path.relative_to(resolved).parts
        ]
    else:
        raise EvalConfigError(f"用例路径不存在: {target}")
    cases = [load_case(path) for path in paths]
    return [case for case in cases if not tags or tags.intersection(case.tags)]


class _TerminalProgress:
    """One-line terminal progress that also remains readable when piped."""

    def __init__(self, stream, total: int) -> None:
        self._stream = stream
        self._total = max(1, total)
        self._case_index = 0
        self._case_name = ""
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._last_line = ""

    def start(self, case_index: int, case_name: str) -> None:
        self._case_index = case_index
        self._case_name = case_name
        self.update(0.0, "等待开始")

    def update(self, case_fraction: float, message: str) -> None:
        local = max(0.0, min(1.0, case_fraction))
        overall = ((self._case_index - 1) + local) / self._total
        percent = int(overall * 100)
        width = 20
        filled = int(overall * width)
        bar = "█" * filled + "░" * (width - filled)
        line = (
            f"评测 [{bar}] {percent:3d}% · "
            f"{self._case_index}/{self._total} · {self._case_name} · {message}"
        )
        if line == self._last_line:
            return
        self._last_line = line
        if self._interactive:
            self._stream.write("\r" + line + "\x1b[K")
        else:
            self._stream.write(line + "\n")
        self._stream.flush()

    def clear(self) -> None:
        if self._interactive and self._last_line:
            self._stream.write("\r\x1b[K")
            self._stream.flush()
        self._last_line = ""


def _summary(report, path: Path) -> str:
    checks = sum(check.passed for check in report.checks)
    return (
        f"评测完成：{report.case}\n"
        f"总分 {report.score:.2f}/100（确定性 {report.deterministic_score:.2f}/70，"
        f"模型评测 {report.judge_score:.2f}/30）\n"
        f"执行：{report.executor}\n"
        f"评测：{report.judge}\n"
        f"状态：{report.status} · 断言 {checks}/{len(report.checks)} · "
        f"轮次 {report.rounds} · 模型请求 {report.model_requests} · Token {report.tokens:,}\n"
        f"报告：{path}"
        + (f"\n保留的工作区：{report.workspace_path}" if report.workspace_path else "")
        + (f"\n评测模型异常：{report.judge_result.error}" if report.judge_result.error else "")
    )
