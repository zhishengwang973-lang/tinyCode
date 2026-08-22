"""Command-line parsing for the TinyCode executable."""

import argparse
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class CLIOptions:
    mode: str | None = None
    resume: bool = False
    trust_project_config: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyCode",
        description="TinyCode terminal coding agent",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "normal", "permissive"),
        help="override the configured security level for this process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="re-enter the last active TinyCode worktree",
    )
    parser.add_argument(
        "--trust-project-config",
        action="store_true",
        help="allow project-local MCP and Hook executable configuration",
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> CLIOptions:
    namespace = build_parser().parse_args(argv)
    return CLIOptions(
        mode=namespace.mode,
        resume=namespace.resume,
        trust_project_config=namespace.trust_project_config,
    )
