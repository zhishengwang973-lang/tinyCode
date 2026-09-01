"""Worktree data models."""

from dataclasses import dataclass, field

from tinyCode.time_utils import beijing_now_iso


@dataclass
class WorktreeInfo:
    name: str                      # user-facing name (e.g. "feature-x")
    path: str                      # absolute filesystem path
    branch: str                    # git branch name
    head_commit: str = ""          # HEAD SHA
    is_active: bool = False        # currently the working directory
    has_changes: bool = False      # uncommitted modifications
    created_at: str = ""


@dataclass
class WorktreeSession:
    active_worktree: str = ""      # name of the active worktree ("" = main)
    original_cwd: str = ""         # cwd before entering worktree
    created_at: str = field(default_factory=beijing_now_iso)
