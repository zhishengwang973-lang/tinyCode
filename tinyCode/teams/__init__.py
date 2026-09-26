"""Team system — multi-agent orchestration with shared tasks and messaging."""

from tinyCode.teams.models import TeamDef, MemberDef, TeamTask, TeamMessage, MemberStatus, TaskStatus
from tinyCode.teams.lead import LeadAgent
from tinyCode.teams.member import TeamMember
from tinyCode.teams.tasks import SharedTaskList
from tinyCode.teams.mailbox import Mailbox
from tinyCode.teams.registry import NameRegistry
from tinyCode.teams.merger import GitMerger
from tinyCode.teams.scheduler import DispatchScheduler
from tinyCode.teams.persistence import load_team_def, list_team_defs, get_team_dir
from tinyCode.teams.orchestrator import run_team
from tinyCode.teams.auto import AutoTeamService, TeamProposal, TeamRunResult

__all__ = [
    "TeamDef", "MemberDef", "TeamTask", "TeamMessage", "MemberStatus", "TaskStatus",
    "LeadAgent", "TeamMember", "SharedTaskList", "Mailbox", "NameRegistry",
    "GitMerger", "DispatchScheduler",
    "load_team_def", "list_team_defs", "get_team_dir", "run_team",
    "AutoTeamService", "TeamProposal", "TeamRunResult",
]
