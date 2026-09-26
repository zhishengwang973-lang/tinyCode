"""Command system — registry, parser, dispatcher, built-in commands."""

from tinyCode.commands.types import CommandType, CommandMeta, UIControl
from tinyCode.commands.registry import CommandRegistry
from tinyCode.commands.parser import parse, is_command
from tinyCode.commands.dispatcher import CommandDispatcher

__all__ = [
    "CommandType", "CommandMeta", "UIControl",
    "CommandRegistry",
    "parse", "is_command",
    "CommandDispatcher",
    "register_builtins",
]


def register_builtins(
    registry: CommandRegistry,
    ui: UIControl,
    note_manager=None,
    skill_registry=None,
    task_manager=None,
    worktree_manager=None,
    team_runner=None,
    team_review_service=None,
    trace_recorder=None,
) -> None:
    """Register all built-in commands with the given registry."""
    from tinyCode.commands.builtin.help_cmd import create as _help
    from tinyCode.commands.builtin.compress_cmd import create as _compress
    from tinyCode.commands.builtin.clear_cmd import create as _clear
    from tinyCode.commands.builtin.mode_cmd import create as _mode
    from tinyCode.commands.builtin.session_cmd import create as _session
    from tinyCode.commands.builtin.memory_cmd import create as _memory
    from tinyCode.commands.builtin.permission_cmd import create as _permission
    from tinyCode.commands.builtin.status_cmd import create as _status
    from tinyCode.commands.builtin.config_cmd import create as _config
    from tinyCode.commands.builtin.exit_cmd import create as _exit
    from tinyCode.commands.builtin.cancel_cmd import create as _cancel
    from tinyCode.commands.builtin.prompt_cmd import create as _prompt
    from tinyCode.commands.builtin.review_cmd import create as _review
    from tinyCode.commands.builtin.skill_cmd import create as _skill
    from tinyCode.commands.builtin.tasks_cmd import create as _tasks
    from tinyCode.commands.builtin.worktree_cmd import create as _worktree
    from tinyCode.commands.builtin.team_cmd import create as _team
    from tinyCode.commands.builtin.trace_cmd import create as _trace
    from tinyCode.commands.builtin.image_cmd import create as _image

    registry.register(_help(registry))
    registry.register(_compress(ui))
    registry.register(_clear(ui))
    registry.register(_mode(ui))
    registry.register(_session(ui))
    registry.register(_permission(ui))
    registry.register(_status(ui))
    registry.register(_config(ui))
    registry.register(_exit(ui))
    registry.register(_cancel(ui))
    registry.register(_prompt(ui))
    registry.register(_review())
    registry.register(_image(ui))
    if skill_registry:
        registry.register(_skill(skill_registry, ui))
    if note_manager:
        registry.register(_memory(note_manager))
    if task_manager:
        registry.register(_tasks(task_manager))
    if worktree_manager:
        registry.register(_worktree(worktree_manager, workspace_changed=ui.workspace_changed))
    registry.register(_team(
        runner=team_runner,
        confirmer=ui.confirm_action,
        review_service=team_review_service,
    ))
    if trace_recorder:
        registry.register(_trace(trace_recorder, confirmer=ui.confirm_action))
