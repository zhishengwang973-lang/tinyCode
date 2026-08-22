"""Hard-coded dangerous command blacklist — always active regardless of mode."""

import re

# Commands that are unconditionally blocked
_BLOCKED_PREFIXES = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "sudo rm",
    "chmod 777",
    "chmod -R 777",
    "chown -R",
    "mkfs.",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "> /dev/sda",
]

_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"curl\s+.*\|\s*(ba)?sh"),       # curl | bash
    re.compile(r"wget\s+.*-O\s*-\s*\|\s*(ba)?sh"),  # wget | bash
    re.compile(r"eval\s+"),                       # eval
    re.compile(r"\bgit\s+push\s+(?:[^;&|]*\s)?(?:--force(?:-with-lease)?|-f)(?:\s|$)"),
    re.compile(r">\s*/dev/[hs]d[a-z]"),          # overwrite disk
    re.compile(r"\bmv\s+.*\s+/dev/null\b"),      # mv to /dev/null
    # Match destructive executables even when invoked through an absolute path
    # or hidden behind a shell wrapper such as `sh -c`.
    re.compile(r"(?:^|[;&|]\s*|\s)(?:/[^\s;&|]+/)?rm(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*|\s)(?:/[^\s;&|]+/)?(?:mkfs(?:\.[^\s]+)?|shutdown|reboot)(?:\s|$)"),
    re.compile(r"\b(?:ba|z|k)?sh\s+-c\b"),
]

# Always blocked — cannot be overridden by rules
HARD_BLOCKED = frozenset({
    "rm -rf /",
    "sudo rm -rf",
    "mkfs",
    "dd if=/dev/zero",
})


def check_blacklist(command: str) -> str | None:
    """Check a command against the blacklist.

    Returns:
        An error message string if blocked, ``None`` if safe.
    """
    cmd_stripped = command.strip()
    cmd_lower = cmd_stripped.lower()

    for prefix in _BLOCKED_PREFIXES:
        if cmd_lower.startswith(prefix.lower()):
            return f"命令被黑名单拦截（匹配: {prefix}）"

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(cmd_stripped):
            return f"命令被黑名单拦截（匹配模式: {pattern.pattern}）"

    return None
