"""Environment info — collects system context for the first system message."""

import platform
from pathlib import Path

from tinyCode.time_utils import BEIJING_TIMEZONE, beijing_now



def collect_environment() -> str:
    """Return the cache-stable part of the environment summary.

    Time is intentionally excluded: a minute-level timestamp before the
    conversation turns otherwise-identical tasks into prompt-cache misses.
    ``collect_current_time`` is added only to tasks that explicitly need it.
    """
    cwd = Path.cwd()
    os_name = platform.system()  # "Windows", "Linux", "Darwin"
    os_version = platform.release()

    return (
        f"工作目录: {cwd}\n"
        f"操作系统: {os_name} {os_version}"
    )


def collect_current_time() -> str:
    """Return a Beijing-time fact for a task that explicitly requests it."""
    now = beijing_now().strftime(
        "%Y-%m-%d %H:%M 北京时间 (UTC+8)"
    )
    return f"当前时间: {now}"
