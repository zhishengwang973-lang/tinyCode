"""Environment info — collects system context for the first system message."""

import os
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path


BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def collect_environment() -> str:
    """Return a short environment summary block."""
    cwd = Path.cwd()
    os_name = platform.system()  # "Windows", "Linux", "Darwin"
    os_version = platform.release()
    now = datetime.now(BEIJING_TIMEZONE).strftime(
        "%Y-%m-%d %H:%M 北京时间 (UTC+8)"
    )

    return (
        f"工作目录: {cwd}\n"
        f"操作系统: {os_name} {os_version}\n"
        f"当前时间: {now}"
    )
