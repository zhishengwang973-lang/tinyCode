"""Project-wide Beijing time helpers.

All persisted and user-visible wall-clock timestamps use an explicit UTC+8
offset so they remain unambiguous when TinyCode runs on machines configured
with a different local timezone. Duration measurements continue to use a
monotonic clock and do not belong in this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def beijing_now() -> datetime:
    """Return the current timezone-aware Beijing datetime."""
    return datetime.now(BEIJING_TIMEZONE)


def beijing_now_iso() -> str:
    """Return an ISO 8601 timestamp carrying the explicit ``+08:00`` offset."""
    return beijing_now().isoformat()


def beijing_filename_timestamp(*, microseconds: bool = False) -> str:
    """Return a sortable, filesystem-safe Beijing timestamp."""
    pattern = "%Y%m%d_%H%M%S_%f" if microseconds else "%Y%m%d_%H%M%S"
    return beijing_now().strftime(pattern)


def to_beijing(value: str | datetime) -> datetime | None:
    """Parse a timestamp and convert it to Beijing time.

    Old TinyCode data may contain naive ISO timestamps. Those values were
    historically written as UTC, so keep that interpretation during migration.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(BEIJING_TIMEZONE)


def format_beijing_time(
    value: str | datetime,
    *,
    pattern: str = "%Y-%m-%d %H:%M:%S",
    default: str = "—",
    label: bool = True,
) -> str:
    """Format current or legacy timestamps consistently for the user."""
    parsed = to_beijing(value)
    if parsed is None:
        return default
    suffix = " 北京时间" if label else ""
    return parsed.strftime(pattern) + suffix


def timestamp_sort_key(value: object) -> float:
    """Return a timezone-safe key for sorting mixed legacy timestamps."""
    parsed = to_beijing(value) if isinstance(value, (str, datetime)) else None
    return parsed.timestamp() if parsed is not None else float("-inf")
