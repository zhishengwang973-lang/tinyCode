"""Render tolerant JSONL traces as terminal text or standalone HTML."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _Span:
    span_id: str
    parent_span_id: str
    name: str
    kind: str
    started_ms: float
    duration_ms: float = 0.0
    status: str = "interrupted"
    attributes: dict[str, Any] = field(default_factory=dict)


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(row, dict) and isinstance(row.get("event"), str):
                rows.append(row)
    except OSError:
        return []
    return rows


def _spans(rows: list[dict[str, Any]]) -> list[_Span]:
    by_id: dict[str, _Span] = {}
    for row in rows:
        span_id = row.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            continue
        attributes = row.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        if row["event"] == "span_start":
            by_id[span_id] = _Span(
                span_id=span_id,
                parent_span_id=str(row.get("parent_span_id", "")),
                name=str(row.get("name", "operation")),
                kind=str(row.get("kind", "operation")),
                started_ms=_number(row.get("elapsed_ms")),
                attributes=dict(attributes),
            )
        elif row["event"] == "span_end":
            span = by_id.get(span_id)
            if span is None:
                span = _Span(
                    span_id=span_id,
                    parent_span_id=str(row.get("parent_span_id", "")),
                    name=str(row.get("name", "operation")),
                    kind=str(row.get("kind", "operation")),
                    started_ms=max(
                        0.0,
                        _number(row.get("elapsed_ms")) - _number(row.get("duration_ms")),
                    ),
                )
                by_id[span_id] = span
            span.duration_ms = _number(row.get("duration_ms"))
            span.status = str(row.get("status", "ok"))
            span.attributes.update(attributes)
    if rows:
        trace_end = max(_number(row.get("elapsed_ms")) for row in rows)
        for span in by_id.values():
            if span.status == "interrupted":
                span.duration_ms = max(0.0, trace_end - span.started_ms)
    return sorted(by_id.values(), key=lambda span: (span.started_ms, span.span_id))


def render_text(path: Path) -> str:
    rows = _load(path)
    if not rows:
        return f"Trace 无法读取或内容为空: {path}"
    spans = _spans(rows)
    start = next((row for row in rows if row["event"] == "task_start"), rows[0])
    end = next((row for row in reversed(rows) if row["event"] == "task_end"), None)
    trace_id = str(start.get("trace_id", path.stem))
    status = str(end.get("status", "interrupted")) if end else "interrupted"
    duration = _number(end.get("duration_ms")) if end else max(
        (_number(row.get("elapsed_ms")) for row in rows), default=0.0,
    )
    end_attrs = end.get("attributes", {}) if isinstance(end, dict) else {}
    if not isinstance(end_attrs, dict):
        end_attrs = {}
    lines = [
        f"任务 {trace_id} · {_status_label(status)} · {_duration(duration)}",
        f"文件: {path}",
    ]
    summary_parts = []
    for key, label in (
        ("turns", "Turn"),
        ("model_requests", "模型请求"),
        ("tool_calls", "工具调用"),
        ("total_tokens", "Token"),
    ):
        value = end_attrs.get(key)
        if isinstance(value, (int, float)):
            summary_parts.append(f"{label} {int(value):,}")
    if summary_parts:
        lines.append("统计: " + " · ".join(summary_parts))
    timing_summary = _timing_summary(spans, duration)
    if timing_summary:
        lines.append(
            "关键路径耗时: " + " · ".join(
                f"{label} {_duration(elapsed)} ({share:.0%})"
                for label, elapsed, share in timing_summary
            )
        )

    # A sub-agent inherits the parent tool span and emits its own round events.
    # Keep only root rounds as section headers; nested model/tool spans remain
    # visible below the outer tool through their parent chain.
    rounds = [
        row for row in rows
        if row["event"] == "round_start" and not row.get("parent_span_id")
    ]
    span_ids = {span.span_id: span for span in spans}
    rendered_span_ids: set[str] = set()
    for round_row in rounds:
        attrs = round_row.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        number = attrs.get("round", "?")
        round_end = next((
            row for row in rows
            if row["event"] == "round_end"
            and not row.get("parent_span_id")
            and isinstance(row.get("attributes"), dict)
            and row["attributes"].get("round") == number
            and _number(row.get("elapsed_ms")) >= _number(round_row.get("elapsed_ms"))
        ), None)
        round_suffix = ""
        if round_end is not None:
            round_duration = max(
                0.0,
                _number(round_end.get("elapsed_ms"))
                - _number(round_row.get("elapsed_ms")),
            )
            round_suffix = (
                f" · {_status_label(str(round_end.get('status', '')))}"
                f" · {_duration(round_duration)}"
            )
        lines.append(f"\nTurn {number}{round_suffix}")
        matching = [
            span for span in spans
            if _effective_round(span, span_ids) == number
        ]
        rendered_span_ids.update(span.span_id for span in matching)
        for index, span in enumerate(matching):
            marker = "└─" if index == len(matching) - 1 else "├─"
            depth = _span_depth(span, span_ids)
            lines.append(f"  {'  ' * depth}{marker} {_span_text(span)}")

    ungrouped = [
        span for span in spans if span.span_id not in rendered_span_ids
    ]
    if ungrouped:
        lines.append("\n其他活动")
        for index, span in enumerate(ungrouped):
            marker = "└─" if index == len(ungrouped) - 1 else "├─"
            depth = _span_depth(span, span_ids)
            lines.append(f"  {'  ' * depth}{marker} {_span_text(span)}")

    notable = [
        row for row in rows
        if row["event"] in {
            "error", "retry", "hitl_wait", "round_limit", "progress_warning",
            "context_compression", "truncation", "steering", "steering_queued",
            "cancellation_requested", "file_changes",
            "tool_call", "tool_result", "tool_blocked",
        }
    ]
    if notable:
        lines.append("\n关键事件")
        for row in notable:
            lines.append(
                f"  {_duration(_number(row.get('elapsed_ms')))} · "
                f"{_event_label(row['event'])}{_attribute_suffix(row.get('attributes'))}"
            )
    return "\n".join(lines)


def render_html(path: Path) -> str:
    rows = _load(path)
    spans = _spans(rows)
    start = next((row for row in rows if row["event"] == "task_start"), {})
    end = next((row for row in reversed(rows) if row["event"] == "task_end"), {})
    trace_id = str(start.get("trace_id", path.stem))
    status = str(end.get("status", "interrupted"))
    task_duration_ms = (
        _number(end.get("duration_ms"))
        if end
        else max((_number(row.get("elapsed_ms")) for row in rows), default=0.0)
    )
    timeline_ms = max(
        1.0,
        task_duration_ms,
        max((_number(row.get("elapsed_ms")) for row in rows), default=0.0),
    )
    span_ids = {span.span_id: span for span in spans}
    bars = []
    for span in spans:
        left = min(100.0, span.started_ms / timeline_ms * 100)
        width = max(0.4, min(100.0 - left, span.duration_ms / timeline_ms * 100))
        depth = _span_depth(span, span_ids)
        details = html.escape(json.dumps(span.attributes, ensure_ascii=False, indent=2))
        cache_summary = _cache_summary(span.attributes)
        bars.append(
            "<details class='span-row'>"
            "<summary class='span-summary'>"
            f"<div class='span-label' style='padding-left:{depth * 18}px'>"
            f"<span class='dot {html.escape(span.status)}'></span>"
            f"{html.escape(_kind_label(span.kind))} · {html.escape(span.name)}"
            f"<small>{html.escape(_duration(span.duration_ms))}{cache_summary}</small></div>"
            "<div class='track'>"
            f"<div class='bar {html.escape(span.status)}' style='left:{left:.3f}%;width:{width:.3f}%' "
            f"title='{html.escape(span.name)} · {_duration(span.duration_ms)}'></div></div>"
            "<span class='detail-toggle'>详情</span></summary>"
            f"<pre class='span-json'>{details}</pre></details>"
        )
    event_rows = []
    for row in rows:
        if row["event"] in {"span_start", "span_end"}:
            continue
        attrs = row.get("attributes", {})
        row_status = str(row.get("status", ""))
        row_class = " class='event-error'" if row_status == "error" else ""
        status_class = " error" if row_status == "error" else ""
        event_rows.append(
            f"<tr{row_class}>"
            f"<td>{_duration(_number(row.get('elapsed_ms')))}</td>"
            f"<td>{html.escape(_event_label(row['event']))}</td>"
            f"<td><span class='event-status{status_class}'>"
            f"{html.escape(row_status)}</span></td>"
            f"<td><pre>{html.escape(json.dumps(attrs, ensure_ascii=False, indent=2))}</pre></td>"
            "</tr>"
        )
    token_svg = _token_chart(spans)
    timing_cards = "".join(
        "<div class='card'><span class='muted'>"
        f"{html.escape(label)}</span><br><strong>{html.escape(_duration(elapsed))}</strong>"
        f" <small>{share:.0%}</small></div>"
        for label, elapsed, share in _timing_summary(spans, task_duration_ms)
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyCode Trace {html.escape(trace_id)}</title>
<style>
:root{{--bg:#111318;--panel:#1a1e26;--muted:#8d98aa;--text:#e8edf5;--line:#303846;--ok:#42d392;--error:#ff6b6b;--warn:#f5c451;--accent:#7799ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:22px;margin:0 0 8px}}h2{{font-size:16px;margin:28px 0 12px}}.muted,small{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:18px 0}}.card{{background:var(--panel);padding:14px;border:1px solid var(--line);border-radius:8px}}
.span-row{{border-top:1px solid var(--line)}}.span-summary{{display:grid;grid-template-columns:minmax(280px,36%) minmax(160px,1fr) 72px;gap:10px;align-items:center;min-height:44px;list-style:none;cursor:pointer}}.span-summary::-webkit-details-marker{{display:none}}.span-label{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.span-label small{{float:right;margin-right:8px}}
.track{{height:14px;background:#222833;border-radius:4px;position:relative}}.bar{{position:absolute;height:100%;border-radius:4px;background:var(--accent);min-width:2px}}.bar.error,.bar.cancelled{{background:var(--error)}}.bar.retry,.bar.interrupted{{background:var(--warn)}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:8px}}.dot.ok{{background:var(--ok)}}.dot.error,.dot.cancelled{{background:var(--error)}}.dot.retry,.dot.interrupted{{background:var(--warn)}}
.detail-toggle{{color:var(--muted);text-align:right;padding-right:4px}}.detail-toggle::before{{content:'▸ ';color:var(--accent)}}.span-row[open] .detail-toggle::before{{content:'▾ '}}.span-json{{margin:0 0 12px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:8px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;overflow:auto}}pre{{white-space:pre-wrap;word-break:break-word;margin:8px 0;font:12px inherit;color:#cbd5e1}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;text-align:left;border-top:1px solid var(--line);vertical-align:top}}th{{color:var(--muted)}}
.event-status.error{{display:inline-block;color:#fff;background:var(--error);font-weight:700;padding:3px 8px;border-radius:999px;box-shadow:0 0 0 1px rgba(255,107,107,.35)}}.event-error{{background:rgba(255,107,107,.055)}}.event-error td:first-child{{box-shadow:inset 3px 0 0 var(--error)}}
svg{{width:100%;height:180px;background:var(--panel);border:1px solid var(--line);border-radius:8px}}.ok-text{{color:var(--ok)}}.error-text{{color:var(--error)}}
@media (max-width:800px){{main{{padding:18px 12px}}.span-summary{{grid-template-columns:minmax(0,1fr) 64px;padding:8px 0}}.track{{grid-column:1/-1;grid-row:2}}.detail-toggle{{grid-column:2;grid-row:1}}.span-label{{grid-column:1;grid-row:1}}.span-json{{margin-bottom:10px}}}}
</style></head><body><main>
<h1>TinyCode 执行 Trace · {html.escape(trace_id)}</h1>
<div class="muted">{html.escape(str(path))}</div>
<div class="cards"><div class="card">状态<br><strong class="{'ok-text' if status in {'ok','no_tool_call'} else 'error-text'}">{html.escape(_status_label(status))}</strong></div>
<div class="card">任务耗时<br><strong>{html.escape(_duration(task_duration_ms))}</strong></div>
<div class="card">Span<br><strong>{len(spans)}</strong></div><div class="card">事件<br><strong>{len(rows)}</strong></div></div>
<h2>关键路径耗时</h2><div class="cards">{timing_cards or "<div class='muted'>没有可归类的耗时 Span</div>"}</div>
<h2>执行时间线</h2>{''.join(bars) or '<div class="muted">没有 Span 数据</div>'}
<h2>模型 Token 曲线</h2>{token_svg}
<h2>上下文窗口曲线</h2>{_context_chart(rows)}
<h2>事件明细</h2><table><thead><tr><th>时间</th><th>事件</th><th>状态</th><th>属性</th></tr></thead><tbody>{''.join(event_rows)}</tbody></table>
</main></body></html>"""


def _token_chart(spans: list[_Span]) -> str:
    points: list[tuple[float, int]] = []
    cumulative = 0
    for span in spans:
        is_model_request = span.kind in {
            "model_request", "compression_model_request"
        }
        is_compression_request = (
            span.kind == "context_compression"
            and span.attributes.get("model_request_made") is True
        )
        if not is_model_request and not is_compression_request:
            continue
        total = span.attributes.get("total_tokens", 0)
        if isinstance(total, (int, float)):
            cumulative += max(0, int(total))
            points.append((span.started_ms + span.duration_ms, cumulative))
    if not points:
        return "<div class='muted'>Provider 未返回 Token 数据</div>"
    max_x = max(point[0] for point in points) or 1
    max_y = max(point[1] for point in points) or 1
    coordinates = " ".join(
        f"{20 + x / max_x * 960:.1f},{160 - y / max_y * 130:.1f}"
        for x, y in points
    )
    return (
        "<svg viewBox='0 0 1000 180' preserveAspectRatio='none'>"
        "<line x1='20' y1='160' x2='980' y2='160' stroke='#303846'/>"
        f"<polyline points='{coordinates}' fill='none' stroke='#7799ff' stroke-width='3'/>"
        f"<text x='24' y='24' fill='#8d98aa'>累计 {max_y:,} Token</text></svg>"
    )


def _timing_summary(
    spans: list[_Span], task_duration_ms: float,
) -> list[tuple[str, float, float]]:
    """Summarize root spans so nested sub-agent time is not double-counted."""
    span_ids = {span.span_id for span in spans}
    totals: dict[str, float] = {}
    labels = {
        "model_request": "模型请求",
        "context_compression": "上下文压缩",
        "compression_model_request": "上下文压缩",
        "tool": "工具执行",
        "user_wait": "用户等待",
        "notes": "自动笔记",
        "workspace_scan": "工作区扫描",
    }
    for span in spans:
        if span.parent_span_id in span_ids:
            continue
        label = labels.get(span.kind, "其他")
        totals[label] = totals.get(label, 0.0) + max(0.0, span.duration_ms)

    denominator = max(1.0, task_duration_ms)
    return [
        (label, elapsed, min(1.0, elapsed / denominator))
        for label, elapsed in sorted(totals.items(), key=lambda item: -item[1])
        if elapsed > 0
    ]


def _context_chart(rows: list[dict[str, Any]]) -> str:
    points: list[tuple[float, int, int]] = []
    for row in rows:
        attrs = row.get("attributes")
        if not isinstance(attrs, dict):
            continue
        if row["event"] == "context_snapshot":
            used = attrs.get("used_tokens")
            window = attrs.get("context_window")
        elif row["event"] == "context_compression":
            used = attrs.get("estimated_tokens_after")
            window = None
        else:
            continue
        if not isinstance(used, (int, float)) or isinstance(used, bool):
            continue
        if not isinstance(window, (int, float)) or isinstance(window, bool):
            window = 0
        points.append((_number(row.get("elapsed_ms")), max(0, int(used)), max(0, int(window))))
    if not points:
        return "<div class='muted'>没有上下文窗口数据</div>"
    known_window = max((point[2] for point in points), default=0)
    max_x = max(point[0] for point in points) or 1
    max_y = max(known_window, max(point[1] for point in points), 1)
    coordinates = " ".join(
        f"{20 + x / max_x * 960:.1f},{160 - used / max_y * 130:.1f}"
        for x, used, _window in points
    )
    window_line = ""
    if known_window:
        y = 160 - known_window / max_y * 130
        window_line = (
            f"<line x1='20' y1='{y:.1f}' x2='980' y2='{y:.1f}' "
            "stroke='#f5c451' stroke-dasharray='8 6'/>"
        )
    return (
        "<svg viewBox='0 0 1000 180' preserveAspectRatio='none'>"
        "<line x1='20' y1='160' x2='980' y2='160' stroke='#303846'/>"
        f"{window_line}<polyline points='{coordinates}' fill='none' "
        "stroke='#42d392' stroke-width='3'/>"
        f"<text x='24' y='24' fill='#8d98aa'>峰值 {max(point[1] for point in points):,} Token"
        + (f" / 窗口 {known_window:,}" if known_window else "")
        + "</text></svg>"
    )


def _span_text(span: _Span) -> str:
    extras = []
    if isinstance(span.attributes.get("first_token_ms"), (int, float)):
        extras.append(f"首 Token {_duration(float(span.attributes['first_token_ms']))}")
    if isinstance(span.attributes.get("total_tokens"), (int, float)):
        extras.append(f"{int(span.attributes['total_tokens']):,} Token")
    cache_text = _cache_summary_text(span.attributes)
    if cache_text:
        extras.append(cache_text)
    if (
        isinstance(span.attributes.get("retry"), (int, float))
        and span.attributes["retry"] > 0
    ):
        extras.append(f"重试 {int(span.attributes['retry'])}")
    suffix = " · " + " · ".join(extras) if extras else ""
    return (
        f"{_kind_label(span.kind)} {span.name} · {_duration(span.duration_ms)} · "
        f"{_status_label(span.status)}{suffix}"
    )


def _cache_summary(attributes: dict[str, Any]) -> str:
    """Return a compact HTML cache summary for model-request spans."""
    text = _cache_summary_text(attributes)
    return f" · {html.escape(text)}" if text else ""


def _cache_summary_text(attributes: dict[str, Any]) -> str:
    """Format cache counters when the provider reports them."""
    if attributes.get("cache_usage_available") is not True:
        return ""
    read = attributes.get("cache_read_tokens")
    miss = attributes.get("cache_miss_tokens")
    write = attributes.get("cache_write_tokens")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (read, miss, write)
    ):
        return ""
    read_count, miss_count, write_count = int(read), int(miss), int(write)
    denominator = read_count + miss_count
    rate = f"{read_count / denominator:.0%}" if denominator else "—"
    parts = [f"缓存 {read_count:,} ({rate})"]
    if write_count:
        parts.append(f"写入 {write_count:,}")
    return " · ".join(parts)


def _span_depth(span: _Span, spans: dict[str, _Span]) -> int:
    depth = 0
    parent = span.parent_span_id
    seen: set[str] = set()
    while parent and parent in spans and parent not in seen and depth < 8:
        seen.add(parent)
        depth += 1
        parent = spans[parent].parent_span_id
    return depth


def _effective_round(span: _Span, spans: dict[str, _Span]) -> object:
    """Use the outermost ancestor round so sub-agent spans stay nested."""
    value = span.attributes.get("round")
    current = span
    seen: set[str] = set()
    while (
        current.parent_span_id
        and current.parent_span_id in spans
        and current.parent_span_id not in seen
    ):
        seen.add(current.parent_span_id)
        current = spans[current.parent_span_id]
        if "round" in current.attributes:
            value = current.attributes["round"]
    return value


def _attribute_suffix(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    selected = []
    for key in ("round", "reason", "error", "message_count", "paths"):
        if key in value:
            selected.append(f"{key}={value[key]}")
    return " · " + ", ".join(selected) if selected else ""


def _kind_label(kind: str) -> str:
    return {
        "model_request": "模型",
        "compression_model_request": "上下文压缩",
        "context_compression": "上下文压缩",
        "tool": "工具",
        "user_wait": "用户等待",
        "notes": "笔记",
        "workspace_scan": "工作区扫描",
    }.get(kind, kind)


def _event_label(event: str) -> str:
    return {
        "task_start": "任务开始",
        "task_end": "任务结束",
        "round_start": "Turn 开始",
        "round_end": "Turn 结束",
        "retry": "模型重试",
        "error": "错误",
        "hitl_wait": "安全确认",
        "round_limit": "轮次扩展确认",
        "progress_warning": "无进展预警",
        "context_compression": "上下文压缩",
        "context_snapshot": "上下文快照",
        "truncation": "工具结果截断",
        "steering": "追加指令",
        "steering_queued": "追加指令已排队",
        "cancellation_requested": "取消请求",
        "file_changes": "文件变更",
        "tool_call": "工具请求",
        "tool_result": "工具结果",
        "tool_blocked": "工具拦截",
    }.get(event, event)


def _status_label(status: str) -> str:
    return {
        "ok": "成功",
        "no_tool_call": "正常完成",
        "error": "失败",
        "cancelled": "已取消",
        "retry": "重试",
        "interrupted": "异常中断",
        "round_budget_stopped": "轮次预算暂停",
        "hard_max_rounds": "达到硬上限",
        "stalled": "无进展暂停",
    }.get(status, status or "未知")


def _duration(milliseconds: float) -> str:
    if milliseconds >= 60_000:
        return f"{milliseconds / 60_000:.2f}m"
    if milliseconds >= 1_000:
        return f"{milliseconds / 1_000:.2f}s"
    return f"{milliseconds:.1f}ms"


def _number(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    return 0.0
