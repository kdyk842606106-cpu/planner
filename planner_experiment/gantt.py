from __future__ import annotations

import html
import math
from itertools import combinations
from pathlib import Path

from .models import CandidatePath, EngineResult, ExecutionRecord, Scenario


GANTT_RENDERER_VERSION = "svg-gantt-v1"
_PALETTE = (
    "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2",
    "#4f46e5", "#65a30d", "#db2777", "#0f766e", "#b45309", "#475569",
)


def temporal_distance(left: CandidatePath, right: CandidatePath) -> float:
    """Fraction of common activity pairs whose before/after/overlap relation differs."""
    left_records = {item.instance_id: item for item in left.executions}
    right_records = {item.instance_id: item for item in right.executions}
    common = sorted(set(left_records) & set(right_records))
    if len(common) < 2:
        return 0.0
    changed = total = 0
    for first, second in combinations(common, 2):
        total += 1
        if _relation(left_records[first], left_records[second]) != _relation(
            right_records[first], right_records[second]
        ):
            changed += 1
    return changed / total


def _relation(left: ExecutionRecord, right: ExecutionRecord) -> int:
    if left.end_time <= right.start_time:
        return -1
    if right.end_time <= left.start_time:
        return 1
    return 0


def select_gantt_candidates(result: EngineResult) -> tuple[tuple[str, CandidatePath, float], ...]:
    if not result.paths:
        return ()
    best = min(result.paths, key=lambda item: (item.metrics.sort_key, item.schedule_signature))
    selected: list[tuple[str, CandidatePath, float]] = [("best", best, 0.0)]
    alternatives = [item for item in result.paths if item.schedule_signature != best.schedule_signature]
    if alternatives:
        ranked = sorted(
            alternatives,
            key=lambda item: (
                -temporal_distance(best, item),
                item.metrics.sort_key,
                item.schedule_signature,
            ),
        )
        alternative = ranked[0]
        selected.append(("alternative", alternative, temporal_distance(best, alternative)))
    return tuple(selected)


def write_gantt_artifacts(
    scenario: Scenario,
    results: dict[str, EngineResult],
    destination: str | Path,
) -> list[dict[str, object]]:
    target = Path(destination)
    artifacts: list[dict[str, object]] = []
    for engine_name in ("ASTAR", "GA"):
        result = results[engine_name]
        for role, candidate, distance in select_gantt_candidates(result):
            engine_slug = engine_name.lower()
            filename = f"{engine_slug}_{role}_gantt.svg"
            label = "最好方案" if role == "best" else "代表性备选"
            title = f"{engine_name} {label} · makespan {candidate.metrics.makespan}"
            (target / filename).write_text(
                render_gantt_svg(scenario, candidate, title=title),
                encoding="utf-8",
            )
            artifacts.append(
                {
                    "engine": engine_name,
                    "role": role,
                    "path_id": candidate.path_id,
                    "makespan": candidate.metrics.makespan,
                    "svg_file": filename,
                    "temporal_distance_from_best": round(distance, 6),
                }
            )
    return artifacts


def render_gantt_svg(scenario: Scenario, candidate: CandidatePath, *, title: str) -> str:
    records = sorted(
        candidate.executions,
        key=lambda item: (item.start_time, item.activity_id, item.ordinal),
    )
    width, label_width, right_margin = 1440, 360, 40
    top, row_height, axis_height = 84, 27, 34
    plot_width = width - label_width - right_margin
    makespan = max(1, max((item.end_time for item in records), default=scenario.start_time) - scenario.start_time)
    resource_ids = sorted(
        {
            resource_id
            for record in records
            for resource_id, _ in scenario.activity_by_id[record.activity_id].resource_reqs
        }
    )
    colors = {resource_id: _PALETTE[index % len(_PALETTE)] for index, resource_id in enumerate(resource_ids)}
    legend_columns = 4
    legend_rows = math.ceil(len(resource_ids) / legend_columns) if resource_ids else 0
    legend_height = legend_rows * 24 + (24 if resource_ids else 0)
    height = top + len(records) * row_height + axis_height + legend_height + 28

    def x_position(absolute_time: int) -> float:
        return label_width + (absolute_time - scenario.start_time) / makespan * plot_width

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{_escape(title)}</title>",
        "<style>text{font-family:'Segoe UI','Microsoft YaHei',sans-serif;fill:#172033}.grid{stroke:#d9e0ea;stroke-width:1}.row{fill:#f8fafc}.axis{font-size:12px;fill:#526075}.label{font-size:12px}.bartext{font-size:11px;fill:white;font-weight:600}.legend{font-size:12px}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="32" font-size="20" font-weight="700">{_escape(title)}</text>',
        f'<text x="20" y="55" class="axis">活动数 {len(records)} · makespan {candidate.metrics.makespan} · 路径 { _escape(candidate.path_id) }</text>',
    ]
    tick_step = _nice_tick(makespan)
    tick = 0
    axis_bottom = top + len(records) * row_height
    while tick <= makespan:
        x = label_width + tick / makespan * plot_width
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top - 12}" x2="{x:.2f}" y2="{axis_bottom}"/>')
        lines.append(f'<text class="axis" x="{x:.2f}" y="{top - 20}" text-anchor="middle">{tick}</text>')
        tick += tick_step
    if (tick - tick_step) != makespan:
        x = label_width + plot_width
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top - 12}" x2="{x:.2f}" y2="{axis_bottom}"/>')
        lines.append(f'<text class="axis" x="{x:.2f}" y="{top - 20}" text-anchor="middle">{makespan}</text>')

    lines.append('<g id="activity-bars">')
    for index, record in enumerate(records):
        y = top + index * row_height
        if index % 2:
            lines.append(f'<rect class="row" x="0" y="{y}" width="{width}" height="{row_height}"/>')
        activity = scenario.activity_by_id[record.activity_id]
        resources = activity.resource_reqs
        primary_resource = resources[0][0] if resources else ""
        color = colors.get(primary_resource, "#64748b")
        label = _truncate(f"{record.activity_id}  {record.activity_name}", 43)
        lines.append(f'<text class="label" x="18" y="{y + 18}">{_escape(label)}</text>')
        x = x_position(record.start_time)
        bar_width = max(2.0, (record.end_time - record.start_time) / makespan * plot_width)
        resource_text = ", ".join(f"{item}×{quantity}" for item, quantity in resources) or "无"
        tooltip = (
            f"{record.activity_id} | {record.activity_name} | "
            f"{record.start_time}–{record.end_time} | 工期 {record.end_time - record.start_time} | 资源 {resource_text}"
        )
        lines.append(
            f'<rect x="{x:.2f}" y="{y + 4}" width="{bar_width:.2f}" height="19" rx="3" fill="{color}" '
            f'data-activity-id="{_escape(record.activity_id)}" data-start="{record.start_time}" '
            f'data-end="{record.end_time}" data-resource="{_escape(primary_resource)}">'
            f'<title>{_escape(tooltip)}</title></rect>'
        )
        interval = f"{record.start_time}–{record.end_time}"
        if bar_width >= 54:
            lines.append(f'<text class="bartext" x="{x + 5:.2f}" y="{y + 18}">{interval}</text>')
        else:
            lines.append(f'<text class="axis" x="{x + bar_width + 4:.2f}" y="{y + 18}">{interval}</text>')
    lines.append("</g>")

    if resource_ids:
        legend_y = axis_bottom + axis_height
        lines.append(f'<text x="18" y="{legend_y}" font-size="13" font-weight="700">主要资源颜色</text>')
        cell_width = (width - 36) / legend_columns
        for index, resource_id in enumerate(resource_ids):
            column, row = index % legend_columns, index // legend_columns
            x = 18 + column * cell_width
            y = legend_y + 14 + row * 24
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="14" height="14" rx="2" fill="{colors[resource_id]}"/>')
            lines.append(f'<text class="legend" x="{x + 20:.2f}" y="{y + 12:.2f}">{_escape(resource_id)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _nice_tick(span: int) -> int:
    raw = max(1.0, span / 10)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiplier in (1, 2, 5, 10):
        candidate = int(multiplier * magnitude)
        if candidate >= raw:
            return max(1, candidate)
    return max(1, int(10 * magnitude))


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)
