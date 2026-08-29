from __future__ import annotations

from statistics import median
from typing import Any

from .models import CandidatePath, EngineResult, to_primitive


def best_path(result: EngineResult) -> CandidatePath | None:
    return min(result.paths, key=lambda item: item.metrics.sort_key, default=None)


def build_comparison(
    astar: EngineResult,
    ga: EngineResult,
    *,
    scenario_id: str,
    run_id: str,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    astar_best = best_path(astar)
    ga_best = best_path(ga)
    diversity = {}
    for label, attribute in (
        ("activity", "structure_signature"),
        ("schedule", "schedule_signature"),
        ("causal_core", "causal_core_signature"),
        ("provider", "provider_signature"),
        ("strategy", "strategy_signature"),
    ):
        left = {getattr(path, attribute) for path in astar.paths}
        right = {getattr(path, attribute) for path in ga.paths}
        diversity[label] = {
            "astar_unique": len(left),
            "ga_unique": len(right),
            "common": sorted(left & right),
            "astar_only": sorted(left - right),
            "ga_only": sorted(right - left),
        }
    activity = diversity["activity"]
    diversity.update(
        {
            "astar_unique_structures": activity["astar_unique"],
            "ga_unique_structures": activity["ga_unique"],
            "common_structures": activity["common"],
            "astar_only_structures": activity["astar_only"],
            "ga_only_structures": activity["ga_only"],
        }
    )
    source = dict(provenance or {})
    projection_checks = {
        "astar": _projection_pair_checks(astar.paths, source),
        "ga": _projection_pair_checks(ga.paths, source),
    }
    limitations = [
        "串行场景保持原语义；并行场景采用不可抢占、事件驱动的具名容量资源模型。",
        "当前并行模型不包含工作日历、消耗型物料、人员个体选择或抢占恢复。",
        "TIMEOUT_EMPTY 只表示预算内未找到方案，不能证明不可行。",
        "只有权重 1、开放列表未裁剪且搜索队列耗尽时才允许证明不可行。",
        "GA 稳定性结论必须使用多个固定种子。",
    ]
    if source.get("excluded_rules"):
        limitations.insert(0, f"本投影明确排除：{source['excluded_rules']}。")
    if source.get("comparison_note"):
        limitations.insert(1, source["comparison_note"])
    return {
        "schema_version": 4,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "provenance": source,
        "projection_checks": projection_checks,
        "astar": _engine_summary(astar, astar_best),
        "ga": _engine_summary(ga, ga_best),
        "diversity": diversity,
        "best_algorithm": _best_algorithm(astar_best, ga_best),
        "limitations": limitations,
    }


def _engine_summary(result: EngineResult, best: CandidatePath | None) -> dict[str, Any]:
    return {
        "status": result.status,
        "seed": result.seed,
        "first_solution_seconds": result.first_solution_seconds,
        "path_count": len(result.paths),
        "best_path": to_primitive(best) if best else None,
        "improvements": to_primitive(result.improvements),
        "stats": to_primitive(result.stats),
        "diagnosis": to_primitive(result.diagnosis),
        "error": result.error,
    }


def _best_algorithm(astar_best: CandidatePath | None, ga_best: CandidatePath | None) -> str | None:
    if astar_best is None and ga_best is None:
        return None
    if ga_best is None:
        return "ASTAR"
    if astar_best is None:
        return "GA"
    if astar_best.metrics.sort_key < ga_best.metrics.sort_key:
        return "ASTAR"
    if ga_best.metrics.sort_key < astar_best.metrics.sort_key:
        return "GA"
    return "TIE"


def comparison_markdown(comparison: dict[str, Any]) -> str:
    astar = comparison["astar"]
    ga = comparison["ga"]
    lines = [
        f"# 双引擎对比报告：{comparison['scenario_id']}",
        "",
        f"运行 ID：`{comparison['run_id']}`　输出协议：v{comparison['schema_version']}",
        "",
    ]
    provenance = comparison.get("provenance") or {}
    if provenance:
        lines.extend(["## 数据来源", ""])
        for key, label in (
            ("source_dataset", "数据集"),
            ("source_project", "源项目"),
            ("source_file", "源文件"),
            ("source_sha256", "源 SHA-256"),
            ("projection", "投影口径"),
        ):
            if provenance.get(key):
                lines.append(f"- {label}：`{provenance[key]}`")
        lines.extend(["", "## 核心结果", ""])
    else:
        lines.extend(["## 核心结果", ""])
    lines.extend([
        "",
        "| 指标 | Anytime A* | GA |",
        "|---|---:|---:|",
        f"| 状态 | {astar['status']} | {ga['status']} |",
        f"| 首解时间（秒） | {_fmt(astar['first_solution_seconds'])} | {_fmt(ga['first_solution_seconds'])} |",
            f"| 合法候选数 | {astar['path_count']} | {ga['path_count']} |",
            f"| 质量池候选 | {astar['stats'].get('archive_tier_counts', {}).get('quality', 0)} | {ga['stats'].get('archive_tier_counts', {}).get('quality', 0)} |",
            f"| 策略池候选 | {astar['stats'].get('archive_tier_counts', {}).get('strategy', 0)} | {ga['stats'].get('archive_tier_counts', {}).get('strategy', 0)} |",
    ])
    for key, label in (
        ("makespan", "最好完工时间"),
        ("execution_count", "活动实例数"),
        ("transition_count", "可逆状态转换"),
        ("state_revisit_count", "状态重访"),
        ("goal_regression_count", "目标回退"),
        ("non_causal_action_count", "非因果活动"),
        ("total_wait", "等待时间"),
        ("activity_duration_sum", "活动工期总和"),
        ("serial_baseline_makespan", "串行重放基线"),
        ("parallel_savings", "并行节省时间"),
        ("compression_ratio", "并行压缩比例"),
        ("peak_parallelism", "峰值并发活动数"),
        ("average_parallelism", "平均并发度"),
        ("critical_path_length", "关键路径长度"),
        ("idle_wait_time", "空闲等待时间"),
    ):
        lines.append(f"| {label} | {_best_metric(astar, key)} | {_best_metric(ga, key)} |")
    lines.append(f"| 规范化前工期 | {_best_field(astar, 'raw_makespan')} | {_best_field(ga, 'raw_makespan')} |")
    lines.append(f"| 排程压缩节省 | {_best_field(astar, 'normalization_saved_time')} | {_best_field(ga, 'normalization_saved_time')} |")
    lines.append(f"| 资源峰值占用 | {_best_metric(astar, 'resource_peak')} | {_best_metric(ga, 'resource_peak')} |")
    lines.append(f"| 资源利用率 | {_best_metric(astar, 'resource_utilization')} | {_best_metric(ga, 'resource_utilization')} |")
    lines.extend(
        [
            f"| Simulator 转移数 | {astar['stats'].get('simulator_transitions', 0)} | {ga['stats'].get('simulator_transitions', 0)} |",
            f"| 峰值内存（字节） | {astar['stats'].get('peak_memory_bytes', 0)} | {ga['stats'].get('peak_memory_bytes', 0)} |",
            "",
            f"统一排序下的最好算法：**{comparison.get('best_algorithm') or '无可行结果'}**",
            "",
        ]
    )
    lines.extend(_gantt_markdown(comparison.get("gantt", [])))
    lines.extend(["## 多层结构覆盖", "", "| 指纹层级 | A* 独特数 | GA 独特数 | 共同数 |", "|---|---:|---:|---:|"])
    for label, title in (("activity", "活动"), ("schedule", "排程"), ("causal_core", "因果核心"), ("provider", "Provider"), ("strategy", "业务策略")):
        item = comparison["diversity"][label]
        lines.append(f"| {title} | {item['astar_unique']} | {item['ga_unique']} | {len(item['common'])} |")
    checks = comparison.get("projection_checks") or {}
    if any(checks.get(engine) for engine in ("astar", "ga")):
        lines.extend(["", "## 源数据并行结构归档覆盖", "", "| 检查 | A* | GA |", "|---|---:|---:|"])
        keys = sorted(set(checks.get("astar", {})) | set(checks.get("ga", {})))
        for key in keys:
            lines.append(
                f"| {key} | {'通过' if checks.get('astar', {}).get(key) else '未满足'} | "
                f"{'通过' if checks.get('ga', {}).get(key) else '未满足'} |"
            )
    lines.extend(["", "## 改进曲线", "", _improvement_table("A*", astar.get("improvements", [])), "", _improvement_table("GA", ga.get("improvements", [])), "", "## 结论限制", ""])
    lines.extend(f"- {item}" for item in comparison["limitations"])
    return "\n".join(lines) + "\n"


def build_benchmark_summary(comparisons: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 4,
        "seeds": seeds,
        "run_count": len(comparisons),
        "provenance": comparisons[0].get("provenance", {}) if comparisons else {},
    }
    for engine in ("astar", "ga"):
        successes = [item for item in comparisons if item[engine]["best_path"] is not None]
        makespans = [item[engine]["best_path"]["metrics"]["makespan"] for item in successes]
        activity: set[str] = set()
        provider: set[str] = set()
        strategy: set[str] = set()
        for item in comparisons:
            for path in item[engine].get("paths", []):
                activity.add(path["structure_signature"])
                provider.add(path["provider_signature"])
                strategy.add(path["strategy_signature"])
        result[engine] = {
            "success_rate": len(successes) / len(comparisons) if comparisons else 0.0,
            "best_makespan": min(makespans, default=None),
            "worst_makespan": max(makespans, default=None),
            "median_makespan": median(makespans) if makespans else None,
            "unique_structures": len(activity),
            "unique_provider_strategies": len(provider),
            "unique_business_strategies": len(strategy),
        }
    result["astar"]["deterministic_best"] = len(
        {item["astar"]["best_path"]["metrics"]["makespan"] for item in comparisons if item["astar"]["best_path"]}
    ) <= 1
    for engine in ("astar", "ga"):
        checks = [
            value
            for comparison in comparisons
            for value in (comparison.get("projection_checks", {}).get(engine, {}) or {}).values()
        ]
        result[engine]["projection_check_pass_rate"] = (
            sum(bool(value) for value in checks) / len(checks) if checks else None
        )
    result["recommendation"] = _recommendation(result, comparisons)
    result["gantt"] = _benchmark_gantt_entries(comparisons)
    result["comparisons"] = comparisons
    return result


def benchmark_markdown(summary: dict[str, Any], scenario_id: str) -> str:
    ga = summary["ga"]
    astar = summary["astar"]
    lines = [
            f"# 双引擎基准汇总：{scenario_id}",
            "",
            f"随机种子：`{summary['seeds']}`",
            "",
    ]
    provenance = summary.get("provenance") or {}
    if provenance:
        lines.extend([
            f"数据来源：`{provenance.get('source_dataset', '-')}` / `{provenance.get('projection', '-')}`",
            "",
        ])
    lines.extend([
            "| 指标 | Anytime A* | GA |",
            "|---|---:|---:|",
            f"| 成功率 | {astar['success_rate']:.0%} | {ga['success_rate']:.0%} |",
            f"| 最好工期 | {astar['best_makespan']} | {ga['best_makespan']} |",
            f"| 中位工期 | {astar['median_makespan']} | {ga['median_makespan']} |",
            f"| 最差工期 | {astar['worst_makespan']} | {ga['worst_makespan']} |",
            f"| 活动结构数 | {astar['unique_structures']} | {ga['unique_structures']} |",
            f"| Provider 策略数 | {astar['unique_provider_strategies']} | {ga['unique_provider_strategies']} |",
            f"| 业务策略家族数 | {astar['unique_business_strategies']} | {ga['unique_business_strategies']} |",
            f"| 源结构检查通过率 | {_pct(astar.get('projection_check_pass_rate'))} | {_pct(ga.get('projection_check_pass_rate'))} |",
            "",
            "## 路线建议",
            "",
            summary["recommendation"],
            "",
    ])
    if provenance.get("comparison_note"):
        lines.extend(["## 对照限制", "", provenance["comparison_note"], ""])
    lines.extend(_gantt_markdown(summary.get("gantt", []), benchmark=True))
    return "\n".join(lines)


def _gantt_markdown(entries: list[dict[str, Any]], *, benchmark: bool = False) -> list[str]:
    if not entries:
        return []
    lines = ["## 求解结果甘特图", ""]
    for entry in entries:
        engine = entry["engine"]
        role = "最好方案" if entry["role"] == "best" else "代表性备选"
        distance = float(entry.get("temporal_distance_from_best", 0.0))
        suffix = "" if entry["role"] == "best" else f"，时态差异 {distance:.1%}"
        run_note = f"，运行 `{entry['run_id']}`" if benchmark and entry.get("run_id") else ""
        lines.extend(
            [
                f"### {engine} {role}",
                "",
                f"makespan：{entry['makespan']}{suffix}{run_note}",
                "",
                f"![{engine} {role}甘特图]({entry['svg_file']})",
                "",
            ]
        )
    return lines


def _benchmark_gantt_entries(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for engine_key, engine_name in (("astar", "ASTAR"), ("ga", "GA")):
        available = [
            comparison
            for comparison in comparisons
            if comparison.get(engine_key, {}).get("best_path")
            and any(item.get("engine") == engine_name for item in comparison.get("gantt", []))
        ]
        if not available:
            continue
        representative = min(
            available,
            key=lambda item: (
                item[engine_key]["best_path"]["metrics"]["makespan"],
                item["run_id"],
            ),
        )
        for entry in representative.get("gantt", []):
            if entry.get("engine") != engine_name:
                continue
            copied = dict(entry)
            copied["run_id"] = representative["run_id"]
            copied["svg_file"] = f"{representative['run_id']}/{entry['svg_file']}"
            result.append(copied)
    return result


def _projection_pair_checks(paths: tuple[CandidatePath, ...], provenance: dict[str, str]) -> dict[str, bool]:
    if not paths:
        return {}
    checks: dict[str, bool] = {}
    for field, expected_overlap in (
        ("expected_overlap_pairs", True),
        ("expected_non_overlap_pairs", False),
    ):
        for token in filter(None, provenance.get(field, "").split(",")):
            left, separator, right = token.partition("+")
            if not separator:
                checks[token] = False
                continue
            label = f"{left}/{right} {'重叠' if expected_overlap else '互斥'}"
            covered = False
            for path in paths:
                records = {record.activity_id: record for record in path.executions}
                if left not in records or right not in records:
                    continue
                overlap = (
                    records[left].start_time < records[right].end_time
                    and records[right].start_time < records[left].end_time
                )
                if overlap is expected_overlap:
                    covered = True
                    break
            checks[label] = covered
    return checks


def _pct(value: Any) -> str:
    return "-" if value is None else f"{value:.0%}"


def _recommendation(summary: dict[str, Any], comparisons: list[dict[str, Any]]) -> str:
    ga = summary["ga"]
    astar = summary["astar"]
    if ga["success_rate"] < 0.8:
        return "GA 合法方案成功率低于 80%，当前不建议作为主算法。"
    if astar["best_makespan"] is None:
        return "A* 未稳定产生可行结果，应优先检查状态空间和内存限制。"
    if sum(bool(item["astar"]["stats"].get("open_list_trimmed")) for item in comparisons) > len(comparisons) / 2:
        return "A* 在多数运行中发生开放列表裁剪，建议继续保留双引擎验证。"
    if ga["best_makespan"] is None or astar["best_makespan"] <= ga["best_makespan"]:
        return "A* 最好质量不差于 GA 且结果稳定，当前优先选择 A*；GA 继续用于策略多样性。"
    return "两种算法各有优势，建议使用更多真实业务场景后再确定生产路线。"


def _best_metric(summary: dict[str, Any], key: str) -> str:
    best = summary.get("best_path")
    return "-" if not best else str(best["metrics"][key])


def _best_field(summary: dict[str, Any], key: str) -> str:
    best = summary.get("best_path")
    return "-" if not best else str(best[key])


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _improvement_table(label: str, points: list[dict[str, Any]]) -> str:
    lines = [f"### {label}", "", "| 时间（秒） | 分数 | 结构 |", "|---:|---|---|"]
    if not points:
        lines.append("| - | 无改进记录 | - |")
    else:
        for point in points:
            lines.append(f"| {_fmt(point['elapsed_seconds'])} | `{point['score']}` | `{point['signature']}` |")
    return "\n".join(lines)
