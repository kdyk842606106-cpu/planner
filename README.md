# Planner 双引擎验证实验

这是一个纯 Python、无数据库、无 API、无第三方依赖的路径探索实验。Anytime A*/ARA* 与状态特征 GA 使用同一个串行/时态事件 Simulator，所有正式候选都由独立 Validator 从初始状态重放。

当前支持：

- 原生显式 `transition` 前置关系、活动自动目标状态和可选业务次数上限；
- 旧等值 `preconditions/effects` 场景的加载时转换；
- 通用定时外部事件及旧 `materials` 格式兼容；
- 自动识别可逆状态域并统计转换、重访和目标回退；
- 固定活动顺序的最早排程规范化；
- 活动、排程、因果核心、provider 和业务策略五层指纹；
- 110% 高质量池与 125% 独立策略池；
- ARA* 轻量父节点、目标 provider 多样性搜索；
- GA 双分支解码、phenotype 缓存和策略家族精英；
- Windows `spawn` 双进程独立对比。
- 事件驱动活动并行、事实读写锁和具名容量资源。

实验支持串行与不可抢占并行模式；仍不实现工作日历、消耗型物料、人员个体选择、抢占恢复或生产级调度。

## 快速运行

```powershell
python -m planner_experiment validate `
  --scenario scenarios/module_x.json `
  --actions scenarios/module_x_early_actions.json

python -m planner_experiment compare `
  --scenario scenarios/thermal_validation.json `
  --seed 11 `
  --output runs/thermal-compare

python -m planner_experiment benchmark `
  --scenario scenarios/generic_modes.json `
  --seeds 11,23,37,53,71 `
  --output runs/generic-benchmark
```

正式压力场景：

```powershell
python -m planner_experiment benchmark `
  --scenario scenarios/synthetic_pressure.json `
  --time-limit 30 `
  --transition-limit 200000 `
  --seeds 11,23,37,53,71 `
  --output runs/phase2-final/synthetic-pressure
```

每次对比生成 `manifest.json`、`astar_result.json`、`ga_result.json`、`comparison.json` 和 `report.md`；并为 A*/GA 的最好方案及代表性备选生成最多四张嵌入式 SVG 甘特图。benchmark 额外生成 `benchmark.json`，根报告链接到代表种子运行的甘特图。当前输出协议为 schema v4，Validator 版本为 `temporal-event-validator-v4`。历史 schema v3 结果只保留归档，不自动迁移或混入 v4 汇总。

场景缺省使用 `execution_mode: "serial"`，因此原有场景和 `EXECUTE/WAIT` 动作文件行为不变。设置 `execution_mode: "parallel"` 后，活动通过 `START` 在当前时刻启动，通过 `ADVANCE` 推进到最近的活动完成或外部事件；可在场景中用 `resources` 定义容量资源、用活动的 `resource_reqs` 声明占用量。CLI 也支持 `--execution-mode serial|parallel` 覆盖场景配置。

并行示例：

```powershell
python -m planner_experiment benchmark --scenario scenarios/parallel_workflow.json --output runs/parallel-phase
```

MI-HP-001 源数据投影与基准：

```powershell
python -m planner_experiment.mi_hp_import --check
python -m planner_experiment benchmark `
  --scenario scenarios/solver_demo_mi_hp_core_parallel.json `
  --time-limit 30 `
  --transition-limit 500000 `
  --seeds 11,23,37,53,71 `
  --output runs/solver-demo-mi-hp-core
```

该投影保留 `solver_demo_project` 的 36 个活动、52 条依赖和 12 类资源，但明确排除工作日历、行吊对全计划独占、责任子系统连续和功能调测独占，因此不能与源项目包含规则的 1170/1316 分钟结果直接比较。

## 显式 transition 场景

新场景以活动为管理对象。每个活动通过 `output_state_id` 绑定一个自动目标状态，前置关系用 `relation_role` 区分普通条件与被替换状态：

```json
{
  "initial_state_ids": ["state:x_uninstalled", "state:power_off"],
  "target_activity_ids": ["install_x"],
  "activities": [
    {
      "id": "install_x",
      "name": "安装 X",
      "duration": 10,
      "output_state_id": "state:x_installed",
      "preconditions": [
        {"state_id": "state:x_uninstalled", "relation_role": "transition"},
        {"state_id": "state:power_off", "relation_role": "required"}
      ]
    }
  ]
}
```

执行结果为“删除 `transition` 前置状态、保留 `required` 状态、加入活动目标状态”。未填写 `output_state_id` 时使用 `activity:<activity_id>:output` 自动生成；没有迁移前置的纯里程碑活动必须显式设置 `is_milestone: true`。目标活动的输出状态会自动成为求解目标。

所有活动默认允许重复：省略 `max_instances` 表示没有活动级次数上限，方案统一由正整数 `max_steps` 限制活动实例总数。只有明确的业务次数约束才配置正整数 `max_instances`。A* 和 GA 会剪除不改善业务状态的重复路径，但 Validator 仍可合法重放 `max_steps` 范围内的手工冗余重复动作。

旧字典格式继续可用。加载器会把每个 `key=value` 转成稳定状态 ID，并把同键前置与效果的值变化转换为显式迁移；旧 SET 效果中没有声明旧值的覆盖行为只保留在兼容适配层。

## 固化场景

- `module_x.json`：到货事件与可逆模式切换；
- `module_x_explicit_transition.json`：活动自动状态、安装/拆卸循环和 required/transition 关系；
- `generic_modes.json`：模式、夹具、组合/分体 provider；
- `approval_release.json`：审批反馈和发布窗口；
- `thermal_validation.json`：组合温控与冷热分体策略；
- `synthetic_pressure.json`：确定性压力场景。
- `parallel_workflow.json` / `parallel_pressure.json`：并行语义与资源压力场景；
- `solver_demo_mi_hp_core_parallel.json`：MI-HP-001 核心并行投影。

旧 `materials`/`material_reqs` 会在加载时转换为同 ID 的外部事件和事件要求。

## 测试

```powershell
python -m unittest discover -s tests -v
```

`TIMEOUT_EMPTY` 只表示预算内没有找到方案；只有权重 1、开放列表未裁剪且主搜索队列耗尽时，A* 才能报告 `PROVEN_INFEASIBLE`。
