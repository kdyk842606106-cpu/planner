# 三算法正确性与性能实验

版本：`three-engine/v1`。正式规模为 8、16、32、64、128、256 个**活动定义**；不包含业务场景。

## 运行

从 `D:/planner` 使用包含 Solver 依赖的同一个 Python 环境运行：

```powershell
$python = 'D:/Solver_demo_project_planner_exploration/.venv/Scripts/python.exe'
& $python -m planner_experiment.benchmark_suite generate --output runs/three-engine-20261101
& $python -m planner_experiment.benchmark_suite smoke --output runs/three-engine-20261101
& $python -m planner_experiment.benchmark_suite run --output runs/three-engine-20261101
& $python -m planner_experiment.benchmark_suite report --output runs/three-engine-20261101
```

`--solver-root` 可显式指定桥接项目。`run` 默认包含精确枚举、独立预热和正式实验。整批最多八小时；`--hours` 只允许 `(0, 8]`。同一命令可恢复未完成运行，已完成项不重跑；已消耗预算保留。进程异常退出后的未记账时间按墙钟保守计入。操作系统文件锁阻止同一输出目录同时启动两个 runner。

`smoke` 运行 8 个代表实例、每引擎一次，保存在独立 `warmup/`，不计入正式数据。诊断选项 `--skip-oracle` 会明确写入批次状态，不应在正式默认实验中使用。

输入、配置、调度及源代码指纹在生成时冻结。恢复时任一变化都会被拒绝；改变实验或代码须使用新输出目录。`source-snapshot.zip` 保存两个项目的 Python 源代码，不包含 `.env`、数据库或账户配置。

## 实例与复杂度

- 主实验：6 规模 × C0～C6 × 3 生成种子 = 126 个实例。
- 多提供者：18 个新增实例，与 C0 共用骨架；活动数增加 `ceil(N/4)`，单独比较。
- 黄金用例：12 个，其中 10 个可解、2 个语义不可解；输入预检均合法。
- 正式预算：1、3、10 秒，各使用运行种子 11、23、37；黄金用例只用 1 秒。合计 **3996 次**，满求解预算 **18252 秒**。

`complexity.csv` 和每个实例的 `complexity` 保存：活动和状态数、前置关系数、骨架可达对数、顺序强度、精确偏序宽度、深度、关键路径、提供者分布、状态转移/删除比例、可逆强连通分量、次数边界、见证重复次数、资源容量/负荷、事件比例/相对时刻、正负目标、结构相关活动数及工期分布。

对 K 个逻辑任务，顺序约束强度为 `可达任务对数 / (K*(K-1)/2)`，包含间接依赖，冗余边不增加分子；K<2 时约定为 0。宽度使用可达关系上的二分图最大匹配，通过 Dilworth 定理计算，不用最大层大小代替。多提供者映射属于 OR 选择，不直接转成骨架先后关系。

资源负荷为 `sum(d_a*q_ar)/(capacity_r*H0)`，分子固定采用原始骨架活动，H0 固定采用变换前 C0 的加权关键路径。它不是执行利用率，不随算法选出的路径改变。不同组可能改变状态数、目标数、宽度或深度等派生统计，这些变化一起报告；没有任意加权的综合难度分数。

生成器使用固定子种子区分时长/资源分配、交叉边、事件选择和备选实现。C0 为独立链；C1 加入相邻层跨链候选边的一半；C2 链数翻倍、链长减半；C3 含 N/8 个状态恢复对；C4 限制资源容量；C5 设置释放事件；C6 组合上述因素。各组原始活动的 ID、时长及资源需求保持配对。

状态恢复对中，后活动通过前活动的独立完成状态维持先后关系，再独立删除其任务语义状态；这符合当前 required 与 independent-delete 不能引用同一状态的合法性约束。见证在后活动之后恢复前活动，保持原设计的重复执行要求。

## 正确答案与观测

C0/C1/C2/C5/提供者组用任务 DAG、释放时间及最短实现时长建立下界；只有达到下界的排程通过重放和独立资源区间检查，才报告 `PROVEN_OPTIMAL`。C3/C4/C6 的 8/16 活动实例另做精确枚举，每例最多 90 秒、20 万个已访问及待访问状态，总阶段最多 30 分钟。触及时间或状态保护上限为 `UNKNOWN`。枚举记录完整的未来相关状态，不调用 A*/GA 的启发式、标签裁剪或相关性裁剪。

精确答案限定于当前 Simulator 动作语义和原输入次数边界。黄金答案有手算证明，并与精确枚举交叉核对。CP-SAT 对旧引擎选定路径报告 OPTIMAL 不等于全局规划最优。没有证明的大实例只报告下界和本批已知最好值。

新进程固定一个逻辑 CPU；CP-SAT 固定一个工作线程和指定随机种子。公共模块加载和输入预检后开始统一计时，包括引擎构造、POP、搜索、排程、规范化、候选校验。旧引擎只拿到 POP 后剩余排程预算。父进程执行截止时间，并保留已刷入磁盘的合法候选。进程加载和完整调用墙钟耗时另列。

唯一候选按实际内容去重后同步重放，观测/校验/写入开销计入预算。旧引擎在完整计划可重放时才发出候选；本实验不添加 CP-SAT 中间解回调，也不新增旧引擎的路径选择能力。A*/GA 关闭 tracemalloc；内存统一从外部操作系统读取进程峰值工作集，包含加载占用。其他算法默认搜索参数及其内部限制保留，停止原因写入逐次结果。三者仅共用同一时间预算，不共用转换次数预算。

## 输出与阅读

- `manifest.json`、`config.json`、`schedule.json`：冻结输入与运行协议。
- `instances/`：156 个场景、复杂度向量、见证动作及答案证据。
- `oracles/`：精确枚举答案、状态和资源记录。
- `runs/<配对块>/<引擎>/`：唯一候选事件、引擎结果/错误、逐次摘要和最好方案完整重放记录。
- `batch_state.json`：进度、PID、预算、最后完成项及状态。
- `runs.csv`、`instances.csv`、`cells.csv`：逐次、实例内及分组汇总。
- `provider_pairs.csv`、`answers.csv`、`complexity.csv`：配对差异、答案及输入属性。
- `report.html`、`report.md`、`summary.json`：交互图表、可读报告和机器可读汇总。

候选必须在截止前通过校验才计成功。超时前已产生合法解的运行仍算找到合法解。没有合法解按两倍预算计 PAR-2；初始已满足目标的零工期只计算绝对工期差。超时、EXHAUSTED_EMPTY、闭包/队列/代数限制均不能自动变为不可解证明。

先在每个独立输入内汇总重复，再对实例等权平均。未运行项只减少覆盖，不计失败。工期和成功样本首解时间必须结合成功率阅读；质量曲线另附该时刻合法解覆盖率。已知最好值是本批回顾性参考，不是最优值。每格只有三个生成实例，报告定位为探索性比较。

调度按配对块随机引擎顺序并轮转规模。临近整批截止时不再启动可能无法完成的块，剩余项保留为 UNRUN。意外中断的单次事件文件移入 `.interrupted-*` 目录后重新运行，不与新尝试混合。

## 验证命令

```powershell
python -m unittest discover -s tests -v
D:/Solver_demo_project_planner_exploration/.venv/Scripts/python.exe -m pytest `
  D:/Solver_demo_project_planner_exploration/tests/integration/test_planner_benchmark_options.py -q
```

生成与 smoke 不代表正式实验已完成。报告明确显示覆盖率；只有批次状态为 COMPLETED 且正式覆盖为 3996/3996 时，才能称本轮全部运行完成。

## 长预算快速验证（2026-11-02）

`long-quick` 仅复用上一轮冻结场景，不重新生成模型或执行精确枚举。固定生成种子 101、运行种子 11；每条件一次运行，三引擎参数及候选观测方式沿用基线。

| 阶段 | 模型 | 每次预算 | 正式运行数 |
|---|---|---:|---:|
| A | 六档规模 × C0/C4/C6 | 30 秒 | 54 |
| B | C0，128/256 活动 | 60 秒 | 6 |
| C | C0，128/256 活动 | 300 秒 | 6 |

18 个测量模型、22 个三引擎配对块、66 次正式运行，满求解预算 3780 秒。另复制 G01 用于独立预热，不纳入正式样本。A 阶段轮转规模和分组，引擎顺序固定随机化；B/C 顺序在 A 后。配置规定整批最多 4200 秒，`--hours` 及后续恢复只能收紧上限。恢复保留累计耗时，并从已提交的 `run.json` 重算完成数；不足以完成剩余配对块时不启动该块，未运行项保留 UNRUN。

```powershell
$python = 'D:/Solver_demo_project_planner_exploration/.venv/Scripts/python.exe'
$batch = 'D:/planner/runs/three-engine-long-quick-20261102'
& $python -m planner_experiment.benchmark_suite generate --output $batch --profile long-quick --baseline D:/planner/runs/three-engine-20261101
& $python -m planner_experiment.benchmark_suite smoke --output $batch
& $python -m planner_experiment.benchmark_suite run --output $batch
& $python -m planner_experiment.benchmark_suite report --output $batch
```

生成时逐一核对模型哈希，并复制原始 JSON、答案/枚举证明记录及相同模型/引擎/种子的十秒结果到新目录。旧目录保持原样。除 `runner.py`、`profiles.py`、`__main__.py`、`report.py`、`long_report.py` 外，所有冻结的 Planner 和 Solver Python 源码均须与基线一致；包括 worker、算法、Simulator、Validator 及桥接/旧引擎。新源码快照和引用证据一起冻结，运行及恢复时再校验。

`baseline_comparison.csv` 逐次区分新增合法解、仍无候选、工期改善/持平/变差及本轮丢失解；包含首解/首次已知最优时间、重复执行和非因果动作变化、初始候选及种子动作是否仍等于最终方案、停止原因和输入哈希匹配。种子身份只有在引擎种子统计和首候选证据可用时报告，单凭工期相同不视为方案相同。父进程超时导致统计缺失时标为未知；不补造停止原因或种子结论。

报告预算选项和时间曲线支持 30/60/300 秒。A 有 18 个模型，B/C 各只有两个大规模 C0 模型，不能混合三阶段总体成功率解释时间收益。最优未知时仅报告工期变化、下界及已知最好值；本批每条件只有一次运行，不重新确定总体排名。正式完成要求 `COMPLETED` 且覆盖 `66/66`。

## C0 / 256 活动：降低偏序宽度

入口：`experiments/c0_width_sweep.py`。复用长预算实验的 C0/N256/seed101 模型，将原来的16条独立链串接为8、4、2、1条链。分别只新增8、12、14、15条依赖；原240条依赖、256活动的所有属性、512状态、16目标、初始条件、资源容量及执行边界都保持不变。每个变体保存新的实际复杂度和达到解析下界的校验排程。

| 宽度 | 深度 | 前置关系数 | 顺序强度 | 已证明最优工期 |
|---:|---:|---:|---:|---:|
| 16（历史基线） | 16 | 240 | 5.882% | 222 |
| 8 | 32 | 248 | 12.157% | 348 |
| 4 | 64 | 252 | 24.706% | 689 |
| 2 | 128 | 254 | 49.804% | 1363 |
| 1 | 256 | 255 | 100% | 2612 |

仅测 GA 和 ASTAR，每个变体300秒、求解种子11、一次，共8次新运行；W16结果直接复制上一轮相同预算/种子的原始证据，不计入新增运行。满求解预算40分钟，累计批次上限50分钟。每次独立进程、CPU0、原算法参数、相同候选观测/校验/截止规则。宽度、深度、顺序强度和关键路径联动变化；质量必须用各自最优值的相对差距衡量。

```powershell
$python = 'D:/Solver_demo_project_planner_exploration/.venv/Scripts/python.exe'
& $python -m experiments.c0_width_sweep generate
& $python -m experiments.c0_width_sweep smoke
& $python -m experiments.c0_width_sweep run
& $python -m experiments.c0_width_sweep report
```

默认新目录 `D:/planner/runs/c0-width-sweep-20261102`。运行前会验证应用源码、输入、引用证据和实验入口自身哈希。报告 `report.html` / `results.csv` / `summary.json` 按宽度分别展示首解、最优差距、重复次数和停止原因；不将同一N的四种宽度混合汇总。此入口的可选报告回调和两引擎预热不改变标准实验的默认行为。
