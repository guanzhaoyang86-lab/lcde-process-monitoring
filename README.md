# LC-DE：工业过程异常监测研究代码

当前研究方法为 **LC-DE v2 self-calibrated（Local-Contrast Dual-Energy Scan，局部对照双能量扫描）**。此仓库是截至 2026-10-18 的研究代码快照，同时保留 UD-MAPIC、CRIS、EA-HX 等历史方法、实验协议与测试；下方的旧版说明不代表当前主方法。

## 当前模型与代码入口

LC-DE 以健康数据训练的 GRU 预测残差为基础，结合原始传感器的水平、均值、方差和差分能量；利用每条运行的起始参考段作 median/MAD 局部标准化，再以 top-2 融合与有限窗口累积扫描形成报警分数。预测模型使用 CUDA GPU；局部统计与扫描包含 NumPy CPU 计算。

| 内容 | 位置 |
|---|---|
| LC-DE 核心模型 | `src/mate_spc_cris_v9/local_contrast.py` |
| GRU 预测骨干 | `src/mate_spc_cris_v9/model.py` |
| 开发、冻结与测试入口 | `scripts/run_lcde_skab_gpu.py` |
| 统一口径基线比较 | `scripts/compare_skab_frozen_baselines_gpu.py` |
| 数据划分记录 | `protocols/cris_v9_skab_preaccess_charter.json` |
| 中文 PDF 报告生成脚本 | `scripts/build_lcde_research_report_pdf.py` |

当前实验使用 [SKAB v0.9.0](https://github.com/waico/SKAB/tree/v0.9.0)：8 个传感器、34 条异常实验运行，按完整运行划分为 18 条开发运行与 16 条测试运行。每条运行的前 300 个采样点用于起始标准化/校准，之后计入监测指标。

## 当前结果及边界

下表是同一组 16 条测试运行、统一评价口径下的历史实验结果（AP、F1 为逐运行等权平均；事件检出仅统计故障仍活动时的报警）：

| 方法 | AP | F1 | 活动事件检出 |
|---|---:|---:|---:|
| LC-DE dual | 0.8266 | 0.4085 | 9/16 |
| LC-DE raw-only 消融 | 0.8312 | 0.4036 | 10/16 |
| LC-DE predictive-only 消融 | 0.7618 | 0.3884 | 8/16 |
| Global PCA T²+Q | 0.7619 | 0.1964 | 4/16 |
| Self-starting predictive residual CUSUM | 0.5441 | 0.3282 | 8/16 |

完整模型并未全面超过 raw-only，不能据此宣称预测分支已被证实有效或达到 SOTA。原始开发门槛没有全部通过，比较规则随后在开发阶段调整；结果应理解为研究阶段的固定模型测试与回顾性统一比较，而不是原始预注册方案全部通过的确认实验。已查看的测试数据不能继续作为调参后的独立验证集。`other/2.csv` 的故障早于起始校准结束，另有排除此运行的敏感性分析。

数值来源为本地 `outputs/skab_frozen_baseline_comparison/comparison_result.json`（SHA-256：`1aab976640a36ad6392af4f4a337e9868c1759f1ddb66739ca7607fba80b87d6`）。该结果文件及模型权重不随本次代码发布上传；表格是实验快照，不是克隆仓库后重新计算的结果。

## 安装与核心测试

要求 Python 3.10+。先在独立环境中安装适配显卡的 PyTorch CUDA 版本，再安装本项目。GPU 实验曾在 RTX 5090 Laptop GPU 上运行；此仓库不捆绑 CUDA 或 PyTorch。

```bash
python -m pip install -e ".[dev]"
python -c "import torch, pytest, sys; sys.exit(pytest.main(['tests/test_cris_v9_local_contrast.py', 'tests/test_run_lcde_skab_gpu.py', 'tests/test_compare_skab_frozen_baselines.py', '-q']))"
```

Windows 下测试命令先导入 `torch`，以兼容本项目的 DLL 加载顺序。完整历史测试集还可能需要额外依赖、外部数据或先前实验产物。

### 实验入口与复现限制

这次发布的是**代码快照，并非自包含的一键复现包**。LC-DE 入口依赖本地 SKAB 数据、已训练的 `outputs/ea_hx_skab/ea_hx_skab_model.npz` 以及前序开发/冻结记录；基线比较还需要前序方法产物。准备好这些依赖后，入口为：

```bash
python scripts/run_lcde_skab_gpu.py develop
python scripts/run_lcde_skab_gpu.py freeze-comparative
python scripts/run_lcde_skab_gpu.py final
python scripts/compare_skab_frozen_baselines_gpu.py
```

已知的历史链路限制：下载器输出名称使用 `skab_v0.9.0`，上述运行脚本读取的是 `skab-v0.9.0`，需显式对齐数据目录及 ZIP 名称；当前 EA-HX 脚本可重训预测模型，但仅生成 self-starting 版本的基线记录，旧 global 基线记录尚没有对应的完整重建入口。

不要把依赖缺失理解为可直接跳过校验。部分历史协议包含原运行环境的绝对路径与冻结哈希，原样保留供审阅；改变路径后需要新的运行配置，不能冒用旧冻结记录。PDF 脚本还需要本地结果、`reportlab`、`pypdf` 和 Windows 微软雅黑字体。

## 发布范围与第三方代码

发布内容包括 `src/`、`scripts/`、`tests/`、`docs/`、`protocols/` 与必要的 vendored 代码。不上传原始数据、模型权重、`outputs/`、`output/` 或 `tmp/`。

`third_party/` 中两份上游代码按固定提交作为 Git 子模块引用；它们只服务于部分历史 TEP 实验，LC-DE 核心测试不需要初始化。确有需要时运行 `git submodule update --init`。`vendor/tsai_v1_0_1_minirocket/` 保留 Apache-2.0 LICENSE、NOTICE 与来源说明。第三方代码遵循各自许可证；本项目尚未选择统一开源许可证，公开可见不等于授予额外再许可权利。

---

## 历史研究：UD-MAPIC（原始说明）

UD-MAPIC（Uncertainty-Decoupled Mask-Adaptive Process Monitoring）是一个可复现的工业过程监测研究原型。它面向随机缺测、单/多通道断连、整行 timing gap 和状态相关缺失，只用健康 Phase-I 数据训练与校准，输出三种状态：`NORMAL`、`PROCESS_ALARM`、`DATA_QUALITY_UNKNOWN`。

核心思想不是把“缺失越多”直接变成“故障分数越大”。对完整残差二次型 `D=z'Az`，线性高斯后验给出精确分解：

```text
E[D | history] = m' A m + tr(A P) = M + U.
```

- `M=m'Am` 表示后验位置偏移，是主异常证据；
- `U=tr(AP)` 表示缺失造成的不可辨识性，只进入数据质量状态、支持域检查和诊断；
- `M+U` 保留为独立拟合、独立校准的 posterior-expectation 消融。

这样做不意味着 `M` 是完整统计量的无偏估计或 Bayes 最优预测；它是一种有意的决策语义：没有被数据支持的后验方差不应自动计作过程故障能量。

## 方法链路

1. 用健康数据拟合稳定 VARX(1) 线性高斯模型，逐点做严格因果滤波。
2. 在真正完整的健康训练行上拟合 PCA，并固定 T²/SPE 二次型和 0.95 参考尺度。
3. 主瞬时分数使用在完整健康训练行上做稳健中心化/尺度化后的 `log(1+M)`；同时输出 `U`、受保护的 `U/(M+U)`（当分母不超过数值阈值时记为 0）、后验迹和因果重构。
4. 用缺失比例、最长活动缺口、后验迹、`log(1+U)` 和 elapsed index 拟合浅层健康状态树；增强场景按机制平衡加权。
5. 先在当前叶内做 median/MAD 标准化，再做 EWMA，而不是反过来。
6. 全通道缺失或质量特征超出训练支持域时输出 UNKNOWN；EWMA 和 `k-of-w` 窗口都在 UNKNOWN 处重置，不能拼接缺口前后的证据。
7. 对最终持久性统计做 clustered block conformal 校准。同一健康时间块的多种 mask/repeat 先取簇内最坏值，有效样本量仍是基础块数。

完整定义、公式、理论边界和实验协议见 [docs/METHOD.md](docs/METHOD.md)；研究契合、创新边界、数据分工和投稿梯度见 [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md)。

## 安装与验证

要求 Python 3.10+，核心依赖为 NumPy 和 pandas。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
mate-spc demo --output outputs/demo_ud_mapic.json --seed 7
```

当前测试套件包含 40 项单元/集成测试，覆盖：后验二次型恒等式和 Monte Carlo 对照、因果性、PSD 协方差、Phase-I 掩码防泄漏、UNKNOWN 后 EWMA/持久窗口重置、聚类 block conformal、事件一对一归因、自然时间缺口恢复、数据时间轴对齐、确认实验切分与 elapsed 连续性、运行时数据哈希校验、TEP 全故障聚合兼容性、Rieth RData 运行边界、运行级 conformal 校准与运行级 bootstrap，以及 Batch/MetroPT/TEP 适配器。

## 数据集与运行方式

### 1. Batch Distillation v1.1.2

这是当前最匹配的方法数据：119 个真实批次实验、18 个传感器、13 个执行器、1 Hz、专家四状态标签，并提供真实整秒 timing gap。下载器固定到 Zenodo v1.1.2；下载时以及 `batch-run`/`batch-confirm` 执行前都会实际计算并核对四个 ZIP 的 MD5，总量约 75.23 MiB。

```bash
python scripts/download_datasets.py batch --output data
mate-spc batch-list data/batch_distillation_v1.1.2 --contains operating_point_031
```

同 operating point、两个独立健康 run 的开发案例：

```powershell
mate-spc batch-run data\batch_distillation_v1.1.2 `
  --train-key 'Operation/batch_dist_ternary_butan-1-ol+propan-2-ol+water/operating_point_031/train_normal_experiment_001.csv' `
  --calibration-key 'Operation/batch_dist_ternary_butan-1-ol+propan-2-ol+water/operating_point_031/train_normal_experiment_002.csv' `
  --test-key 'Operation/batch_dist_ternary_butan-1-ol+propan-2-ol+water/operating_point_031/test_anormal_experiment_001.csv' `
  --block-minutes 1 --calibration-repeats 2 `
  --output outputs\batch_op031_ud_v020_elapsedfix.json
```

固定的 op028 回顾性复分析协议会同时运行 31 通道部署口径和 18-sensor-only 响应口径。v1 的 split、purge、block、alpha、seed、增强重复和 baseline 清单均被冻结；传入不同值会直接报错，而不会仍冒用 v1 标识。结果 JSON 同时保存请求的分钟参数及其实际换算得到的 samples/minutes：

```powershell
mate-spc batch-confirm data\batch_distillation_v1.1.2 `
  --design-fraction 0.65 --purge-minutes 10 --block-minutes 5 `
  --alpha 0.05 --calibration-repeats 2 --seed 20260907 `
  --output outputs\batch_op028_confirmation_v020_elapsedfix.json
```

op028 v1 将唯一健康 run 顺序切成 12,927 行设计集、600 行 purge 和 6,962 行校准集。校准段保留其在原健康 run 中从 13,527 开始的 elapsed context；每个独立测试 run 从 0 开始。修复后结果见 `outputs/batch_op028_confirmation_v020_elapsedfix.json`：31 通道下 UD-MAPIC 检出 2/4 个 visible events，出现 11 个正常期假警 episode（130.98/100h）；18-sensor-only 下检出 2/4，出现 13 个假警 episode（154.80/100h）。部分简单基线召回更高，但假警同样很高；该回顾性结果不支持方法优越性，反而提示单健康 batch 前/后切分的跨 run 校准失效。由于公开标签、文档和先前结果已被查看，它不能重新称为 untouched/held-out confirmation；单 run 也不满足 run-level 独立校准，conformal 块可交换性只能作近似解释。

数据来源：[Zenodo v1.1.2](https://zenodo.org/records/21535243)；数据说明：[Scientific Data 论文](https://www.nature.com/articles/s41597-026-07124-3)。`Data_interpolated=1` 的发布值来自双向线性插值，没有缺失真值；这些位置只评估报警与覆盖，禁止计算自然缺口插补 RMSE。

### 2. Tennessee Eastman Process

下载器当前固定到经典单 realization 仓库提交，可用于可复现代码压测：

```bash
python scripts/download_datasets.py tep --output data
mate-spc tep-run data/tep_classic_0643318/tennessee-eastman-profBraatz-0643318858ce7c87292bd59820b459e12a13d009 \
  --fault-id 1 --mask-seeds 11 29 47 \
  --output outputs/tep_fault01_ud_v020_elapsedfix.json
```

完整经典集合可在 PowerShell 中运行并聚合：

```powershell
$tepRoot = 'data\tep_classic_0643318\tennessee-eastman-profBraatz-0643318858ce7c87292bd59820b459e12a13d009'
1..21 | ForEach-Object {
  $target = 'outputs\tep_all_v020\fault_{0:D2}.json' -f $_
  mate-spc tep-run $tepRoot --fault-id $_ --mask-seeds 11 29 47 `
    --output $target
}
$env:PYTHONPATH = (Resolve-Path src).Path
python scripts\aggregate_tep_faults.py outputs\tep_all_v020 `
  --first-fault 1 --last-fault 21 `
  --output-json outputs\tep_all_v020\aggregate_faults_01_21.json `
  --output-csv outputs\tep_all_v020\aggregate_faults_01_21.csv `
  --output-markdown outputs\tep_all_v020\REPORT.md
```

最终的 [可读报告](outputs/tep_all_v020/REPORT.md)、[完整聚合 JSON](outputs/tep_all_v020/aggregate_faults_01_21.json) 和 [逐故障 CSV](outputs/tep_all_v020/aggregate_faults_01_21.csv) 覆盖 fault 1–21。每条故障轨迹有五个预声明场景 × 三个 mask seeds，共 21 条不同故障轨迹和 315 个嵌套缺失压力视图；后者不是 315 个独立故障样本。

UD-MAPIC 检出 295/315 个事件评估视图，已检出视图的延迟中位数为 21 个采样点，正常期出现 14 个假警 episode（0.572/100h），平均决策覆盖为 97.53%。完整输入同图参考为 300/315；`M+U` 为 291/315；mask-only/Kalman mean 为 294/315。总体上 P/U 质量上下文只比 mask-only 多检出一个视图，尚不能支持稳定优越性。fault 21 上 UD 的条件延迟为 3，而 mask-only 为 252，这是值得在独立多运行数据上验证的假设，不是单轨迹结论。fault 3 只有 10/15，fault 9 为 0/15，fault 15 的条件延迟为 579，清楚暴露了当前框架的困难区域。

聚合器会验证 21 个文件的配置、数据布局、场景/seed 设计、方法集合和拟合校准摘要，并写入每个原始文件的 SHA-256。插补 RMSE 按 Phase-I 逐特征稳健尺度标准化；检测延迟只在已检出视图中计算，必须和召回一起解释。

### 3. Rieth multi-run TEP pilot

正式多运行入口使用 [Rieth 等人的 TEP 数据](https://doi.org/10.7910/DVN/6C3JR1)。适配器按 `faultNumber × simulationRun` 返回独立序列，不会在运行边界创建伪状态转移。默认冻结协议使用 20 条 fault-free training runs 做设计、20 条不同编号的 fault-free testing runs 做运行级校准，并测试 fault 1、3、9、15、19 各 10 条独立运行。每条测试运行含五个缺失场景 × 三个 mask seeds；750 个视图嵌套在 50 条运行内，bootstrap 在 fault 层内重采样完整运行。

```powershell
python -m pip install -e ".[rdata]"
python scripts\download_datasets.py rieth --output data
python scripts\run_rieth_pilot.py data\tep_rieth_2017 `
  --output-dir outputs\rieth_pilot_v1
```

入口会校验三份官方 RData 的 MD5，冻结 `manifest.json`，保存本地可信的拟合 suite，并逐 fault/run 原子写入结果；相同命令可安全续跑。不要加载来源不可信的 pickle checkpoint。

已完成的 [pilot 报告](outputs/rieth_pilot_v1/REPORT.md) 覆盖 50 条独立故障运行和 750 个嵌套视图。UD-MAPIC 检出 268/750（0.3573，运行级 95% bootstrap CI 0.3240–0.3800），明显高于 `M+U` 的 180/750，却低于 mask-only/Kalman mean 的 280/750；UD 相对 mask-only 的配对召回差为 -0.0160（95% CI -0.0240 至 -0.0080）。因此 pilot 支持“不把后验方差直接计为故障能量”，但不支持当前 P/U quality-context 优于简单 mask-only 的主张。fault 3 和 15 完全未检出，fault 9 几乎未检出；当前方法仍需改进后再扩展到 fault 1–20、更多 test runs 和每条 run 至少 10 个 mask seeds。

### 4. 外部数据

| 角色 | 数据集 | 用途与限制 |
|---|---|---|
| 主要真实案例 | [Batch Distillation v1.1.2](https://zenodo.org/records/21535243) | 真实 timing gap、批相漂移、四状态专家标签 |
| 多运行统计主表 | [Rieth TEP](https://doi.org/10.7910/DVN/6C3JR1) | 500 个模拟运行；用于 run-level 外层重复、消融和置信区间 |
| 外部工程验证 | [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B) | 长序列、真实故障区间；时间重采样必须保留空 bin |
| 小型补充 | [SKAB](https://github.com/waico/skab) | 水循环装置外部泛化，不宜单独支撑主结论 |

下载器支持 `batch`、`metropt`、`tep`、`rieth`、`skab` 和 `all`；各数据集许可证以其官方来源为准，下载数据和 `outputs/` 默认不提交到 Git。

## 必做对照与报告规则

主表至少包含：完整输入同图、UD-MAPIC、`M+U`、mask-only quality context、Kalman mean、中位数、LOCF 和离线线性插值。离线线性插值使用未来值，只能作为非因果参考。所有方法各自拟合状态标准化和 conformal 阈值；传统点方法只使用 mask/gap/time 上下文，不继承提议方法的 P/U 质量变量。

缺失机制固定覆盖 MCAR、单通道 blackout、多通道 blackout、整行 timing gap 和 MNAR 压力生成器。每个设置至少 10 个 mask seeds，但统计置信区间的独立单位必须是 process run，而不是同一运行的 mask 重复。

报告事件召回、逐事件首次检测延迟、正常期 alarm episodes/100h、正常告警占用率、point precision/recall/F1、决策覆盖和 UNKNOWN 比例。主标签 `{2,3}` 表示可见效应；次标签 `{1,2,3}` 表示包含原因潜伏、可见效应和原因移除后恢复期的完整非正常 episode，不能称为“cause present”。故障落在 UNKNOWN 内仍算未检出；UNKNOWN 正常时间不进入假警暴露，但必须单独报告覆盖率。

## 重要边界

- clustered block conformal 只在上游完全冻结且健康基础块 envelope 可交换时有标准 rank 保证。实现把 UNKNOWN 作为 `-∞` 纳入最终块统计，因此该保证是包含拒判机制的整块边际告警概率界，不是给定“可决策”的条件错误率保证；必须同步报告决策覆盖。单条相关、非平稳批序列通常只能给近似解释。
- 结果接口中的 `threshold` 作用于状态标准化后的 EWMA/持久统计尺度；沿用历史命名的 `raw_alarm` 实际表示 EWMA 在应用 `k-of-w` 前越过该阈值，而非原始瞬时分数越界。`instantaneous_reference_threshold` 仅是支持段首点的一步反变换参考，不能用于复算一般时刻的报警。
- 当前 `predict` 是整段输入的逐点因果 replay；每次调用重置滤波器和 EWMA，不是持久化流式服务。
- 未知 MNAR、状态空间错设、超出增强支持域的缺失模式和跨 operating-point 泛化没有分布无关保证。
- `±Δ` 是官方提供的绝对不确定度/容差幅值，不是已知标准差；当前实现没有把它直接当 Kalman 方差。
- 通过测试表示已覆盖路径在当前环境可复现，不等于证明软件不存在任何缺陷，也不等于论文性能结论已经成立。
