# UD-MAPIC 论文与投稿方案

## 1. 研究对象与契合度

本文按新加坡国立大学 ISEM 的 Ye Zhisheng（叶志胜）教授进行匹配。他的官方研究方向包括：

- 数据驱动运营管理；
- 数理与工业统计；
- 系统韧性、可靠性工程、状态维修与复杂系统建模。

来源：

- [NUS ISEM faculty profile](https://cde.nus.edu.sg/isem/staff/ye-zhisheng/)
- [Ye Zhisheng research page](https://blog.nus.edu.sg/iseyezh/research/)

与本文最接近的近期工作是：

- [High-dimensional process monitoring under time-varying operating conditions via covariate-regulated PCA](https://www.sciencedirect.com/science/article/pii/S095183202400512X)
- [Condition monitoring based on corrupted multiple time series with common trends](https://www.sciencedirect.com/science/article/pii/S095183202400396X)
- [Auxiliary covariate-assisted condition monitoring for machinery](https://www.sciencedirect.com/science/article/pii/S0951832025011573)

因此，结构化缺失、数据质量、时变工况和可靠性监测的交叉点与该教授当前研究脉络直接相连。

## 2. 建议题目与研究问题

英文题目：

> Uncertainty-Decoupled Mask-Adaptive Process Monitoring under Structured Sensor Missingness

可选副标题：

> Separating Process Evidence from Data-Quality Uncertainty

核心问题：

> 当工业传感器随机缺测、连续断连或整行丢包时，如何区分“有证据表明过程异常”与“数据不足以作出可靠判断”，并在时变缺失质量下形成可校准的事件级报警？

## 3. 可辩护的创新边界

已有研究已经讨论缺失观测下 PCA/PLS 统计量的不确定性，也已有 TEP 上的 conformal 报警方法。因此不能把“后验不确定性”“EWMA”或“conformal”单独写成首创：

- [Nelson et al. (2006), missing measurements in PCA/PLS monitoring](https://www.sciencedirect.com/science/article/pii/S0169743905000675)
- [Probabilistic process monitoring with incomplete measurements](https://www.sciencedirect.com/science/article/abs/pii/S0169743914002639)
- [RBC-AD: conformal anomaly detection for TEP](https://www.tandfonline.com/doi/abs/10.1080/09540091.2026.2707826)

本文应限定为以下组合创新，并把它们作为需要实验检验的贡献，而不是预先宣称已经优于现有方法：

1. **决策语义解耦**：对后验二次型期望
   `E[D|H]=M+U`，只把位置项 `M=m'Am` 作为过程异常证据；把 `U=tr(AP)` 解释为缺失诱发的不可辨识性，不自动累加成故障能量。
2. **数据质量治理**：由缺失比例、连续缺口、后验迹和 `U` 构成质量上下文，进行健康状态分层、支持域判断与拒判；输出 `NORMAL / PROCESS_ALARM / DATA_QUALITY_UNKNOWN`。
3. **最终报警校准**：先做状态内稳健标准化，再做 EWMA 与分段 `k-of-w`；同一健康时间块的重复 mask 先聚类取 envelope，再校准最终事件报警阈值。
4. **面向数据质量的评估**：故障落在 UNKNOWN 中仍算漏检，同时报告决策覆盖、风险—覆盖关系、事件召回、延迟和正常期 alarm episodes/100h。

完整数学与算法定义见 [METHOD.md](METHOD.md)。

## 4. 数据集分工

### 主要真实案例

[Batch Distillation v1.1.2](https://zenodo.org/records/21535243)：

- 119 个真实批次；
- 18 个传感器、13 个执行器；
- 1 Hz 时间序列；
- 专家四状态异常标签；
- 真实整秒 timing gaps；
- 可形成“数据质量 + 批相变化 + 异常监测”的应用主线。

正式实验应优先选择具有多个独立健康 run 的 operating points，按完整 run 分配设计、校准和测试角色；不能用同一健康 run 的前后时间段替代独立 run。

### 多运行统计主表

[Rieth multi-run Tennessee Eastman Process](https://doi.org/10.7910/DVN/6C3JR1)：

- 每种故障有 500 个独立模拟运行；
- 适合按 run 做外层拆分、置信区间与 20 类故障分层；
- 是验证缺失机制 × 故障类型交互的主要统计基准。

这里的 20 类是 Rieth 数据自身的 fault 1–20；当前 classic single-realization stress suite 另含 fault 21。两者范围不同，不能把 Rieth 写成 fault 1–21。

经典单 realization TEP 只保留为代码压力测试，不能把 mask seeds 当成独立过程样本。

### 外部工程验证

[MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B)：

- 1,516,948 个时间点、15 个信号；
- 真实铁路压缩机与故障区间；
- 用来检查长序列、报警负担和跨设备外部有效性。

## 5. 冻结实验协议

方法：

- 完整观测同图；
- UD-MAPIC；
- `M+U` posterior-expectation 消融；
- mask-only context 消融；
- median、LOCF、Kalman mean；
- offline linear interpolation（必须标记非因果）。

缺失压力：

- MCAR；
- 单通道 blackout；
- 多通道 blackout；
- 整行 timing gap；
- MNAR 压力生成。

统计规则：

- 每个独立 process run 再生成至少 10 个 mask seeds；
- 独立统计单位始终是 process run，mask replicate 只作 run 内重复；
- bootstrap 置信区间按 run 聚类；
- 主指标为事件召回、事件首次报警延迟、alarm episodes/100h、决策覆盖和 UNKNOWN 比例；
- 在查看正式测试标签前冻结主公式、阈值流程和超参数；
- 不把 conformal 的边际 block 保证误写成条件于可决策样本的错误率保证。

## 6. 当前证据

- 代码版本 0.2.0，37 项单元/集成测试通过。
- [classic TEP fault 1–21 聚合结果](../outputs/tep_all_v020/aggregate_faults_01_21.json)：21 条不同故障轨迹形成 315 个相关 stress views。UD 检出 295/315、条件中位延迟 21、14 个假警 episode、平均覆盖 0.9753；fault 9 完全未检出，fault 3 部分检出，fault 15 延迟很大。该结果提供工程覆盖但不提供独立多运行证据，且不支持主方法对基线稳定占优。
- `outputs/batch_op031_ud_v020_elapsedfix.json`：UD 检出 1/2、延迟 9 s、0 假警；LOCF 检出 2/2 但延迟 331 s。
- `outputs/batch_op028_confirmation_v020_elapsedfix.json`：UD 在两种通道口径均检出 2/4，但假警为 130.98 和 154.80 episodes/100h；该结果暴露单健康 batch 时间切分的跨 run 校准失效。

结论：实现已完成 classic 全 fault-type 压力覆盖，但“覆盖 fault 1–21”不等于“完成独立多运行验证”。当前证据仍然**不够投稿**，也不支持“主方法优于基线”。最低投稿门槛是补齐独立多运行验证，并证明 P/U 质量治理至少在风险—覆盖、严重 blackout 或 MNAR 场景中带来稳定增益。

## 7. 投稿梯度

### 首选快速路线

[Quality Engineering](https://www.tandfonline.com/journals/lqen20/about-this-journal) 的 [Data Quality 专刊](https://think.taylorandfrancis.com/special_issues/data-quality/)。

- 主题直接覆盖缺失数据、数据质量及其后果；
- 期刊接受应用驱动的新方法和工业案例；
- 当前官方截稿日期为 **2027 年 1 月 31 日**。

适合定位为“方法 + 数据质量治理 + 两个工业案例”，但必须先达到上面的最低实验门槛。

### 强结果路线

[Reliability Engineering & System Safety](https://shop.elsevier.com/journals/reliability-engineering-and-system-safety/0951-8320)。

与 Ye 教授近期论文及可靠性/故障检测主题最一致；需要 Rieth 全故障多运行结果、真实案例、清晰理论边界和模型错设敏感性，门槛明显更高。

### 化工过程备选

[Chemometrics and Intelligent Laboratory Systems](https://shop.elsevier.com/journals/chemometrics-and-intelligent-laboratory-systems/0169-7439)。

若最终贡献更偏 Batch Distillation、PCA/SPE/T² 与过程数据分析，这是合理备选；仅做常规应用不够，必须保留方法贡献。

## 8. 最短可执行路线

1. 冻结 v0.2.0 为开发版，不再根据当前 op031/op028 标签调整主公式。
2. Rieth RData/multi-run 适配器、运行边界隔离、运行级 conformal 与可续跑入口已经完成。首个冻结 pilot 使用 fault 1、3、9、15、19 各 10 条运行；结果不支持当前 P/U context 优于 mask-only，因此先做困难故障诊断，不直接扩大并宣称胜出。
3. 方法冻结后，在 fault 1–20 的更多独立 runs 上按相同 run/scenario/mask 配对运行完整方法矩阵；mask replicate 留在 run 内，bootstrap、置信区间和统计检验按 run 聚类。fault 21 若要进入推断性主表，必须另找或生成具有独立随机种子的 multi-run 数据，否则只放 classic stress 附录。
4. 若 P/U 治理没有稳定增益，及时把论文转成“何时应该拒判”的数据质量研究，或停止该方法路线；不要只挑有利 fault。
5. 通过门槛后，以 Quality Engineering Data Quality 专刊为第一投稿目标；若理论和多运行结果显著增强，再投 RESS。
