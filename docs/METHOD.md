# UD-MAPIC 方法与无泄漏实验协议

## 1. 目标、信息集与划分

令完整过程向量为 `x_t ∈ R^d`，观测掩码为 `r_t ∈ {0,1}^d`，可靠协变量为 `u_t`。在线时刻 `t` 只能使用当前及过去的观测、当前已知协变量和冻结参数，不能使用未来样本或测试标签。

标准实验使用三个互斥角色：

1. `D_train`：健康设计数据，用于动态模型、PCA、分数尺度、质量状态树及叶内中心/尺度；
2. `D_cal`：独立健康校准数据，只用于最终 clustered block-conformal order statistic；
3. `D_test`：含故障测试数据，标签只用于最终指标。

若只有一个健康 batch，简单的前段训练、后段校准可能因批相非平稳而失败。op028-v1 只能作为冻结协议下的回顾性复分析；不能把它重新描述成 untouched confirmation，也不能把单 run 时间切分描述成 run-level 独立校准。

## 2. 因果线性高斯后验

先用健康训练数据的稳健 location/scale 标准化。设标准化状态满足稳定 VARX(1)：

```text
z_t = a + F z_{t-1} + B u_t + ε_t,    ε_t ~ N(0,Q).
```

`F` 的谱半径截到 0.995，`Q` 做对角收缩。动态模型的位置、尺度和 VARX 参数只使用完全观测训练行，VARX 转移只使用相邻两端均完全观测的转移；改变 masked release values 不会改变拟合结果。

传播得到 `(μ_t^-,P_t^-)` 后，对当前观测维度 `o` 和缺失维度 `m` 做高斯条件化：

```text
m_m = μ_m^- + P_mo^- (P_oo^-)^{-1}(z_o-μ_o^-)
P_mm = P_mm^- - P_mo^- (P_oo^-)^{-1}P_om^- .
```

观测坐标的后验均值等于实际观测、后验方差为零；全通道缺失时保留传播先验并输出 UNKNOWN。实现使用带 jitter 的 Cholesky 求解，不显式求逆。

## 3. PCA 二次型与干净尺度

在真正完整的健康训练行上拟合可靠协变量基线，对残差做 PCA。令保留子空间的 Hotelling T² 矩阵为 `A_T`，残差子空间 SPE 矩阵为 `A_Q`。

两个参考尺度只从 burn-in 后的完整健康训练行估计：

```text
q_T = Q_.95(z' A_T z),
q_Q = Q_.95(z' A_Q z),
A*  = A_T/q_T + A_Q/q_Q .
```

因此 UD 主模型和 `M+U` 消融共享同一个 PCA 二次型及全局 complete-row 尺度；消融差异不会混入另一套 component scaling。

## 4. 异常证据与数据质量不确定性解耦

设过滤残差后验为：

```text
z_t | H_t ~ N(m_t,P_t),       D_t = z_t' A* z_t.
```

高斯二次型满足精确恒等式：

```text
E[D_t | H_t] = m_t' A* m_t + tr(A*P_t) = M_t + U_t.
```

其中：

- `M_t=m_t'A*m_t` 是被当前数据支持的后验位置异常能量；
- `U_t=tr(A*P_t)` 是缺失造成的完整统计量不可辨识性；
- `rho_t=U_t/(M_t+U_t)` 只作为可解释诊断输出；实现仅在 `M_t+U_t>ε` 时执行该除法，否则令 `rho_t=0`。

UD-MAPIC 主瞬时证据为：

```text
X_t = [log(1+M_t)-c_0]/s_0,
```

其中 `c_0,s_0` 只由完整健康训练行的 median/MAD（必要时用标准差回退）确定。主方法不把 `U_t` 加入异常能量，因为健康过程仅因缺测而增大的不可辨识性不应自动触发故障报警。

独立消融 `MAPIC_integrated_expectation` 使用：

```text
X_t^int = [log(1+M_t+U_t)-c_0]/s_0.
```

必须准确表述：`M+U` 才是模型正确时 `D` 的条件期望和平方损失最优预测；`M` 不是 `D` 的无偏估计，也不宣称 Bayes 最优。选择 `M` 是报警语义选择。

## 5. 后验质量治理与状态标准化

主方法的健康状态特征为：

```text
c_t = [missing_fraction_t,
       log(1+longest_active_gap_t),
       log(1+tr(P_t)/d),
       log(1+U_t),
       log(1+run_elapsed_sample_index_t)].
```

`rho_t` 不进入树和支持域，因为它的分母含 `M_t`，真实故障会改变它并可能被错误拒判。前四项是数据质量支持特征；run-relative elapsed index 可用于树分裂但允许外推。独立运行从 0 开始；若同一健康 run 被顺序切成设计/校准段，校准段必须携带原始起点 offset，不能把晚期片段伪装成从 0 开始的新 batch。

在 `D_train` 的自然 mask 和预声明人工缺失增强上拟合浅层回归树。不同机制先按完整参数标识，再以机制平衡权重进入树，避免大量增强重复支配状态定义。部署时在叶 `ℓ(t)` 内做：

```text
Z_t = (X_t - median_ℓ)/(MAD_ℓ or safe scale).
```

全通道缺失，或任一质量支持特征超出训练增强的 min/max 支持域时，输出 `DATA_QUALITY_UNKNOWN`。全通道缺失行不进入树的中心、尺度和分裂训练。

直接消融 `UD-MAPIC_mask_only_context` 保留 `M_t`，但只使用 missing fraction、gap 和 elapsed；它隔离 P/U 质量治理的增量。传统点插补基线也只使用这个 mask-only context，不能暗中继承提议方法的后验协方差特征。在当前线性高斯实现中，mask-only 消融与 Kalman-mean 基线使用相同滤波、位置分数、上下文和校准链路，二者算法等价；结果相同只是一项实现一致性检查。

## 6. 正确的时序顺序与 UNKNOWN 断点

先按状态标准化，再做 EWMA：

```text
E_t = Z_t,                                  若 t 是支持段首点
E_t = λZ_t + (1-λ)E_{t-1},                 若连续可判定
E_t = -∞ 且清空状态,                       若 UNKNOWN.
```

默认 `λ=0.2`。这避免“旧缺失状态的 EWMA”被当前叶中心/尺度错误解释。

在每个连续可判定片段内，令 `H_t` 为最近 `w` 个 `E` 值中的第 `k` 大值。`H_t>τ` 等价于最近窗口至少 `k` 个 EWMA 值超过 `τ`。UNKNOWN 不仅重置 EWMA，也切断 `k-of-w` 窗口；当 `k<w` 时仍不能拼接缺口前后的证据。默认 `k=w=3`。

结果接口中 `score=X_t`、`standardized_score=Z_t`、`ewma_score=E_t`、`persistence_score=H_t`，`threshold=τ` 作用于 EWMA/持久统计尺度。沿用历史命名的 `raw_alarm` 表示 `E_t>τ`、即应用 `k-of-w` 前的 EWMA 越界，不表示未经处理的 `X_t` 越界。`instantaneous_reference_threshold` 只是把 `τ` 按当前叶中心/尺度反变换得到的一步参考：仅在支持段首点 `E_t=Z_t` 时有直接阈值解释，不能用于复算一般时刻的报警。

## 7. Clustered block conformal 最终校准

对校准 run 的每个预声明缺失机制 `g`、mask 重复 `r` 和基础时间块 `b`，计算最终持久统计的块最大值：

```text
M_bgr = max(H_t in base block b),
E_bg  = max_r M_bgr,
C_b   = max_g E_bg.
```

同一基础健康块的不同 mask 是相关增强，不是独立样本。它们必须先在簇内取最坏值；有效样本量始终是完整基础块数 `B`。不足一个完整 block 的尾段丢弃。

阈值为：

```text
τ = 第 ceil((B+1)(1-α)) 小的 C_b.
```

若 `B < max(min_blocks,ceil(1/α)-1)`，实现直接报错。只有当上游模型完全冻结，并且 `C_1,...,C_B,C_new` 可交换时，标准 split-conformal rank argument 才给出新健康块出现至少一次最终告警的概率不超过 `α`。实现将 UNKNOWN 位置的 `H_t=-∞` 纳入块统计，所以这是包含拒判机制的整块边际告警概率界，并不提供条件于“当前可决策”的 selective false-alarm 保证；coverage 必须另行报告。单条相关或批相非平稳序列通常只满足近似条件。

Rieth 多运行协议使用更严格的运行级版本。对每条独立健康校准运行 `r`，先在同一机制的所有 mask repeats 与全部时间点上取最大值，再在机制间取最大值：

```text
E_rg = max_(repeat,t) H_rgt,
C_r  = max_g E_rg.
```

阈值为 `C_1,...,C_R` 的 conformal quantile，有效样本量是健康运行数 `R`，不是时间块、mask 或机制数。默认 pilot 在 `α=0.05` 下使用 20 条健康校准运行，满足 `R >= ceil(1/α)-1`。若健康运行可交换且整条新运行采用同一冻结预测规则，这给出包含全部预声明机制的运行级边际解释；它仍不保证跨仿真设置、跨工况或未知缺失机制。

## 8. 数据与无泄漏规则

### Batch Distillation

- `Data_interpolated=1` 表示整秒原始 sensor/actuator 数据缺失后由发布方线性插值；训练、校准和测试均恢复成全行未观测。
- 发布插值值可作为完整 archive input 传入，但参数估计、状态树和真实输入都受 observed mask 约束。
- MNAR 人工 mask 从因果重构生成，不读取发布方的双向插值未来信息。
- 自然 timing gap 没有真实值，禁止对这些位置报告 imputation RMSE。
- 插补 MAE/RMSE 只在原本可观测单元上再施加人工 mask 后计算，并按 Phase-I 逐特征尺度标准化。
- 主标签为专家状态 `{2,3}`（可见效应）；`{1,2,3}` 为覆盖原因潜伏、可见效应及原因移除后恢复期的完整非正常 episode，只作次要敏感性分析，不能命名为“原因仍存在”。

### TEP 与其他数据

- 经典 single-realization TEP 已完成 fault 1–21 的工程压力覆盖：每个 fault 使用五个固定场景和三个 mask seeds，共形成 315 个相关 stress views。它用于验证端到端执行、聚合和故障异质性，不把同一轨迹的 mask views 当作独立运行。
- 正式统计主表使用 Rieth multi-run TEP，并以 process run 为外层独立单位。
- MetroPT 重采样保留空时间 bin 为缺失行，不能通过删除空 bin 隐藏 timing gaps。
- 离线线性插值使用未来值，只能标记为 noncausal reference。

## 9. 必做实验矩阵

方法：完整输入同图、UD-MAPIC、`M+U`、mask-only context、Kalman mean、median、LOCF、offline linear interpolation。

缺失：五个预声明场景来自 MCAR、blackout、multi-blackout、MNAR 四类生成机制，其中 multi-blackout 分别覆盖部分通道和全通道 timing gap；固定场景参数后，对每个独立 process run 生成至少 10 个 mask seeds。

指标：逐事件召回、逐事件首次报警延迟、正常期 alarm episodes/100h、正常告警占用率、point precision/recall/F1、决策覆盖、UNKNOWN 比例和人工隐藏单元的标准化插补误差。故障在 UNKNOWN 内仍算未检出。

建议分层汇报：缺失机制 × 缺失率 × fault type × operating condition；置信区间和统计检验以独立 run 聚类，不能以时间点或 mask replicate 伪扩样本量。

## 10. 当前证据与边界

- 40 项测试、Rieth 官方 RData smoke 和断点续跑复核均通过。
- Batch op031 开发结果见 `outputs/batch_op031_ud_v020_elapsedfix.json`：UD 检出 1/2 个事件、延迟 9 s、0 假警；它是单工况案例，不是确认性证据。
- 修复 elapsed-context offset 后的 op028 回顾性结果见 `outputs/batch_op028_confirmation_v020_elapsedfix.json`。UD 在 31 通道和 18-sensor-only 口径均检出 2/4，但分别产生 130.98 和 154.80 个假警 episode/100h；部分基线召回更高但假警也很高。它提示单健康 batch 前/后切分存在跨 run 校准失效，不支持方法优越性。公开标签、文档和先前结果已经被查看，且该协议只有一个健康 run，不能作 untouched 或 run-level 独立确认。
- 经典 TEP fault 1–21 汇总见 [可读报告](../outputs/tep_all_v020/REPORT.md) 与 [聚合 JSON](../outputs/tep_all_v020/aggregate_faults_01_21.json)：UD-MAPIC 为 295/315 view-level 检出、已检出 views 的中位延迟 21、14 个假警 episode（0.572/100h）、平均覆盖 0.9753。fault 3、9 和 15 显示明显困难或延迟；UD 与 mask-only/Kalman mean 的总检出仅差一个 view，不能据此宣称 P/U context 优越。
- 当前 classic TEP 每个 fault 仅有一条过程轨迹且只使用三个 mask seeds；315 是相关 views 数，不是有效样本量，也不满足第 9 节正式矩阵中“每个独立 run 至少 10 个 mask seeds”的确认性设计要求。
- Rieth pilot 使用 20 条健康设计运行、20 条独立健康校准运行，以及 fault 1、3、9、15、19 各 10 条测试运行；50 条运行内共有 750 个嵌套 stress views。UD 检出 268/750，`M+U` 为 180/750，mask-only/Kalman mean 为 280/750。UD 相对 mask-only 的运行级配对召回差为 -0.0160（95% bootstrap CI -0.0240 至 -0.0080），所以当前 P/U context 未获支持；fault 3、9、15 是优先改进对象。
- 下一步是在不查看新 held-out runs 的前提下，用现有 pilot 做诊断与方法冻结，然后扩展到 Rieth fault 1–20、更多独立 runs、每 run 至少 10 个 mask seeds，并补风险—覆盖曲线、计算成本和模型错设敏感性。Rieth 不含当前 classic stress suite 中的 fault 21，应明确报告数据集范围差异。
- `predict` 是整段因果 replay，每次调用重置状态；生产流式部署仍需持久化 `reset/update` API。
- 未知 MNAR、状态空间错设、未覆盖 mask、跨 operating-point 泛化和单序列 block exchangeability 都没有无条件保证。
