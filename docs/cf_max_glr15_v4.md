# CF-Max-GLR15 v4：嵌套校准的故障特异检测协议

## 1. 目标与状态

v4 针对 v3 的两个已观测问题：单一全局 detector 对 Fault 15 分离度不足，以及使用模型拟合健康运行本身定阈值造成 held-in 乐观偏差。v4 不增加 TCN 容量，而采用三个低参数、机制特异的 one-vs-healthy 专家，并在每个开发外折中保留独立健康校准折。

本文件是方法与治理预注册。创建代码和运行合成测试不等于正式实验；只有显式运行开发命令才读取 magnitude-50 数值。25% 校准集和 75% 确认集在正式协议冻结之前不得读取。25% 故障分支和全部 magnitude-100 单元始终禁止进入本协议。

## 2. 固定方法

### 2.1 因果专家特征

features.py 从每个 500×52 运行直接生成三个只读 float64 路径：

- Fault 3：500×20；
- Fault 9：500×20；
- Fault 15：500×28。

特征 schema 必须精确为 cf_glr_mechanism_features_v2。Fault 3/9 使用覆盖全部 52 个过程通道的固定 top-5 汇总，Fault 15 使用控制回路机制特征。行 [0,128) 固定为零；[128,160) 只使用当时及过去信息并为 prefix-negative loss 提供输入；监控区间为 [160,500)。任何未来行依赖均属协议错误。

模型内部的逐专家特征 median/MAD 标准化只从该模型的 fit role 中估计，同时使用数量平衡的健康与对应目标故障运行。中心、尺度与网络权重一起安全持久化；校准或测试运行不得重估。

### 2.2 三个独立专家

- Fault 3：20→1 线性 logit，seed 20260909；
- Fault 9：20→1 线性 logit，seed 20260909；
- Fault 15：三套 28→16→1 GELU 网络，seeds 20260909、20260910、20260911；最终逐端点算术平均 logits。

每个专家独立以 IDV1 为负类、其目标故障为正类。主 MIL endpoints 为 160,164,…,496，共 85 点；bag score 是最高 10%，即 9 点均值。每个 expert 的主损失为健康和故障 bag BCE 各 0.5，加权 0.1 的 prefix-negative loss（endpoints 128,132,…,156，共 8 点），再加 1e-4 显式 L2。优化固定为 full-batch CUDA float64 AdamW，80 epochs，学习率 1e-3，optimizer weight decay 为零，无 gradient clipping、validation、early stopping 或候选搜索。

### 2.3 健康标准化、融合与报警

每个外折只以 60 个 fit-role 健康运行在监控区间内的所有专家 logits 拟合每专家全局 median/MAD。给定三个标准化 logits：

U_t = max_k Z_(k,t)

P_t = min(U_(t-1), U_t)

运行统计量为合法监控点上的 R = max_t P_t。全家族融合发生在定阈值之前，因此一个健康校准最大值同时涵盖三个专家、监控时间点和两点持续性，不另做逐专家多重比较。

第一次满足 P_t > h 的位置为报警点；比较必须严格为大于。归因取该位置标准化 logit 最大的专家；完全相等时固定列顺序 (3,9,15) 使较小 fault id 获胜。归因只用于报告，不替代主检测门槛。

## 3. 五折三向嵌套开发

run id 1–100 以固定 salt cf-max-glr15-v4-mode1-mag50-three-way-2026-09-08 做 SHA-256 排序，再循环分配到五个各 20 个 run block 的折。四个 IDV 条件始终共享同一个 run-block 角色。

对外折 j：

- test 为 F_j，20 runs/class；
- calibration 为 F_(j+1 mod 5)，只允许其中 20 个 IDV1 定阈值；该折 60 个故障运行明确排除；
- fit 为其余三个折，60 runs/class。

每折重新拟合全部专家、模型内特征标准化和健康 logit 标准化。独立 calibration 健康运行的 20 个运行最大值以 rank-20（最大值）定阈值，并对 test 严格使用大于。要求至少 19 个按 CPU/CUDA parity tolerance 可区分的最大值，rank20 大于 rank19，且相邻尾 gap 大于 100×max(1e-12, parity error)；否则唯一候选为 NO_GO。

每个开发运行只在作为外折 test 时产生 OOF 评价。test 健康或故障值不得用于该折的特征标准化、训练、monitor 标准化或阈值。没有超参数候选选择；candidate id 固定为 cf-max-glr15-fixed。

## 4. 开发门槛

门槛与 v3 原始预注册一致：

- 100 个 OOF 健康运行报警不超过 10，每折不超过 4；
- Fault 3、9、15 各自总检出率至少 0.85；
- 每个 fold×fault 检出率至少 0.70；
- 每个 fault 的 median delay 不超过 140；
- 每个 fault 的 nearest-rank p90 delay 不超过 220。

miss delay 固定为 340。任一嵌套校准非退化检查、数值有限性、CPU/CUDA parity 或上述门槛失败均为 DEVELOPMENT_NO_GO，且不得生成模型或 monitor。

若全部通过，才以全部 100 runs/class 的 magnitude-50 开发集重训一次固定模型，并仅用其中 100 个健康 logits 拟合最终 monitor standardizer。安全 NPZ 内含网络状态、逐专家特征中心/尺度、完整配置和状态哈希；独立 canonical JSON 持久化 monitor。二者及全部实现文件均由 SHA-256 绑定。

## 5. 正式校准和确认

正式 freeze 只接受完整、哈希一致且结果为 PASS 的开发 iteration。freeze 不读取 HDF5 数值，但验证官方 Mode1 文件：

- size：23,924,710,848 bytes；
- MD5：c79ecc6d12a5ccb49caffcd8f531feae；
- SHA-256：d6b0ed0ef3974ff571afd6717ba0418a5b8eb58a7822c84a05735572d52b3294。

freeze 通过 output-root claim、canonical protocol JSON、SHA-256 sidecar 和 commit receipt 原子绑定源文件、配置、代码、PASS 结果、模型与 monitor。冻结后任何绑定文件变化都会 fail closed。

### 5.1 一次性 25% 校准

只读取 Mode1 magnitude-25 IDV1 runs1–100 的 pre-fault 500-row 健康窗口。模型、特征 standardizer 和 monitor standardizer 都只加载、不重拟合。100 个全家族运行最大值的 ascending rank-98 为正式阈值，后续严格使用大于。要求 100 个有限值、按 parity tolerance 至少 96 个 distinct，并满足 rank98 小于 rank100。其分布无关边际超限界为 3/101；不能宣称小于 0.05。

校准无论 PASS 或 INVALID 都由 completion receipt 标记为已消费。INVALID 校准永久阻止确认。

### 5.2 一次性 75% 确认

只在校准 PASS 后读取 Mode1 magnitude-75 IDV1/3/9/15 runs1–100，一次性使用冻结阈值。确认门槛：

- 健康报警不超过 4/100；
- 每个 fault 至少检出 90/100；
- 每个 fault median delay 不超过 140；
- 每个 fault nearest-rank p90 delay 不超过 220。

确认结果无论 PASS 或 FAIL 都是最终一次性结果，不允许再回到 25% 或 75% 调参。

## 6. 工程治理与运行

runner 顶层只允许 Python 标准库；numeric backend 必须先 import Torch，再 import NumPy、data-v2、features/model/monitor。所有 numeric phase 必须为 CUDA float64。backend 调用前，runner 必须已用 fsync 写入 access intent；每个 phase 以排他目录或文件 claim 开始，以绑定全部 artifact 哈希的 completion receipt 结束。中断产生的 partial claim 可见且不得静默覆盖。

开发命令示例：

    conda run -n torch5090 python scripts/run_cf_max_glr15_gpu_v4.py development-iteration --data-dir data/extended_tep_dtu_mode1 --iteration-id cf-max-glr15-v4-dev01

只有开发 PASS 且人工审阅结果后才运行 freeze-formal；之后依次运行 calibrate 和 confirm。不得跳过阶段或直接调用 backend 绕过治理 runner。

v3 正式开发迭代在同一 RTX 5090 Laptop GPU 上约 5 分 08 秒。v4 网络明显更小且每个 phase 只提取一次固定专家特征；实际 wall time 应由首次开发 receipt 记录，不在看见真实结果之前修改协议。
