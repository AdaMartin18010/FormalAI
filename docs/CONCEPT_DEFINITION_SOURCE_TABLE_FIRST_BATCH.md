# 核心概念定义溯源表（首批）/ Concept Definition Source Table (First Batch)

**创建日期**：2025-02-02
**目的**：为 docs 模块 02.1、03.1、04.1、09.2、07.1 核心概念提供权威定义溯源、项目定义、对齐度与可操作检验
**维护**：与 [AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) 同步更新
**关联**：[concepts/DEFINITION_SOURCE_TABLE](../concepts/DEFINITION_SOURCE_TABLE.md)

---

## 使用说明

| 列 | 说明 |
|----|------|
| 概念 | 该 docs 模块中的核心术语 |
| 权威定义 | 一手文献中的原文或等价表述（含索引编号） |
| 项目定义 | 当前 docs 中的形式化/非形式化表述 |
| 对齐度 | 完全一致 / 扩展 / 冲突 / 待验证 |
| 可操作检验 | 如何验证该定义在具体场景中成立 |

---

## 一、02.1 统计学习理论

| 概念 | 权威定义 | 项目定义 | 对齐度 | 可操作检验 |
|------|----------|----------|--------|------------|
| **PAC 学习** | Valiant (1984); Mohri et al. (2018): 若 $\forall \epsilon,\delta>0$ 存在 $m$ 使 $P[\text{error}(h)\leq\epsilon]\geq 1-\delta$，则概念类 PAC 可学习 [SLT-02][SLT-03] | [02.1.1-PAC学习理论](../02-machine-learning/02.1-统计学习理论/02.1.1-PAC学习理论.md): $\forall \epsilon,\delta>0,\exists m\in\mathbb{N}$, $P[\text{error}(h)\leq\epsilon]\geq 1-\delta$ | 一致 | 给定 $\epsilon,\delta$ 与假设类，验证样本复杂度 $m$ 满足界 |
| **VC 维** | Vapnik-Chervonenkis: 假设类能粉碎的最大样本点数 [SLT-02][SLT-03] | [02.1.2-VC维理论](../02-machine-learning/02.1-统计学习理论/02.1.2-VC维理论.md): 能被子集粉碎的最大样本大小 | 一致 | 对给定假设类构造粉碎集或证明上界 |
| **ERM** | 经验风险最小化：$\hat{h}=\arg\min_{h\in\mathcal{H}} \hat{R}(h)$ [SLT-03] | README §2: ERM 算法，有限/无限假设空间 | 一致 | 实现 ERM 并比较泛化误差与理论界 |
| **Rademacher 复杂度** | Mohri et al.: $\mathfrak{R}_S(\mathcal{H})=\mathbb{E}_\sigma[\sup_{h\in\mathcal{H}}\frac{1}{m}\sum_i \sigma_i h(x_i)]$ [SLT-03] | README §4: Rademacher 复杂度定义与泛化界 | 一致 | 估计 Rademacher 复杂度并验证泛化界 |

**权威引用**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) §2.15 (SLT-01, SLT-02, SLT-03)

---

## 二、03.1 形式化验证

| 概念 | 权威定义 | 项目定义 | 对齐度 | 可操作检验 |
|------|----------|----------|--------|------------|
| **形式化验证** | Clarke et al. (2018): 用数学方法证明系统满足规约 [FV-01] | [03.1 README](../03-formal-methods/03.1-形式化验证/README.md): 通过数学方法证明系统满足其规范 | 一致 | 给出系统模型与规约，产出证明或反例 |
| **模型检测** | Baier & Katoen: 自动验证有穷状态系统是否满足时序逻辑公式 [FV-02] | README §1: 模型检测，状态转移系统 $\mathcal{M}=(S,s_0,\to)$，满足关系 $(\mathcal{M},s)\vDash\varphi$ | 一致 | 对给定 Kripke 结构与 LTL/CTL 公式运行模型检测器 |
| **安全性 (Safety)** | 某坏状态不可达 | README §0: 安全性（Safety）与活性（Liveness） | 一致 | BFS/DFS 可达性分析 |
| **抽象解释** | Cousot & Cousot (1977): 在抽象域上近似不动点 [FV-03] | README 提及抽象解释 | 扩展 | 构造抽象域与伽罗瓦连接，验证不动点包含具体语义 |

**权威引用**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) §2.2 (FV-01~FV-14)

---

## 三、04.1 大型语言模型

| 概念 | 权威定义 | 项目定义 | 对齐度 | 可操作检验 |
|------|----------|----------|--------|------------|
| **大语言模型** | 基于 Transformer 的大规模预训练语言模型，自监督学习 [LLM-01][LLM-02] | [04.1 README](../04-language-models/04.1-大型语言模型/README.md): 基于 Transformer 架构的大规模预训练语言模型，通过自监督学习从大规模文本数据学习表示 | 一致 | 架构检查（Transformer）、训练目标（LM loss）、规模指标（参数量/token） |
| **涌现能力** | Wei et al.: 在规模增大时突然出现的能力 | README 及 LATEST_AI_DEVELOPMENTS：涌现性质 | 扩展 | 随规模扫描能力指标，观察相变 |
| **Agentic LLM** | 规划-执行-验证闭环、工具调用 [LLM-01] | README §2024/2025 进展：Agentic LLM 的规划-执行-验证闭环 | 一致 | 实现规划/执行/验证与工具调用，评估任务完成率 |

**权威引用**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) §2.6 (LLM-01~LLM-08)

---

## 四、09.2 意识理论

| 概念 | 权威定义 | 项目定义 | 对齐度 | 可操作检验 |
|------|----------|----------|--------|------------|
| **意识（形式化）** | Chalmers (1995): 主观体验的“难问题”；IIT: $\Phi>\Phi_{th}$；GWT: 全局工作空间 [CO-01][CO-02][CO-03] | [09.2 README](../09-philosophy-ethics/09.2-意识理论/README.md): $\text{Consciousness}=\text{Subjective Experience}\land\text{Awareness}\land\text{Self-Reference}$ | 扩展 | **非权威标准定义，项目解释**；与 IIT $\Phi$/GWT 可分别检验（见 concepts [CONSCIOUSNESS_THEORY_MATRIX](../concepts/04-AI意识与认知模拟/CONSCIOUSNESS_THEORY_MATRIX.md)） |
| **IIT (Φ)** | Tononi et al. (2016): 整合信息量 $\Phi$ [CO-02] | README 及 concepts/04: $\Phi(S)$ | 扩展 | Φ 计算复杂度高，工程中多用近似或替代指标 |
| **GWT** | Dehaene (2014): 全局工作空间、信息广播 [CO-03] | 全局可访问性、广播 | 一致 | 实验设计：全局可访问性、报告一致性 [CO-05] |
| **机器意识** | 无共识定义；Nature 2025: IIT/GNWT 对抗测试 [CO-04] | README §2: 机器意识定义、实现、测试 | 待验证 | 引用 [CO-04][CO-05][CO-06]；可操作检验依赖具体理论 |

**权威引用**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) §2.5 (CO-01~CO-06)

---

## 五、07.1 对齐理论

| 概念 | 权威定义 | 项目定义 | 对齐度 | 可操作检验 |
|------|----------|----------|--------|------------|
| **对齐 (Alignment)** | 超智能对齐：系统行为符合人类意图/价值 [AL-01]；CS120: 可靠、伦理、对齐的 AI [AL-02] | [07.1 README](../07-alignment-safety/07.1-对齐理论/README.md): AI 系统行为与人类价值观、目标、意图的一致性程度 | 一致 | 偏好/奖励一致性、违规率、人类评估 [AL-01][AL-02] |
| **价值学习** | MS&E338: 奖励学习、CIRL、偏好学习 [AL-01] | README: 价值学习、偏好学习 | 一致 | 奖励模型准确率、策略与人类偏好相关性 |
| **RLHF** | Ouyang et al.: 人类反馈的强化学习微调 | README §2024/2025: RLHF、DPO、Constitutional AI | 一致 | 奖励模型与策略改进指标 |

**权威引用**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) §2.12 (AL-01~AL-05)

---

## 六、02.4 因果推理

| 概念 | 权威定义 | 项目定义 | 对齐度 | 可操作检验 |
|------|----------|----------|--------|------------|
| **do-演算** | Pearl (2000): 三条规则用于因果效应识别；后门/前门准则 [CAUS-01][CAUS-02] | [02.4 README](../02-machine-learning/02.4-因果推理/README.md): 后门/前门调整公式 | 一致 | 给定 DAG，验证 d-分离条件，应用 do-演算规则 |
| **因果效应** | Pearl: $P(y\mid do(x))$ 与 $P(y\mid x)$ 区分 [CAUS-01] | README §4: 平均因果效应 | 一致 | 随机对照或可识别条件下估计 ACE |
| **混淆** | 共同原因导致虚假相关 | 后门路径阻断 | 一致 | 后门准则检验 |

**权威引用**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) §2.17 (CAUS-01, CAUS-02)

---

## 七、更新记录

| 日期 | 更新内容 |
|------|----------|
| 2025-02-02 | 初版创建；覆盖 02.1 PAC/VC/ERM/Rademacher、03.1 形式化验证/模型检测/安全/抽象解释、04.1 LLM/涌现/Agentic、09.2 意识/IIT/GWT、07.1 对齐/价值学习/RLHF |
| 2025-02-02 | 新增 02.4 因果推理（do-演算、因果效应、混淆）；扩展 AL 引用至 AL-05；权威对标与认知优化计划 |

---

**维护者**：FormalAI 项目组
