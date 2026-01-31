# 概念-属性-关系三元组索引

**创建日期**：2025-02-01  
**目的**：哲科批判对齐——建立「概念-属性-关系」三元组索引，支撑属性关系与解释论证  
**依据**：COMPREHENSIVE_CRITICAL_ANALYSIS_AND_ADVANCEMENT_PLAN_2025 §2.4 属性关系  
**关联**：[DEFINITION_SOURCE_TABLE](../concepts/DEFINITION_SOURCE_TABLE.md)、[PROJECT_CROSS_MODULE_MAPPING](../PROJECT_CROSS_MODULE_MAPPING.md)

---

## 一、使用说明

| 列 | 说明 |
|----|------|
| 概念 | 项目或权威文献中的核心概念 |
| 属性 | 该概念的形式化或可观测属性 |
| 关系 | 与其他概念/属性的关系类型及目标 |
| 溯源 | 定义或关系出处（DEFINITION_SOURCE_TABLE / 矩阵 / 证明树） |

---

## 二、三元组索引（按 concepts 主题）

### 2.1 03-Scaling Law 与收敛分析

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| Scaling Law | 幂指数 α、常数 a,b | 与参数量 N 满足 $L(N)=a N^{-\alpha}+b$ | DEFINITION_SOURCE_TABLE §一 |
| Chinchilla 最优 | $D_{opt}/N$ 比、指数 0.74 | $D_{opt}\propto N^{0.74}$，约 20 tokens/param | DEFINITION_SOURCE_TABLE §一；CHINCHILLA_VERIFICATION_APPENDIX |
| 计算最优训练 | 推理成本、训练 token 长度 | 固定推理预算下成本-性能比 | concepts/03 §11.2.14（Sardana） |
| Kaplan vs Hoffmann | last layer cost、warmup、optimizer | 差异归因三因素 | concepts/03 §11.2.15（Porian） |

### 2.2 04-AI 意识与认知模拟

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| Qualia | 主观体验度 $Q(S)$ | 无直接可操作检验（难问题） | DEFINITION_SOURCE_TABLE §二 |
| IIT (Φ) | 整合信息量 Φ | $\Phi>\Phi_{th}$ → 有意识 | DEFINITION_SOURCE_TABLE §二；CONSCIOUSNESS_THEORY_MATRIX |
| GWT | 全局广播容量 | $C(S)=\text{全局广播}(S)$ | DEFINITION_SOURCE_TABLE §二 |
| GWT+HOT | 广播容量 + 质量控制 | 互补；Phua et al. 2025 | concepts/04 §10.9；CONSCIOUSNESS_THEORY_MATRIX |
| 意识度 $C(S)$ | 权重 $w_i$（项目假设） | 多理论加权组合 | concepts/04 §3.1 |

### 2.3 01-AI 三层模型架构

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| 执行层 | 七元组 $M=(Q,\Sigma,\Gamma,\delta,q_0,B,F)$ | 图灵完备；与数据层/控制层接口 | DEFINITION_SOURCE_TABLE §三 |
| 控制层 | 形式文法、λ演算、ReAct | 约束数据层输出 | DEFINITION_SOURCE_TABLE §三 |
| 数据层 | 概率模型、Transformer、loss | 依赖执行层计算 | DEFINITION_SOURCE_TABLE §三 |
| 图灵完备性 | 可计算性 | 等价于图灵机可计算 | DEFINITION_SOURCE_TABLE §三 |

### 2.4 02-AI 炼金术转化度模型

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| 转化度 | Level 1–5、可交付物清单 | 从黑箱到形式验证的层级 | DEFINITION_SOURCE_TABLE §四；CONCEPT_DECISION_TREE_转化度 |
| 五维度 D₁~D₅ | 理论完备性、可复现性、商业化、可解释性、自我改进 | 与转化度层级联动 | DEFINITION_SOURCE_TABLE §四 |

### 2.5 05-AI 科学理论

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| 涌现 | 半可预测性、临界点 | 依赖数据层；Scaling Law 预测 | DEFINITION_SOURCE_TABLE §五；CONCEPT_MATRIX_科学理论 |
| RLHF | 奖励模型、策略改进 | 人类偏好 → 强化学习微调 | DEFINITION_SOURCE_TABLE §五 |
| CoT | 多步推理、中间步骤 | 推断时间计算 | DEFINITION_SOURCE_TABLE §五 |

### 2.6 06-AI 反实践判定系统

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| 反实践 | 违背科学/工程原则 | 判定清单、案例匹配 | DEFINITION_SOURCE_TABLE §六 |
| 可判定性 | 算法停机且输出正确 | 图灵 1936；与哥德尔边界 | DEFINITION_SOURCE_TABLE §六 |
| 哥德尔不完备 | 一致形式系统存在不可证真命题 | 价值对齐不可判定 | DEFINITION_SOURCE_TABLE §六 |

### 2.7 07-AI 框架批判与重构

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| 本体论暴政 | 单一本体强加 | 多本体兼容性 | DEFINITION_SOURCE_TABLE §七；CRITIQUE_DIMENSION_MATRIX |
| 神经算子 | 算子学习、PDE 解 | 与 FNO/DeepONet 对比 | DEFINITION_SOURCE_TABLE §七 |

### 2.8 08-AI 历史进程与原理演进

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| 符号主义 | 物理符号系统假设 | Russell & Norvig | DEFINITION_SOURCE_TABLE §八 |
| 联结主义 | 分布式表征、并行处理 | Rumelhart | DEFINITION_SOURCE_TABLE §八 |

### 2.9 Philosophy-Ontology/DKB

| 概念 | 属性 | 关系 | 溯源 |
|------|------|------|------|
| Ontology | O 层：对象、链接、属性 | W3C OWL 形式化；与 RDF/OWL 对标 | DEFINITION_SOURCE_TABLE §九；STANFORD_CS520 |
| DKB | $(O,L,H)$ 三元组 | ARI、HR 实证 | DEFINITION_SOURCE_TABLE §九；AXIOM_THEOREM_INFERENCE_TREE |
| Phronesis | H 层隐性知识 | 决策血缘覆盖率 | DEFINITION_SOURCE_TABLE §九 |
| ARI 指数 | 语义对齐率、逻辑封装、闭环系数 | 与 Scaling Law/收敛关联 | DEFINITION_SOURCE_TABLE §九；view02 |

---

## 三、与证明树/矩阵的链接

| 表征 | 路径 | 说明 |
|------|------|------|
| 公理-定理推理树 | Philosophy/model/AXIOM_THEOREM_INFERENCE_TREE.md | A1–A6 → L1–L4 → T1–T9；前提→结论 |
| 意识理论矩阵 | concepts/04/CONSCIOUSNESS_THEORY_MATRIX.md | IIT/GWT/HOT/RCUET/MCT/RIIU 属性对比 |
| 收敛 L0–L4 矩阵 | concepts/03/CONVERGENCE_L0_L4_MATRIX.md | 收敛策略层级与属性 |
| 科学理论矩阵 | concepts/05/CONCEPT_MATRIX_科学理论.md | 涌现/RLHF/CoT 多维度 |
| 批判维度矩阵 | concepts/07/CRITIQUE_DIMENSION_MATRIX.md | 批判维度 vs 严重性、重构方案 |

---

## 四、更新记录

| 日期 | 更新内容 |
|------|----------|
| 2025-02-01 | 初版创建；覆盖 concepts 01–08 + Philosophy 核心概念-属性-关系三元组 |

---

**维护者**：FormalAI 项目组
