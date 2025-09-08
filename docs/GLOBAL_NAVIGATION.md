# 全局导航系统 / Global Navigation System

[返回总览](../README.md)

---

## 概述

本文档提供FormalAI项目的全局导航系统，确保所有文档间的导航一致性和完整性。

## 1. 主要主题导航

### 1.1 总览与导航 (0-总览与导航)

- [0.1 全局主题树形目录](./0-总览与导航/0.1-全局主题树形目录.md)
- [0.2 交叉引用与本地跳转说明](./0-总览与导航/0.2-交叉引用与本地跳转说明.md)
- [0.3 持续上下文进度文档](./0-总览与导航/0.3-持续上下文进度文档.md)
- [0.4 现有内容哲科批判分析](./0-总览与导航/0.4-现有内容哲科批判分析.md)

### 1.2 数学基础 (00-foundations)

- [00.0 ZFC公理系统](./00-foundations/00-mathematical-foundations/00-set-theory-zfc.md)
- [00.1 范畴论](./00-foundations/00-mathematical-foundations/01-category-theory.md)
- [00.2 类型理论](./00-foundations/00-mathematical-foundations/02-type-theory.md)
- [00.3 逻辑演算系统](./00-foundations/00-mathematical-foundations/03-logical-calculus.md)
- [00.4 理论依赖关系图](./00-foundations/00-mathematical-foundations/04-theory-dependency-graph.md)
- [00.5 形式化证明](./00-foundations/00-mathematical-foundations/05-formal-proofs.md)

### 1.3 基础理论 (01-foundations)

- [01.1 形式逻辑](./01-foundations/01.1-形式逻辑/README.md)
- [01.2 数学基础](./01-foundations/01.2-数学基础/README.md)
- [01.3 计算理论](./01-foundations/01.3-计算理论/README.md)
- [01.4 认知科学](./01-foundations/01.4-认知科学/README.md)

### 1.4 机器学习理论 (02-machine-learning)

- [02.1 统计学习理论](./02-machine-learning/02.1-统计学习理论/README.md)
  - [02.1.1 PAC学习理论](./02-machine-learning/02.1-统计学习理论/02.1.1-PAC学习理论.md)
  - [02.1.2 VC维理论](./02-machine-learning/02.1-统计学习理论/02.1.2-VC维理论.md)
- [02.2 深度学习理论](./02-machine-learning/02.2-深度学习理论/README.md)
  - [02.2.1 神经网络理论](./02-machine-learning/02.2-深度学习理论/02.2.1-神经网络理论.md)
- [02.3 强化学习理论](./02-machine-learning/02.3-强化学习理论/README.md)
- [02.4 因果推理](./02-machine-learning/02.4-因果推理/README.md)

### 1.5 形式化方法 (03-formal-methods)

- [03.1 形式化验证](./03-formal-methods/03.1-形式化验证/README.md)
- [03.2 程序综合](./03-formal-methods/03.2-程序综合/README.md)
- [03.3 类型理论](./03-formal-methods/03.3-类型理论/README.md)
- [03.4 证明系统](./03-formal-methods/03.4-证明系统/README.md)

### 1.6 语言模型 (04-language-models)

- [04.1 大型语言模型](./04-language-models/04.1-大型语言模型/README.md)
- [04.2 形式语义](./04-language-models/04.2-形式语义/README.md)
- [04.3 知识表示](./04-language-models/04.3-知识表示/README.md)
- [04.4 推理机制](./04-language-models/04.4-推理机制/README.md)
- [04.5 AI代理](./04-language-models/04.5-AI代理/README.md)

### 1.7 多模态AI (05-multimodal-ai)

- [05.1 视觉语言模型](./05-multimodal-ai/05.1-视觉语言模型/README.md)
- [05.2 多模态融合](./05-multimodal-ai/05.2-多模态融合/README.md)
- [05.3 跨模态推理](./05-multimodal-ai/05.3-跨模态推理/README.md)

#### 1.7.1 快速索引与依赖指引（05.1–05.3）

- 进入顺序：
  1) [05.1 视觉语言模型](./05-multimodal-ai/05.1-视觉语言模型/README.md) → 概念、对齐、InfoNCE、泛化界、安全与幻觉界、评测YAML、术语表（0.16）
  2) [05.2 多模态融合](./05-multimodal-ai/05.2-多模态融合/README.md) → 融合代数、稳定性与可分性、SMT一致性片段、融合稳定性YAML
  3) [05.3 跨模态推理](./05-multimodal-ai/05.3-跨模态推理/README.md) → 语义/语法、正确性/完备性、SAT/SMT一致性、推理安全界

- 关键交叉引用：
  - 术语与符号表锚点：05.1 的 [0.16 术语与符号表](./05-multimodal-ai/05.1-视觉语言模型/README.md#016-术语与符号表--terminology-and-notation)
  - 运行时安全与回退：05.1 的 0.15/0.19；与 05.3 的 0.4/0.5 协同
  - RAG+SMT：05.1 的 0.13 与 05.3 的 0.3 配套

- 评测与配置：
  - 多任务评测（05.1：0.7/0.10/0.11/0.18）
  - 融合稳定性评测（05.2：评测YAML → ./05-multimodal-ai/05.2-多模态融合/README.md#融合稳定性评测配置yaml）
  - 推理一致性与显著性（05.3：评测配置 → ./05-multimodal-ai/05.3-跨模态推理/README.md#评测配置一致性与显著性yaml）

### 1.8 可解释AI (06-interpretable-ai)

- [06.1 可解释性理论](./06-interpretable-ai/06.1-可解释性理论/README.md)
- [06.2 公平性与偏见](./06-interpretable-ai/06.2-公平性与偏见/README.md)
- [06.3 鲁棒性理论](./06-interpretable-ai/06.3-鲁棒性理论/README.md)

### 1.9 对齐与安全 (07-alignment-safety)

- [07.1 对齐理论](./07-alignment-safety/07.1-对齐理论/README.md)
- [07.2 价值学习](./07-alignment-safety/07.2-价值学习/README.md)
- [07.3 安全机制](./07-alignment-safety/07.3-安全机制/README.md)

### 1.10 涌现与复杂性 (08-emergence-complexity)

- [08.1 涌现理论](./08-emergence-complexity/08.1-涌现理论/README.md)
- [08.2 复杂系统](./08-emergence-complexity/08.2-复杂系统/README.md)
- [08.3 自组织](./08-emergence-complexity/08.3-自组织/README.md)

### 1.11 哲学与伦理 (09-philosophy-ethics)

- [09.1 AI哲学](./09-philosophy-ethics/09.1-AI哲学/README.md)
- [09.2 意识理论](./09-philosophy-ethics/09.2-意识理论/README.md)
- [09.3 伦理框架](./09-philosophy-ethics/09.3-伦理框架/README.md)

### 1.12 具身AI (10-embodied-ai)

- [10.1 具身智能](./10-embodied-ai/10.1-具身智能/README.md)

### 1.13 边缘AI (11-edge-ai)

- [11.1 联邦学习](./11-edge-ai/11.1-联邦学习/README.md)

### 1.14 量子AI (12-quantum-ai)

- [12.1 量子机器学习](./12-quantum-ai/12.1-量子机器学习/README.md)

### 1.15 神经符号AI (13-neural-symbolic)

- [13.1 神经符号AI](./13-neural-symbolic/13.1-神经符号AI/README.md)

### 1.16 绿色AI (14-green-ai)

- [14.1 可持续AI](./14-green-ai/14.1-可持续AI/README.md)

### 1.17 元学习 (15-meta-learning)

- [15.1 元学习理论](./15-meta-learning/15.1-元学习理论/README.md)

## 2. 快速导航

### 2.1 按主题类型

**数学基础**:

- [ZFC公理系统](./00-foundations/00-mathematical-foundations/00-set-theory-zfc.md)
- [范畴论](./00-foundations/00-mathematical-foundations/01-category-theory.md)
- [类型理论](./00-foundations/00-mathematical-foundations/02-type-theory.md)

**机器学习**:

- [PAC学习理论](./02-machine-learning/02.1-统计学习理论/02.1.1-PAC学习理论.md)
- [神经网络理论](./02-machine-learning/02.2-深度学习理论/02.2.1-神经网络理论.md)

**形式化方法**:

- [形式化验证](./03-formal-methods/03.1-形式化验证/README.md)
- [程序综合](./03-formal-methods/03.2-程序综合/README.md)

### 2.2 按学习路径

**初学者路径**:

1. [0.1 全局主题树形目录](./0-总览与导航/0.1-全局主题树形目录.md)
2. [00.0 ZFC公理系统](./00-foundations/00-mathematical-foundations/00-set-theory-zfc.md)
3. [02.1.1 PAC学习理论](./02-machine-learning/02.1-统计学习理论/02.1.1-PAC学习理论.md)

**进阶路径**:

1. [00.1 范畴论](./00-foundations/00-mathematical-foundations/01-category-theory.md)
2. [00.2 类型理论](./00-foundations/00-mathematical-foundations/02-type-theory.md)
3. [02.2.1 神经网络理论](./02-machine-learning/02.2-深度学习理论/02.2.1-神经网络理论.md)

**专家路径**:

1. [00.4 理论依赖关系图](./00-foundations/00-mathematical-foundations/04-theory-dependency-graph.md)
2. [03.1 形式化验证](./03-formal-methods/03.1-形式化验证/README.md)
3. [09.1 AI哲学](./09-philosophy-ethics/09.1-AI哲学/README.md)

## 3. 交叉引用索引

### 3.1 理论依赖关系

**基础依赖**:

- ZFC公理系统 → 范畴论 → 类型理论 → 逻辑演算系统

**应用依赖**:

- 统计学习理论 → 深度学习理论 → 神经网络理论
- 形式化验证 → 程序综合 → 证明系统

### 3.2 相关主题

**数学与AI**:

- [ZFC公理系统](./00-foundations/00-mathematical-foundations/00-set-theory-zfc.md) ↔ [PAC学习理论](./02-machine-learning/02.1-统计学习理论/02.1.1-PAC学习理论.md)
- [范畴论](./00-foundations/00-mathematical-foundations/01-category-theory.md) ↔ [神经网络理论](./02-machine-learning/02.2-深度学习理论/02.2.1-神经网络理论.md)

**理论与应用**:

- [类型理论](./00-foundations/00-mathematical-foundations/02-type-theory.md) ↔ [形式化验证](./03-formal-methods/03.1-形式化验证/README.md)
- [逻辑演算系统](./00-foundations/00-mathematical-foundations/03-logical-calculus.md) ↔ [程序综合](./03-formal-methods/03.2-程序综合/README.md)

## 4. 搜索与索引

### 4.1 关键词索引

**数学概念**:

- 集合论、范畴论、类型理论、逻辑演算

**AI概念**:

- 机器学习、深度学习、神经网络、强化学习

**形式化概念**:

- 验证、综合、证明、类型安全

### 4.2 主题索引

**基础理论**:

- 数学基础、计算理论、认知科学

**应用理论**:

- 机器学习、形式化方法、语言模型

**前沿理论**:

- 量子AI、神经符号AI、元学习

## 5. 更新日志

### 5.1 版本历史

- **v2025-01**: 初始版本，建立基础导航结构
- **v2025-01.1**: 添加机器学习理论导航
- **v2025-01.2**: 完善交叉引用系统

### 5.2 最近更新

- 2025-01-01: 创建全局导航系统
- 2025-01-01: 添加快速导航功能
- 2025-01-01: 建立交叉引用索引

## 6. 对齐标准与权威来源（2025）

为确保内容持续对齐至2025年国际最成熟的理论、模型与技术，并统一引用口径，项目遵循如下权威来源体系（示例性锚点，随“最近更新”滚动同步）：

- 国际知识库与标准
  - Wikipedia/Wikidata：概念与术语的跨语种锚点（审慎核验一手论文）
  - arXiv/ACL Anthology/IEEE Xplore/ACM DL：论文一手来源
  - NIST、ISO/IEC JTC 1、W3C：评测基准、术语标准与技术规范

- 顶尖大学课程（2024/2025学季代表）
  - MIT、Stanford、Carnegie Mellon、Berkeley、Harvard 的公开课与课程讲义（如深度学习理论、统计学习、形式化方法、LLM、RL、因果推断）
  - 选取近两学年的最新授课大纲与作业/项目要求作为教学对齐参考

- 权威会议与期刊（A类优先）
  - 机器学习与AI：NeurIPS、ICML、ICLR、AAAI、IJCAI、KDD、WWW
  - 计算机视觉与多模态：CVPR、ICCV、ECCV
  - NLP与语言模型：ACL、EMNLP、NAACL、COLING
  - 系统安全与验证：CAV、POPL、PLDI、CSF、S&P、CCS、USENIX Security

- 长期综述与基准
  - Survey/Review/Position：系统性综述与领域蓝皮书作为稳定锚点
  - Benchmarks/Leaderboards：遵循公开可复现评测及数据卡/模型卡规范

- 更新与治理
  - 在 `docs/LATEST_UPDATES_INDEX.md` 维护“年度权威索引”，提供课程、论文、基准与标准的链接清单
  - 在各主题 `README.md` 以“参考/进一步阅读”区块落地至一手来源
  - 最小合规核对单：见 `docs/STANDARDS_CHECKLISTS.md`（模型卡/数据卡/评测卡）

注：如二手资料与一手论文不一致，以一手论文与标准规范为准。

---

**最后更新**：2025-01-01  
**版本**：v2025-01  
**维护者**：FormalAI项目组
