# Stanford CS520 与项目 Ontology/知识图谱 对标矩阵

**创建日期**：2025-02-01  
**目的**：逐项对标 Stanford CS520 课程维度与项目 Philosophy/concepts 知识图谱内容  
**依据**：BENCHMARKING_REPORT_Q1_2025 §三待办

---

## 一、维度对标矩阵

| CS520 课程维度 | 项目对应文档 | 对齐度 | 说明 |
|----------------|--------------|--------|------|
| **数据模型** (有向标注图、RDF) | Philosophy/model/01 §2.1 O层、CONCEPT_DECISION_TREE_Ontology | ⚠️ 部分 | 项目 O 层=对象-链接-属性，与 RDF 三元组兼容；未显式引用 RDF 语法 |
| **知识获取** (结构化/非结构化) | concepts/04 认知模拟、Philosophy view01 实施路径 | ⚠️ 部分 | 项目强调 H 层决策血缘捕获，与"从数据获取知识"不同侧重 |
| **推理算法** | Philosophy model/01 L 层逻辑工具、model/03 矩阵2 | ✅ 已对标 | L 层=规则/ML/优化器，与 CS520 推理模块对应 |
| **演化与维护** | Philosophy model/01 §1.3 知识复利、H 层闭环 | ✅ 已对标 | History 层使知识持续进化 |
| **用户交互** | Philosophy model/01 共在、人机协同、model/05 决策树 | ⚠️ 部分 | 项目侧重决策闭环，CS520 侧重检索/问答交互 |
| **应用** (检索、推荐、问答) | view/、Philosophy 案例 | ⚠️ 部分 | 项目侧重企业决策，CS520 侧重通用 KG 应用 |

---

## 二、W3C OWL/RDF 对标

| 标准/规范 | 项目表述 | 对齐度 |
|-----------|----------|--------|
| RDF 三元组 (S,P,O) | O 层 对象-链接-属性 | 兼容，链接≈谓词 |
| OWL 类/属性 | O 层 对象类型、属性 | 兼容，未用 OWL 语法 |
| 推理 (OWL reasoner) | L 层 规则引擎 | 功能等价 |

---

## 三、W3C RDF/OWL 显式语法映射（2025-02 补充）

| W3C 标准 | 项目 O 层/L 层表述 | 显式语法对应 | 规范引用 |
|----------|-------------------|--------------|----------|
| **RDF 三元组** (Subject, Predicate, Object) | O 层 对象-链接-属性 | 对象↔Subject，链接↔Predicate，属性值↔Object | W3C RDF 1.1 Concepts |
| **RDF 图** (有向标注图) | Philosophy model/01 §2.1 O 层 | 节点=对象，边=链接，边标签=谓词类型 | W3C RDF 1.1 |
| **OWL 类** (rdfs:Class, owl:Class) | O 层 对象类型 | 对象类型↔OWL Class | W3C OWL 2 Web Ontology Language |
| **OWL 属性** (owl:ObjectProperty, owl:DatatypeProperty) | O 层 链接、属性 | 链接↔ObjectProperty，属性↔DatatypeProperty | W3C OWL 2 |
| **OWL 推理** (reasoner) | L 层 规则引擎、ML/优化器 | 规则引擎↔OWL reasoner 功能等价 | OWL 2 Direct Semantics |
| **RDF 序列化** (Turtle, RDF/XML) | 项目未规定 | 可选：O 层导出为 Turtle | W3C RDF 1.1 Turtle |

**关联**：Philosophy [model/07 术语表](../Philosophy/model/07-术语表与概念索引.md) Ontology 词条已增 RDF/OWL 引用。

---

## 四、待完善项（2025-02 更新）

| 项 | 状态 | 说明 |
|----|------|------|
| Philosophy model/07 增 RDF、OWL 引用 | ✅ 已闭环 | 07-术语表与概念索引 Ontology 词条 + 标准与规范引用节 |
| W3C 显式语法映射 | ✅ 已闭环 | 本矩阵 §三 已补充 |

---

**维护者**：FormalAI 项目组  
**关联**：[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) KG-01~04、[BENCHMARKING_REPORT_Q1_2025](BENCHMARKING_REPORT_Q1_2025.md)
