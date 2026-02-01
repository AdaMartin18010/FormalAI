# LATEST_AI_DEVELOPMENTS 条目编写规范 / Entry Specification

**创建日期**：2025-02-02
**目的**：规范 [LATEST_AI_DEVELOPMENTS_2025](LATEST_AI_DEVELOPMENTS_2025.md) 每条模型/技术条目的必填字段，确保数据可溯源、可验证
**维护**：与 [AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) 联动

---

## 一、每条条目必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| **数据来源** | 一手来源（官方技术报告、论文、发布会） | OpenAI 官方技术报告；arXiv 2401.00448 |
| **截止日期** | 数据或引用的截止日期 | 2024-12-XX；2025-01-15 |
| **定量指标**（若可得） | 参数量、token 数、基准分数、推理成本等 | 参数量 70B；MATH ~90%；20 tokens/param |
| **AUTHORITY 编号** | [AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) 中的引用编号 | [LLM-01]；[SL-01][SL-02]；[DL-06][DL-07] |

---

## 二、推荐结构（每条模型/技术）

```markdown
#### 模型/技术名称（年份）

**发布时间**：YYYY年MM月 或 YYYY年

**核心特点**：（1~3 句）

**定量指标**：
- 参数量：（若公开或可推测，注明“未公开”或范围）
- 训练数据：（token 数/数据量，若可得）
- 基准分数：（MATH、HumanEval、GSM8K 等，若可得）
- 其他：（上下文长度、推理成本等）

**数据来源**：（必填）具体来源
**截止日期**：（必填）YYYY-MM 或 YYYY-MM-DD
**权威引用**：（必填）[AUTHORITY_REFERENCE_INDEX](AUTHORITY_REFERENCE_INDEX.md) 编号，如 [LLM-01]、[SL-01]
```

---

## 三、Scaling Law 相关条目

涉及 Scaling Law、Chinchilla、推理成本扩展的条目须显式引用：

- **Chinchilla 最优**：[SL-01] Hoffmann et al. (2022)
- **推理成本扩展**：[SL-02] Sardana et al. (2024), arXiv 2401.00448
- **Kaplan vs Hoffmann 差异**：[SL-03] Porian et al. (2024), NeurIPS 2024

定量指标若引用上述论文，须标注公式或图表出处（如“与论文 Figure 1 一致”）。

---

## 四、季度检查

- 每季度（见 [QUARTERLY_UPDATE_CHECKLIST](QUARTERLY_UPDATE_CHECKLIST.md)）检查：
  - [ ] 所有条目是否包含“数据来源”“截止日期”“权威引用”；
  - [ ] 过期数据是否更新或标注“历史数据”；
  - [ ] 新增条目是否按本规范填写。

---

**维护者**：FormalAI 项目组
