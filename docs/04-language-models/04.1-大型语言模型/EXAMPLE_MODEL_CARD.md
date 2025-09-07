# 示例模型卡（LLM） / Example Model Card (LLM)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-LLM-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产、敏感或高风险场景

## 1. 概述 / Overview

- 目标：作为课程/文档示例，展示模型卡最佳实践与合规模板
- 能力边界：少量中文/英文文本生成；不保证事实性与安全性
- 已知限制：可能产生幻觉；缺乏面向安全的对抗防护

## 2. 训练数据 / Training Data

- 来源：公开演示文本片段（示例）
- 许可证：遵守原数据源许可；不再分发原始内容（示例）
- 代表性：极小规模，不具代表性
- 质量控制：基本去重；未进行污染检测（示例）

## 3. 方法与配置 / Methods & Configs

- 架构：微型Transformer（示例）
- 配置：batch=8, lr=3e-4, epoch=1（示例）
- 代码：repo=<https://example.com/formalai-llm-example.git>, commit=abc1234（示例）

## 4. 评测与性能 / Evaluation & Performance

- 任务：简单文本补全（示例）
- 指标：BLEU/ROUGE（示例）
- 统计：未做显著性检验（示例）
- 复现：docker image formalai/example:0.1；run.sh（示例）
- 对比：与小型基线TinyLM（示例）

## 5. 公平/鲁棒/安全 / FRS

- 公平：未进行子群体评估（示例）
- 鲁棒：未进行对抗/分布外评测（示例）
- 安全：未进行红队；提示注入/越狱未防护（示例）

## 6. 风险与使用 / Risks & Usage

- 风险：误导性输出、偏见、隐私泄露风险（示例）
- 缓解：仅限教学；需人工审查；禁用于决策（示例）
- 法律/伦理：遵守地区法律与平台政策（示例）

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A（示例）
- 反馈：<issues@example.com>（示例）

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
