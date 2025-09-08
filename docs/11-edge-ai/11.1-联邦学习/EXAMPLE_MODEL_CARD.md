# 示例模型卡（联邦学习） / Example Model Card (Federated Learning)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-Federated-Learning-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产/高风险场景

## 1. 概述 / Overview

- 目标：演示联邦学习模型的卡片编写要点
- 能力边界：小规模客户端演示；不提供实际隐私保证
- 已知限制：隐私保护机制简化；通信效率未优化

## 2. 训练数据 / Training Data

- 来源：合成/公开演示数据（示例）
- 许可证：遵守原数据许可；不再分发原始内容
- 代表性：极小规模，不具统计代表性
- 质量控制：基本去重；无污染检测

## 3. 方法与配置 / Methods & Configs

- 架构：FedAvg聚合、差分隐私（ε=1.0）、安全聚合（示例）
- 目标：分类损失+隐私损失
- 训练：客户端数=5, 轮数=10, 本地epoch=1（示例）
- 代码：repo=<https://example.com/formalai-federated-learning.git>, commit=abc1234

## 4. 评测与性能 / Evaluation & Performance

- 任务：联邦分类、隐私保护评估、通信效率分析
- 指标：Acc、隐私预算消耗、通信轮数、收敛速度
- 显著性：95%CI（自助法；示例）
- 复现：docker ghcr.io/formalai/federated-learning:0.1；`bash eval.sh`

## 5. 公平/鲁棒/安全 / FRS

- 公平：按客户端数据分布报告性能差异
- 鲁棒：客户端掉线/恶意客户端下的性能
- 安全：隐私泄露风险评估；差分隐私保证

## 6. 风险与使用 / Risks & Usage

- 风险：隐私泄露；模型投毒；通信开销
- 缓解：仅限教学；需人工监督；禁用于敏感数据
- 合规：遵守数据保护法规与隐私标准

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
