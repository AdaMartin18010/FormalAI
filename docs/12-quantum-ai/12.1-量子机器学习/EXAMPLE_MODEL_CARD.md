# 示例模型卡（量子机器学习） / Example Model Card (Quantum Machine Learning)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-Quantum-ML-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产/高风险场景

## 1. 概述 / Overview

- 目标：演示量子机器学习模型的卡片编写要点
- 能力边界：小规模量子电路演示；不提供实际量子优势
- 已知限制：量子噪声模拟简化；无实际量子硬件验证

## 2. 训练数据 / Training Data

- 来源：合成/公开演示数据（示例）
- 许可证：遵守原数据许可；不再分发原始内容
- 代表性：极小规模，不具统计代表性
- 质量控制：基本去重；无污染检测

## 3. 方法与配置 / Methods & Configs

- 架构：变分量子电路（VQC）、量子神经网络（QNN）
- 量子门：RX、RY、RZ、CNOT（示例）
- 优化：经典优化器（Adam）+ 量子期望值估计
- 代码：repo=<https://example.com/formalai-quantum-ml.git>, commit=abc1234

## 4. 评测与性能 / Evaluation & Performance

- 任务：量子分类、量子优化、量子生成建模
- 指标：分类准确率、优化目标值、保真度
- 显著性：95%CI（自助法；示例）
- 复现：`bash eval_quantum.sh --backend simulator --shots 1000`

## 5. 公平/鲁棒/安全 / FRS

- 公平：未做子群体评估
- 鲁棒：量子噪声下的性能保持率（建议）
- 安全：量子密钥分发安全性（建议）

## 6. 风险与使用 / Risks & Usage

- 风险：量子噪声影响；经典模拟限制
- 缓解：仅限教学；需量子硬件验证；禁用于实际应用
- 合规：遵守量子计算相关法规与标准

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
