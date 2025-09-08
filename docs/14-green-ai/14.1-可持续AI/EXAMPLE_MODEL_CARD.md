# 示例模型卡（可持续AI） / Example Model Card (Sustainable AI)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-Sustainable-AI-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产/高风险场景

## 1. 概述 / Overview

- 目标：演示可持续AI系统的卡片编写要点
- 能力边界：小规模绿色计算演示；不提供实际碳减排保证
- 已知限制：能耗计算简化；环境影响评估有限

## 2. 训练数据 / Training Data

- 来源：合成/公开演示数据（示例）
- 许可证：遵守原数据许可；不再分发原始内容
- 代表性：极小规模，不具统计代表性
- 质量控制：基本去重；无污染检测

## 3. 方法与配置 / Methods & Configs

- 架构：绿色神经网络、节能优化、碳足迹监控（示例）
- 组件：能耗评估器、效率优化器、环境影响分析器
- 配置：目标能耗<100W，碳足迹<1kg CO2，效率提升>20%（示例）
- 代码：repo=<https://example.com/formalai-sustainable-ai.git>, commit=abc1234

## 4. 评测与性能 / Evaluation & Performance

- 任务：绿色计算、能耗优化、环境影响评估
- 指标：能耗效率、碳足迹、计算性能保持率
- 显著性：95%CI（自助法；示例）
- 复现：`bash eval_sustainable.sh --metrics energy,carbon,performance`

## 5. 公平/鲁棒/安全 / FRS

- 公平：按计算资源/地理区域做差异分析（建议）
- 鲁棒：硬件变化/负载波动下的能耗稳定性（建议）
- 安全：绿色计算的安全性验证（建议）

## 6. 风险与使用 / Risks & Usage

- 风险：虚假绿色声明；性能与环保权衡
- 缓解：仅限教学；需第三方验证；禁用于实际环保认证
- 合规：遵守绿色计算与可持续发展相关标准

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
