# 示例模型卡（具身智能） / Example Model Card (Embodied Intelligence)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-Embodied-Intelligence-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产/高风险场景

## 1. 概述 / Overview

- 目标：演示具身智能系统的卡片编写要点
- 能力边界：小规模机器人/虚拟具身演示；不提供实际物理操作
- 已知限制：物理模拟简化；感知能力有限

## 2. 训练数据 / Training Data

- 来源：合成/公开演示数据（示例）
- 许可证：遵守原数据许可；不再分发原始内容
- 代表性：极小规模，不具统计代表性
- 质量控制：基本去重；无污染检测

## 3. 方法与配置 / Methods & Configs

- 架构：感知-行动循环、具身认知模型、物理世界模拟（示例）
- 组件：多模态感知、运动控制、环境交互
- 配置：传感器数=8，动作维度=6，环境复杂度=中等（示例）
- 代码：repo=<https://example.com/formalai-embodied-intelligence.git>, commit=abc1234

## 4. 评测与性能 / Evaluation & Performance

- 任务：具身导航、物体操作、环境理解
- 指标：任务完成率、感知准确率、动作精度
- 显著性：95%CI（自助法；示例）
- 复现：`bash eval_embodied.sh --task navigation,manipulation,understanding`

## 5. 公平/鲁棒/安全 / FRS

- 公平：按环境类型/任务复杂度做差异分析（建议）
- 鲁棒：环境变化/传感器故障/物理约束下的性能（建议）
- 安全：具身系统的物理安全与操作安全（建议）

## 6. 风险与使用 / Risks & Usage

- 风险：物理操作失误；环境理解错误
- 缓解：仅限教学；需人工监督；禁用于实际物理操作
- 合规：遵守机器人安全与AI伦理相关标准

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
