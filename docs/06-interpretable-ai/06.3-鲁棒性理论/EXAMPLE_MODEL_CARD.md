# 示例模型卡（鲁棒性理论） / Example Model Card (Robustness Theory)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-Robustness-Toolkit-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不可用于实际安全防护

## 1. 概述 / Overview

- 目标：演示鲁棒性检测/增强工具链的模型卡编写要点
- 能力边界：小样本演示；不提供实际安全保证
- 已知限制：攻击覆盖不全；防御机制简化

## 2. 数据 / Data

- 来源：合成/公开演示数据（示例）
- 许可：遵守原数据许可；不含隐私
- 代表性：不具统计代表性

## 3. 方法与配置 / Methods & Configs

- 组件：PGD/FGSM攻击、对抗训练、分布偏移检测
- 防御：数据增强、正则化、不确定性量化（示例）
- 配置：seed=123；报告攻击参数与防御策略

## 4. 评测与性能 / Evaluation & Performance

- 指标：对抗鲁棒性（ASR↓）、分布偏移鲁棒性（性能保持率）、不确定性校准（ECE↓）
- 显著性：差异显著性+效应量报告
- 复现：`bash eval_robust.sh --attacks pgd,fgsm --defenses adv_training,augmentation`

## 5. 公平/鲁棒/安全 / FRS

- 公平：按子群体报告鲁棒性差异
- 鲁棒：重点评估对抗攻击与分布偏移下的性能
- 安全：关注攻击面与防御失效场景

## 6. 风险与使用 / Risks & Usage

- 风险：虚假安全感；防御被绕过
- 缓解：人类在环复核；持续红队测试
- 合规：配合安全标准与监管要求

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
