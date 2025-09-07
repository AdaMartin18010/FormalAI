# 示例模型卡（公平性与偏见） / Example Model Card (Fairness & Bias)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型/系统名称：FormalAI-Fairness-Toolkit-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不可用于实际合规审计

## 1. 概述 / Overview

- 目标：演示公平性检测/缓解工具链的模型卡编写要点
- 能力边界：小样本演示；不提供法律/监管保证
- 已知限制：子群体覆盖不足；指标不完备

## 2. 数据 / Data

- 来源：合成/公开演示数据（示例）
- 许可：遵守原数据许可；不含隐私
- 代表性：不具统计代表性

## 3. 方法与配置 / Methods & Configs

- 组件：统计检测（DP/EO/PRP）、个体一致性、简单因果图近似
- 缓解：重采样/阈值调整/正则化（示例）
- 配置：seed=123；报告敏感属性清单

## 4. 评测与性能 / Evaluation & Performance

- 指标：群体公平（DP/EO/PRP）、个体一致性分、性能权衡曲线（Acc vs Fairness）
- 显著性：差异显著性+效应量报告
- 复现：`bash eval_fair.sh --dataset toy --metrics dp,eo,prp`

## 5. 公平/鲁棒/安全 / FRS

- 公平：重点报告最弱势子群体
- 鲁棒：分布偏移下公平指标稳定性（建议）
- 安全：不涉及越狱/注入；关注隐私泄露风险

## 6. 风险与使用 / Risks & Usage

- 风险：错误缓解可能降低关键人群可及性
- 缓解：人类在环复核；多利益相关方评审
- 合规：配合地区法规与机构政策

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
