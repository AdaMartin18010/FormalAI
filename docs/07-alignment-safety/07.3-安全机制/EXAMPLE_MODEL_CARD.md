# 示例模型卡（安全机制） / Example Model Card (Safety Mechanisms)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型/系统名称：FormalAI-Safety-Guard-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产与高风险场景

## 1. 概述 / Overview

- 目标：演示内容安全与提示注入防护的卡片编写范式
- 能力边界：轻量策略过滤；不保证全面拦截
- 已知限制：覆盖面有限，对抗样本可能绕过

## 2. 数据 / Data

- 规则与样例来源：公开安全策略与演示用例（示例）
- 隐私：不含个人敏感信息（示例）

## 3. 方法与配置 / Methods & Configs

- 机制：策略黑白名单 + 轻量分类器（示例）
- 配置：阈值τ=0.7；拒答模板；审计日志开启（示例）
- 代码：repo=<https://example.com/formalai-safety-guard.git>, commit=abc1234（示例）

## 4. 评测与性能 / Evaluation & Performance

- 任务：有害内容检测、提示注入拦截
- 指标：TPR/FPR、拒答准确率、越狱率↓
- 显著性：自助法CI（示例）
- 复现：docker image formalai/safety:0.1；eval.sh（示例）

## 5. 公平/鲁棒/安全 / FRS

- 公平：敏感子群体误拒差异监测（示例）
- 鲁棒：同义改写/混淆攻击下性能（示例）
- 安全：红队流程与审计留痕（示例）

## 6. 风险与使用 / Risks & Usage

- 风险：过拟合策略、误杀正常请求
- 缓解：人工复核；灰度上线；持续红队
- 合规：遵循平台政策与适用法规（示例）

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A（示例）
- 反馈：<issues@example.com>（示例）

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
