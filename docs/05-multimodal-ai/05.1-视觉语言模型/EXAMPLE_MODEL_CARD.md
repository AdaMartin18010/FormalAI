# 示例模型卡（VLM） / Example Model Card (VLM)

[模板与核对单](../../TEMPLATES_MODEL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 模型名称：FormalAI-VLM-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）
- 适用/不适用：教学示例；不用于生产/高风险场景

## 1. 概述 / Overview

- 目标：演示视觉-语言模型的模型卡编制要点
- 能力边界：小规模图像-文本匹配/简单VQA演示
- 已知限制：不具备稳健性与安全性保证

## 2. 训练数据 / Training Data

- 来源：公开演示样例（示例）
- 许可证：遵守原数据许可；不再分发原始内容
- 代表性：极小规模，不具统计代表性
- 质量控制：基本去重；无污染检测

## 3. 方法与配置 / Methods & Configs

- 架构：图像编码器（ViT-small）+ 文本编码器（Tiny-Transformer）+ 对比头（示例）
- 目标：InfoNCE/双向对比；τ=0.07（示例）
- 训练：batch=32, lr=5e-4, epoch=1（示例）
- 代码：repo=<https://example.com/formalai-vlm-example.git>, commit=abc1234

## 4. 评测与性能 / Evaluation & Performance

- 任务：图文检索、简易VQA（toy）
- 指标：R@1/R@5；VQA简易准确率（示例实现）
- 显著性：95%CI（自助法；示例）
- 复现：docker ghcr.io/formalai/vlm:0.1；`bash eval.sh`

## 5. 公平/鲁棒/安全 / FRS

- 公平：未做子群体评估
- 鲁棒：未做遮挡/噪声/模态丢失鲁棒性
- 安全：未做红队；需接入安全策略后再评估

## 6. 风险与使用 / Risks & Usage

- 风险：对图像细节/长文本敏感；易受分布外干扰
- 缓解：仅限教学；需人工校对；禁用于决策
- 合规：遵守数据/模型许可证与场景政策

## 7. 版本与治理 / Versioning & Governance

- 历史：v0.1.0 初版
- 变更：N/A
- 反馈：<issues@example.com>

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
