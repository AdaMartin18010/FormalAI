# 示例评测卡（多模态融合） / Example Evaluation Card (Multimodal Fusion)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Multimodal-Fusion-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：多模态分类、跨模态检索、模态缺失推理
- 指标：Acc、R@1/R@5、模态缺失性能保持率；报告95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：CUDA 12.1、PyTorch 2.2；seed=42
- 容器：ghcr.io/formalai/multimodal-fusion-eval:0.1（示例）
- 一键命令：`bash eval.sh --task classification,retrieval,missing --data toy`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-multimodal-classification、toy-cross-modal-retrieval、toy-missing-modality（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按模态/语言/主题子集做差异分析（建议）
- 鲁棒：模态缺失/噪声/分布偏移下的性能稳定性（建议）
- 安全：接入安全策略后执行越狱/注入套件（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验
- 撤榜：不实复现/违规范例撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
