# 示例评测卡（联邦学习） / Example Evaluation Card (Federated Learning)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Federated-Learning-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：联邦分类、隐私保护评估、通信效率分析
- 指标：Acc、隐私预算消耗、通信轮数、收敛速度；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11、PyTorch 2.2；seed=123
- 容器：ghcr.io/formalai/federated-eval:0.1（示例）
- 一键命令：`bash eval_fed.sh --clients 5 --rounds 10 --privacy eps=1.0`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-federated-classification、toy-privacy-eval、toy-communication-eval（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按客户端数据分布报告性能差异与公平性指标
- 鲁棒：客户端掉线/恶意客户端/数据异构性下的性能稳定性
- 安全：隐私泄露风险评估；差分隐私保证验证

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+隐私审查
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
