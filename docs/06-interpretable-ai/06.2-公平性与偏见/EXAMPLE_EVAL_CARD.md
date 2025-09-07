# 示例评测卡（公平性与偏见） / Example Evaluation Card (Fairness & Bias)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Fairness-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：群体公平（DP/EO/PRP）、个体公平、一致性
- 指标：差异度↓、一致性分↑、性能权衡曲线；95%CI+效应量

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11；seed=123
- 容器：ghcr.io/formalai/fair-eval:0.1（示例）
- 一键命令：`bash eval_fair.sh --dataset toy --metrics dp,eo,prp`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-fairness（示例）
- 许可：遵守数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按敏感属性/子群体出具差异表与显著性
- 鲁棒：分布偏移/采样变动的敏感性分析
- 安全：强调隐私合规与最小化可识别信息

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+抽样复核
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
