# 示例评测卡（鲁棒性理论） / Example Evaluation Card (Robustness Theory)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Robustness-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：对抗鲁棒性、分布偏移鲁棒性、不确定性校准
- 指标：ASR↓、性能保持率↑、ECE↓；95%CI+效应量

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11；seed=123
- 容器：ghcr.io/formalai/robust-eval:0.1（示例）
- 一键命令：`bash eval_robust.sh --attacks pgd,fgsm --defenses adv_training,augmentation`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-adversarial、toy-distribution-shift、toy-uncertainty（示例）
- 许可：遵守数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按子群体报告鲁棒性差异与显著性
- 鲁棒：重点评估对抗攻击与分布偏移下的性能稳定性
- 安全：关注攻击面与防御失效场景的评估

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+安全审查
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
