# 示例评测卡（可持续AI） / Example Evaluation Card (Sustainable AI)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Sustainable-AI-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：绿色计算、能耗优化、环境影响评估
- 指标：能耗效率、碳足迹、计算性能保持率；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11、PowerAPI 2.0、CarbonTracker 1.0；seed=42
- 容器：ghcr.io/formalai/sustainable-ai-eval:0.1（示例）
- 一键命令：`bash eval_sustainable.sh --metrics energy,carbon,performance`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-green-computing、toy-energy-optimization、toy-environmental-impact（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按计算资源/地理区域子集做差异分析（建议）
- 鲁棒：硬件变化/负载波动/环境条件下的性能稳定性（建议）
- 安全：绿色计算的安全性验证与环境影响评估（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+环保审查
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
