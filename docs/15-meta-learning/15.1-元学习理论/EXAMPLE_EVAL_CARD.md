# 示例评测卡（元学习理论） / Example Evaluation Card (Meta-Learning Theory)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Meta-Learning-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：少样本分类、快速适应、任务泛化
- 指标：N-way K-shot准确率、适应速度、泛化性能；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：PyTorch 2.2、Learn2Learn 0.1.7；seed=42
- 容器：ghcr.io/formalai/meta-learning-eval:0.1（示例）
- 一键命令：`bash eval_meta_learning.sh --n_way 5 --k_shot 1,5`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-few-shot-classification、toy-rapid-adaptation、toy-task-generalization（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按任务类型/数据域子集做差异分析（建议）
- 鲁棒：任务分布偏移/域适应/样本稀缺下的性能稳定性（建议）
- 安全：元学习模型的安全性验证与快速适应鲁棒性（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+任务验证
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
