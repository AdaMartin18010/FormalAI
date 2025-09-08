# 示例评测卡（神经符号AI） / Example Evaluation Card (Neural-Symbolic AI)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Neural-Symbolic-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：知识图谱补全、逻辑推理、可解释预测
- 指标：链接预测准确率、推理正确率、可解释性分数；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：PyTorch 2.2、DGL 1.1、SymPy 1.12；seed=42
- 容器：ghcr.io/formalai/neural-symbolic-eval:0.1（示例）
- 一键命令：`bash eval_neural_symbolic.sh --task kg_completion,reasoning,interpretability`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-knowledge-graph、toy-logical-reasoning、toy-interpretability（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按知识域/实体类型子集做差异分析（建议）
- 鲁棒：知识图谱不完整/噪声/分布偏移下的性能稳定性（建议）
- 安全：符号推理的安全性验证与逻辑一致性检查（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+符号验证
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
