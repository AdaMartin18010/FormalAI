# 示例评测卡（量子机器学习） / Example Evaluation Card (Quantum Machine Learning)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Quantum-ML-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：量子分类、量子优化、量子生成建模
- 指标：分类准确率、优化目标值、保真度；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：Qiskit 0.45、Cirq 1.2；seed=42
- 容器：ghcr.io/formalai/quantum-ml-eval:0.1（示例）
- 一键命令：`bash eval_quantum.sh --backend simulator --shots 1000`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-quantum-classification、toy-quantum-optimization、toy-quantum-generation（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按量子态/测量基子集做差异分析（建议）
- 鲁棒：量子噪声/退相干下的性能稳定性（建议）
- 安全：量子密钥分发与量子安全通信协议（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+量子硬件验证
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
