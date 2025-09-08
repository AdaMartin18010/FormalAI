# 示例评测卡（涌现理论） / Example Evaluation Card (Emergence Theory)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Emergence-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：涌现现象检测、复杂性分析、自组织识别
- 指标：涌现检测准确率、复杂性度量、自组织度；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11、Mesa 2.1、NetworkX 3.1；seed=42
- 容器：ghcr.io/formalai/emergence-eval:0.1（示例）
- 一键命令：`bash eval_emergence.sh --system multi_agent,cellular_automata,network`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-emergence-detection、toy-complexity-analysis、toy-self-organization（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按系统类型/规模子集做差异分析（建议）
- 鲁棒：系统参数变化/噪声/初始条件下的性能稳定性（建议）
- 安全：复杂系统行为的安全性评估与涌现控制（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+涌现验证
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
