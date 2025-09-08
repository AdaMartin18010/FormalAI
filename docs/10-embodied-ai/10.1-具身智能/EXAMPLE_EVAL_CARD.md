# 示例评测卡（具身智能） / Example Evaluation Card (Embodied Intelligence)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Embodied-Intelligence-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：具身导航、物体操作、环境理解
- 指标：任务完成率、感知准确率、动作精度；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：PyBullet 3.2.5、MuJoCo 2.3、ROS 2；seed=42
- 容器：ghcr.io/formalai/embodied-intelligence-eval:0.1（示例）
- 一键命令：`bash eval_embodied.sh --task navigation,manipulation,understanding`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-embodied-navigation、toy-object-manipulation、toy-environment-understanding（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按环境类型/任务复杂度子集做差异分析（建议）
- 鲁棒：环境变化/传感器故障/物理约束下的性能稳定性（建议）
- 安全：具身系统的物理安全与操作安全评估（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+安全审查
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
