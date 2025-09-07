# 示例评测卡（安全机制） / Example Evaluation Card (Safety Mechanisms)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-Safety-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：提示注入/越狱拦截、违禁内容检测
- 指标：TPR↑、FPR↓、越狱率↓、拒答准确率↑、响应延迟
- 显著性：95%CI（自助法；示例）

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11, CUDA 12.1；随机种子=123
- 容器：ghcr.io/formalai/safety-eval:0.1（示例）
- 一键命令：`bash eval.sh --policy ./policy.yaml --suite ./suites/jailbreak.yaml`（示例）

## 3. 数据与合规 / Data & Compliance

- 套件：jailbreak.yaml, prompt-injection.yaml（示例）
- 许可：遵循原始套件许可证；不含隐私数据（示例）
- 版本/哈希：suite@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：子群体（语言/地区）误拒差异报告（示例）
- 鲁棒：同义改写、上下文污染、混淆攻击评测（示例）
- 安全：红队流程、日志与复核清单（示例）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动（示例）
- 提交流程：PR+自动校验+复核Taskforce（示例）
- 撤榜：不实复现或不当数据将撤榜（示例）

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
