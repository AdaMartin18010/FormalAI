# 示例评测卡（AI哲学） / Example Evaluation Card (AI Philosophy)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 评测名称：FormalAI-AI-Philosophy-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：图灵测试、意识评估、哲学推理
- 指标：通过率、意识评分、推理正确率；95%CI

## 2. 协议与环境 / Protocol & Environment

- 环境：Python 3.11、NLTK 3.8、spaCy 3.7；seed=42
- 容器：ghcr.io/formalai/ai-philosophy-eval:0.1（示例）
- 一键命令：`bash eval_ai_philosophy.sh --test turing,consciousness,reasoning`

## 3. 数据与合规 / Data & Compliance

- 套件：toy-turing-test、toy-consciousness-assessment、toy-philosophical-reasoning（示例）
- 许可：遵守原始数据许可；不含隐私
- 版本/哈希：toy@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：按测试类型/文化背景子集做差异分析（建议）
- 鲁棒：测试环境变化/问题类型/语言多样性下的性能稳定性（建议）
- 安全：AI哲学思考的伦理边界与安全性评估（建议）

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：季度滚动
- 提交流程：PR+自动校验+哲学审查
- 撤榜：不实复现或违反评测协议撤榜

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
