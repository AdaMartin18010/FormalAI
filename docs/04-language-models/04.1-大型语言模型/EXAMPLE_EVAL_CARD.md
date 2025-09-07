# 示例评测卡（LLM） / Example Evaluation Card (LLM)

[模板与核对单](../../TEMPLATES_EVAL_CARD.md) · [最小合规核对单](../../STANDARDS_CHECKLISTS.md)

---

## 0. 摘要 / Summary

- 基准/评测名称：FormalAI-LLM-Eval-Example
- 版本：v0.1.0
- 发布日期：2025-01-02
- 维护者/联系：FormalAI 项目组 <contact@example.com>
- 许可证：Apache-2.0（示例）

## 1. 任务与指标 / Task & Metrics

- 任务：短文本补全（few-shot）
- 划分：toy-train / toy-valid / toy-test（固定种子）
- 指标：BLEU-1/ROUGE-L（示例实现）
- 显著性：自助法95%CI（示例）

## 2. 协议与环境 / Protocol & Environment

- 环境：CUDA 12.1，PyTorch 2.2，随机种子=42
- 容器：ghcr.io/formalai/example:0.1（示例）
- 一键命令：`bash eval.sh --model formalai-llm-example --data toy`（示例）

## 3. 数据与合规 / Data & Compliance

- 来源：公开演示文本，遵守原许可；不分发原始数据
- 隐私：无个人敏感信息（示例）
- 版本/哈希：toy-test@v0.1（sha256:deadbeef…示例）

## 4. 公平/鲁棒/安全 / FRS

- 公平：不适用（toy）；正式评测需提供子群体切分
- 鲁棒：简单扰动一致性测试（示例）
- 安全：未开展红队；正式评测需纳入提示注入与越狱用例

## 5. 榜单与治理 / Leaderboard & Governance

- 更新：手动更新（示例）；正式应月度滚动
- 提交流程：PR+自动校验（示例）
- 争议与撤榜：维护者审核（示例）

---

最后更新：2025-01-02  · 维护：FormalAI 项目组（示例）
