# 跨模态推理 评测卡（示例） / Cross-Modal Reasoning Eval Card (Example)

## 任务与设置 / Task & Setup

- 子任务：跨模态检索、问答、逻辑一致性验证
- 设备：A100 40GB ×1
- 随机种子：2025

## 数据 / Data

- 基准：选择2–3个公开多模态推理基准（标注版本与许可）
- 划分：官方划分；如自定义需附脚本

## 指标 / Metrics

- Top-1/Top-5、EM/F1
- 一致性通过率（逻辑/语义）
- 显著性与统计检验（p值/置信区间）

## 协议 / Protocol

- 固定prompt与超参，报告完整配置YAML
- 每次评测重复≥3次，报告均值±方差

## 复现 / Reproducibility

- 提供评测脚本、环境与版本锁定文件
- 公开日志与随机种子
