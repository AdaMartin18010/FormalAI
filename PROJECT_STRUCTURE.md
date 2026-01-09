# FormalAI项目结构说明 / Project Structure

**创建日期**：2025-01-XX  
**最后更新**：2025-01-XX  
**状态**：✅ 已重构

---

## 📋 概述

本文档说明FormalAI项目的完整结构，包括目录组织、文件分类和主题层次关系。

---

## 一、项目目录结构

```
FormalAI/
├── README.md                    # 项目主文档
├── LICENSE                      # 许可证
├── PROJECT_STRUCTURE.md         # 项目结构说明（本文档）
├── PROJECT_REORGANIZATION_PLAN.md  # 项目重构计划
│
├── docs/                        # 核心文档目录（20个主题模块）
│   ├── README.md               # Docs模块主文档
│   ├── GLOBAL_NAVIGATION.md    # 全局导航（完整版）
│   ├── GLOBAL_NAVIGATION_CLEAN.md  # 全局导航（精简版）⭐推荐
│   ├── THEME_HIERARCHY_STRUCTURE.md  # 主题层次结构 ⭐新创建
│   ├── THEME_SEMANTIC_STRUCTURE.md   # 主题语义结构 ⭐新创建
│   ├── LEARNING_PATH_DESIGN.md      # 学习路径设计
│   ├── LATEST_AI_DEVELOPMENTS_2025.md  # 2025年最新AI发展总结 ⭐核心参考
│   │
│   ├── 0-总览与导航/            # 总览与导航
│   ├── 00-foundations/          # 数学与逻辑基础
│   ├── 01-foundations/          # 基础理论
│   ├── 02-machine-learning/    # 机器学习理论
│   ├── 03-formal-methods/       # 形式化方法
│   ├── 04-language-models/      # 语言模型
│   ├── 05-multimodal-ai/        # 多模态AI
│   ├── 06-interpretable-ai/     # 可解释AI
│   ├── 07-alignment-safety/     # 对齐与安全
│   ├── 08-emergence-complexity/ # 涌现与复杂性
│   ├── 09-philosophy-ethics/    # 哲学与伦理
│   ├── 10-embodied-ai/          # 具身AI
│   ├── 11-edge-ai/              # 边缘AI
│   ├── 12-quantum-ai/           # 量子AI
│   ├── 13-neural-symbolic/      # 神经符号AI
│   ├── 14-green-ai/             # 绿色AI
│   ├── 15-meta-learning/       # 元学习
│   ├── 16-agi-theory/           # AGI理论
│   ├── 17-social-ai/            # 社会AI
│   ├── 18-cognitive-architecture/  # 认知架构
│   ├── 19-neuro-symbolic-advanced/  # 高级神经符号AI
│   └── 20-ai-philosophy-advanced/   # 高级AI哲学
│
├── concepts/                     # 核心概念目录（8个主题）
│   ├── README.md
│   ├── 01-AI三层模型架构/
│   ├── 02-AI炼金术转化度模型/
│   ├── 03-Scaling Law与收敛分析/
│   ├── 04-AI意识与认知模拟/
│   ├── 05-AI科学理论/
│   ├── 06-AI反实践判定系统/
│   ├── 07-AI框架批判与重构/
│   └── 08-AI历史进程与原理演进/
│
├── Philosophy/                   # 哲学模块目录
│   ├── README.md
│   ├── 00-主题总览与导航.md
│   ├── view01-06.md            # 视角文档
│   └── model/                  # 模型文档
│
├── archive/                      # 归档目录
│   ├── ARCHIVE_INDEX_2025.md   # 归档索引 ⭐新创建
│   ├── improvement-reports/     # 改进工作报告
│   ├── benchmarking-reports/    # 对标分析报告
│   ├── batch-update-reports/    # 批量更新报告
│   ├── status-reports/          # 状态报告
│   ├── other-reports/           # 其他报告
│   ├── concepts-reports/        # Concepts模块报告
│   ├── docs-reports/            # Docs模块报告
│   └── philosophy-reports/      # Philosophy模块报告
│
├── scripts/                      # 脚本目录
│   ├── update_2025_developments.py
│   └── validate_document_links.py
│
└── view/                         # 视图目录
    └── *.md                      # 各种视图文档
```

---

## 二、核心文档说明

### 2.1 项目根目录

- **README.md** - 项目主文档，提供项目概述和快速导航
- **PROJECT_STRUCTURE.md** - 项目结构说明（本文档）
- **PROJECT_REORGANIZATION_PLAN.md** - 项目重构计划

### 2.2 docs目录核心文档

**导航文档**：
- `GLOBAL_NAVIGATION.md` - 完整全局导航
- `GLOBAL_NAVIGATION_CLEAN.md` - 精简全局导航 ⭐推荐
- `THEME_HIERARCHY_STRUCTURE.md` - 主题层次结构 ⭐新创建
- `THEME_SEMANTIC_STRUCTURE.md` - 主题语义结构 ⭐新创建
- `LEARNING_PATH_DESIGN.md` - 学习路径设计

**参考文档**：
- `LATEST_AI_DEVELOPMENTS_2025.md` - 2025年最新AI发展总结 ⭐核心参考
- `DOCUMENT_INDEX_2025.md` - 文档索引
- `QUICK_REFERENCE_2025.md` - 快速参考

**标准文档**：
- `TEMPLATES_MODEL_CARD.md` - 模型卡模板
- `TEMPLATES_EVAL_CARD.md` - 评测卡模板
- `STANDARDS_CHECKLISTS.md` - 标准检查清单
- `CITATION_STYLE_GUIDE.md` - 引用格式指南

### 2.3 主题模块文档

每个主题目录包含：
- `README.md` - 主题主文档（必须）
- `EXAMPLE_MODEL_CARD.md` - 示例模型卡（如适用）
- `EXAMPLE_EVAL_CARD.md` - 示例评测卡（如适用）
- 子主题文档（如适用）

---

## 三、主题层次关系

### 3.1 层次结构

```
0. 总览与导航
  │
  ├─ 00. 数学与逻辑基础
  │     │
  │     └─ 01. 基础理论
  │           │
  │           ├─ 02. 机器学习理论
  │           │     │
  │           │     └─ 04. 语言模型
  │           │           │
  │           │           ├─ 05. 多模态AI
  │           │           ├─ 06. 可解释AI
  │           │           └─ 07. 对齐与安全
  │           │
  │           ├─ 03. 形式化方法
  │           │     │
  │           │     └─ 04. 语言模型
  │           │
  │           └─ 09. 哲学与伦理
  │                 │
  │                 └─ 20. 高级AI哲学
  │
  └─ 10-15. 前沿应用主题
        │
        └─ 16-20. 高级理论主题
```

### 3.2 语义层次

1. **基础语义层**：00-01（数学、逻辑、计算、认知基础）
2. **方法语义层**：02-03（机器学习方法、形式化方法）
3. **应用语义层**：04-09（语言模型、多模态、可解释性、对齐安全、涌现复杂性、哲学伦理）
4. **前沿语义层**：10-15（具身AI、边缘AI、量子AI、神经符号AI、绿色AI、元学习）
5. **高级语义层**：16-20（AGI理论、社会AI、认知架构、高级神经符号AI、高级AI哲学）

---

## 四、文件组织原则

### 4.1 核心内容优先

- ✅ 核心主题文档保留在docs/目录
- ✅ 核心概念文档保留在concepts/目录
- ✅ 哲学模块文档保留在Philosophy/目录

### 4.2 报告文档归档

- ✅ 所有进度报告归档到archive/
- ✅ 所有总结报告归档到archive/
- ✅ 所有状态报告归档到archive/

### 4.3 导航文档精简

- ✅ 移除对归档文档的引用
- ✅ 专注于核心主题导航
- ✅ 保持导航系统简洁

---

## 五、维护指南

### 5.1 添加新主题

1. 确定主题的语义层次
2. 分配唯一的序号
3. 创建目录和README.md
4. 添加交叉引用
5. 更新导航文档

### 5.2 更新现有主题

1. 保持序号不变
2. 更新内容
3. 更新交叉引用
4. 更新最后更新日期

### 5.3 归档文件

1. 识别与主题无关的文件
2. 移动到archive/相应目录
3. 更新归档索引

---

**最后更新**：2025-01-XX  
**维护者**：FormalAI项目组
