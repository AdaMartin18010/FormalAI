# 多语言翻译标准 / Multilingual Translation Standards / Mehrsprachige Übersetzungsstandards / Normes de traduction multilingue

## 标准目标 / Objectives
- 统一术语与标题格式，确保四语言一致
- 保持语义等价与学术严谨
- 便于维护与自动校验

## 标准内容 / Standards

### 1. 标题规范 / Heading Conventions
- 一级标题采用四语言格式：中文 / 英文 / 德文 / 法文
- 保留严格序号前缀：如 `# 2.3 强化学习理论 / Reinforcement Learning Theory / ...`
- 二级及以下标题按需求可采用四语言或中文+英文，保持一致性

### 2. 术语规范 / Terminology
- 所有核心术语应引用 `docs/MULTILINGUAL_TERMINOLOGY_GLOSSARY.md`
- 优先使用项目既有译法，其次参考权威来源（ISO/ACL/ACM）
- 出现新术语时应在词典中登记

### 3. 语言风格 / Style
- 中文：学术、简洁、避免口语
- 英文：学术英式或美式可任选，但需全局一致
- 德文：遵循复合名词与大写规范
- 法文：注意连字符与性数配合

### 4. 数学与公式 / Math & LaTeX
- 统一使用 LaTeX，块级公式使用 `$$`，行内使用 `$...$`
- 符号和函数名遵循学术惯例（如 `\mathbb{R}`, `\mathcal{L}`）
- 公式下方尽量提供多语言说明

### 5. 链接与交叉引用 / Links & Cross-references
- 链接文本采用中文主语 + 英文补充
- 相对路径指向标准 `README.md`
- 相关章节区使用“前置依赖 / 后续应用”的双向链路

### 6. 版本与变更 / Versioning
- 每次批量翻译或修订需记录到 `docs/PROJECT_PROGRESS_UPDATE_REPORT.md`
- 对术语变更需在词典中标注“旧/新”对照

### 7. 校验与验收 / QA
- 标题四语言完整性检查（1级标题必检）
- 术语一致性与拼写检查
- 数学公式渲染检查

---

该标准将随项目演进持续迭代，以保证 FormalAI 的国际化质量与一致性。 