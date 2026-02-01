# 自动化工具使用指南 / Automation Tools Guide

**创建日期**：2025-02-02
**目的**：提供FormalAI项目自动化工具的使用指南和说明
**维护**：随工具更新而更新

---

## 一、执行摘要

### 1.1 工具分类

FormalAI项目提供以下自动化工具：

1. **引用格式检查工具**：检查文档中的引用格式是否符合规范
2. **死链接检测工具**：检测文档中的失效链接
3. **内容完整性检查工具**：检查文档结构的完整性和一致性
4. **交叉引用检查工具**：验证跨模块引用的有效性
5. **术语一致性检查工具**：确保术语使用的一致性
6. **文档质量检查工具**：检查文档质量指标

### 1.2 工具位置

所有自动化工具位于 `scripts/` 目录下。

---

## 二、引用格式检查工具

### 2.1 check_citation_format.py

**功能**：检查文档中的引用格式是否符合 [CITATION_STYLE_GUIDE.md](CITATION_STYLE_GUIDE.md) 规范。

**使用方法**：

```bash
python scripts/check_citation_format.py [文件路径或目录]
```

**检查项**：

- arXiv引用格式：`arXiv:编号`
- 会议论文格式：`会议名称 年份`
- 期刊论文格式：`期刊名称, 卷(期), 页码`
- 教材引用格式：`作者 (年份). 书名. 出版社. 版次, 页码`

**输出**：

- 列出不符合规范的引用
- 提供修正建议

**示例**：

```bash
python scripts/check_citation_format.py docs/
```

---

## 三、死链接检测工具

### 3.1 check_cross_references.py

**功能**：检测文档中的失效链接（包括内部链接和外部链接）。

**使用方法**：

```bash
python scripts/check_cross_references.py [文件路径或目录]
```

**检查项**：

- 内部Markdown链接：`[文本](./path.md)`
- 外部HTTP/HTTPS链接
- 锚点链接：`[文本](./path.md#anchor)`

**输出**：

- 列出所有失效链接
- 提供链接位置（文件路径和行号）

**示例**：

```bash
python scripts/check_cross_references.py docs/
```

---

## 四、内容完整性检查工具

### 4.1 check_document_integrity.py

**功能**：检查文档结构的完整性和一致性。

**使用方法**：

```bash
python scripts/check_document_integrity.py [文件路径或目录]
```

**检查项**：

- 必需章节是否存在（概述、目录、核心结论等）
- 章节编号是否连续
- 目录链接是否有效
- 文档元数据是否完整

**输出**：

- 列出缺失的必需章节
- 列出章节编号错误
- 列出无效的目录链接

**示例**：

```bash
python scripts/check_document_integrity.py concepts/
```

---

## 五、交叉引用检查工具

### 5.1 check_cross_references.py（扩展功能）

**功能**：验证跨模块引用的有效性。

**检查项**：

- docs ↔ concepts 引用
- Philosophy ↔ concepts 引用
- docs ↔ Philosophy 引用

**输出**：

- 列出无效的跨模块引用
- 提供引用映射建议

---

## 六、术语一致性检查工具

### 6.1 check_terminology_consistency.py

**功能**：确保术语使用的一致性。

**使用方法**：

```bash
python scripts/check_terminology_consistency.py [文件路径或目录]
```

**检查项**：

- 核心术语定义是否一致
- 术语使用是否符合 [CONCEPT_REFERENCE_STANDARD.md](../CONCEPT_REFERENCE_STANDARD.md)
- 跨模块术语是否对齐

**输出**：

- 列出不一致的术语使用
- 提供术语对齐建议

**示例**：

```bash
python scripts/check_terminology_consistency.py .
```

---

## 七、文档质量检查工具

### 7.1 check_document_quality.py

**功能**：检查文档质量指标。

**使用方法**：

```bash
python scripts/check_document_quality.py [文件路径或目录]
```

**检查项**：

- 文档长度是否合适
- 章节结构是否清晰
- 代码示例是否完整
- 图表是否有效

**输出**：

- 文档质量评分
- 改进建议

**示例**：

```bash
python scripts/check_document_quality.py docs/
```

---

## 八、文档更新检查工具

### 8.1 check_document_updates.py

**功能**：检查文档是否需要更新。

**使用方法**：

```bash
python scripts/check_document_updates.py [文件路径或目录]
```

**检查项**：

- 文档最后更新日期
- 引用数据是否过期（超过6个月）
- 是否需要补充2025最新发展

**输出**：

- 列出需要更新的文档
- 提供更新建议

**示例**：

```bash
python scripts/check_document_updates.py docs/
```

---

## 九、批量工具使用

### 9.1 运行所有检查工具

**脚本**：`scripts/run_all_checks.py`（需要创建）

**功能**：一次性运行所有检查工具。

**使用方法**：

```bash
python scripts/run_all_checks.py
```

**输出**：

- 汇总所有检查结果
- 生成检查报告

---

## 十、工具配置

### 10.1 配置文件

工具配置可以通过环境变量或配置文件设置：

- **引用格式规范**：`CITATION_STYLE_GUIDE.md`
- **术语标准**：`CONCEPT_REFERENCE_STANDARD.md`
- **文档结构标准**：`concepts/DOCUMENT_STRUCTURE_STANDARD.md`

### 10.2 自定义检查规则

可以在各工具脚本中自定义检查规则，或创建配置文件。

---

## 十一、更新记录

| 日期 | 更新内容 |
|------|----------|
| 2025-02-02 | 初版创建，包含所有自动化工具的说明 |

---

**维护者**：FormalAI项目组
**关联文档**：

- [引用格式指南](CITATION_STYLE_GUIDE.md)
- [概念引用规范](../CONCEPT_REFERENCE_STANDARD.md)
- [文档结构标准](../concepts/DOCUMENT_STRUCTURE_STANDARD.md)
- [季度更新检查清单](QUARTERLY_UPDATE_CHECKLIST.md)
