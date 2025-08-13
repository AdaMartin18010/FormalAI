# 自动化质量校验工具设计 / Automated QA Tooling Design / Entwurf automatisierter Qualitätssicherungstools / Conception d'outils d'assurance qualité automatisés

## 目标 / Goals
- 自动化检查四语言标题完整性、严格序号、交叉引用可达性
- 自动化检测术语一致性与数学公式渲染风险
- 生成项目级质量报告并集成到CI

## 范围 / Scope
- 目标目录：`docs/` 全部 Markdown 文件
- 检查项：标题规范、术语一致、LaTeX、链接有效性、目录编号

## 模块 / Modules
1. HeadingChecker
   - 规则：一级标题“四语言 + 严格序号”
   - 实现：正则匹配 `^#\s+\d+\.\d+\s+.+\s/\s.+\s/\s.+\s/\s.+$`
2. NumberingChecker
   - 规则：章节与子节编号连续且层级一致
   - 实现：扫描 `#`,`##`,`###` 等并构造层级树校验
3. CrossRefChecker
   - 规则：相关章节链接可达，路径合法
   - 实现：解析 Markdown 链接并验证文件存在
4. TerminologyChecker
   - 规则：术语来自词典；检测未登记术语
   - 实现：基于 `MULTILINGUAL_TERMINOLOGY_GLOSSARY.md` 的匹配
5. MathChecker
   - 规则：`$...$` 与 `$$...$$` 成对；常见LaTeX拼写
   - 实现：标记配对与黑名单关键字扫描
6. ReportGenerator
   - 输出：HTML/Markdown 报告，总分与各项得分

## 接口 / Interfaces
- CLI: `formalai-qa scan docs/ --report out/qa.md`
- Exit Codes: 0=通过，1=警告，2=失败

## CI 集成 / CI Integration
- GitHub Actions / GitLab CI: 在 PR 中运行并上传报告
- 阻断标准：Heading/Numbering/CrossRef 严重错误阻断合并

## 路线 / Roadmap
- v0.1: 本地原型（Node.js/TS 或 Python）
- v0.2: 术语与LaTeX检查模块
- v0.3: CI集成与评分体系
- v1.0: 规则配置化与自定义插件

## 样例规则 / Sample Rules
- 一级标题必须匹配：`^#\s+\d+\.\d+\s+.*\s/\s.*\s/\s.*\s/\s.*$`
- 二级标题建议包含中文与英文：`^##\s+.+\s/\s.+$`

---

本设计文档将指导后续“自动化工具开发”目标的实现落地。 