# 变更日志 / Changelog

本文档记录FormalAI项目的所有重大变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [未发布]

### 新增

- 创建全局导航文档（`docs/GLOBAL_NAVIGATION.md`），整合Philosophy、concepts、view模块
- 在view模块中补充Philosophy内容引用：
  - `view/ai_models_view.md` - 添加形式化数学基础引用
  - `view/ai_engineer_view.md` - 添加Ontology作为基础设施引用
  - `view/ai_scale_view.md` - 添加ARI/HR评估指标引用
- 创建项目任务总览文档（`PROJECT_TASK_OVERVIEW.md`）
- 创建变更日志（`CHANGELOG.md`）
- 创建批量处理脚本（`scripts/add_last_updated.py`）用于添加"最后更新"标记

### 改进

- 更新`docs/GLOBAL_NAVIGATION.md`，添加跨模块整合导航部分
- 完善view模块与Philosophy模块的关联
- 完成Docs模块的TODO标记：
  - `docs/00-foundations/00-mathematical-foundations/05-formal-proofs.md` - 添加Proof对象和check函数草案
  - `docs/00-foundations/00-mathematical-foundations/03-logical-calculus.md` - 添加公式语法和推理规则定义
  - `docs/03-formal-methods/03.5-DKB案例研究.md` - 添加T1定理的证明思路说明
- 完成Concepts模块"最后更新"标记添加（已为108个文件添加标记，60个文件已有标记）
- 确认Concepts模块"与三层模型的关系"章节完整性（所有主题文档都已包含该章节）
- 完成Concepts模块07.7-07.11系列文档检查（所有文档结构完整，内容已相当完善）
- 创建自动化检查脚本：
  - `scripts/check_cross_references.py` - 交叉引用检查
  - `scripts/check_document_integrity.py` - 文档完整性检查
  - `scripts/check_terminology_consistency.py` - 术语一致性检查
- 创建Philosophy模块数据更新指南（`Philosophy/DATA_UPDATE_GUIDE.md`）
  - 建立数据时效性检查机制
  - 提供数据更新模板和检查清单
  - 明确数据来源和更新周期
- 分析Concepts模块交叉引用错误
  - 创建错误分析脚本（`scripts/analyze_cross_ref_errors.py`）
  - 发现实际需要修复112个错误（涉及68个文件）
  - 创建锚点修复脚本（`scripts/fix_cross_ref_anchors.py`、`scripts/fix_anchor_double_dash.py`）
  - 创建修复总结文档（`CROSS_REF_FIX_SUMMARY.md`）
  - 修复锚点双连字符问题（17个文件已修复）

---

## [1.0.0] - 2025-01-XX

### 新增

#### Philosophy模块

- 创建Philosophy模块完整文档体系
- 创建6个view文档（view01-view06）
- 创建12个model文档（model/01-12）
- 创建风险与反证总览（`model/12-风险与反证总览.md`）
- 创建跨模块映射索引（`model/09-跨模块映射索引.md`）

#### 跨模块整合

- 在`docs/03-formal-methods/`中创建DKB案例研究（`03.5-DKB案例研究.md`）
- 在`docs/04-language-models/`中添加ARI/HR评估指标（`04.6-AI评估指标.md`）
- 在`docs/09-philosophy-ethics/09.1-AI哲学/`中嵌入哲学转译内容（§6）
- 在`concepts/05-AI科学理论/`中添加Ontology章节（§9）
- 在`concepts/07-AI框架批判与重构/`中添加Ontology视角（§7.2、§9.2）

#### 形式化证明

- 为所有公理（A1-A6）添加"不可证伪性"元论证
- 补充证明树10的详细数学推导
- 为范畴论/类型论映射添加严格的同构证明

#### 风险与反证

- 扩展矩阵9，从8个失败模式扩展到16个失败模式
- 完善证明树8，补充16个失败模式的详细分析
- 创建风险与反证总览文档

### 改进

- 完善所有文档的交叉引用
- 统一术语定义（`model/07-术语表与概念索引.md`）
- 统一文档格式和结构规范

---

## 版本说明

### 版本号格式

版本号遵循语义化版本规范：`主版本号.次版本号.修订号`

- **主版本号**：不兼容的API修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

### 变更类型

- **新增**：新功能、新文档、新模块
- **改进**：功能增强、内容完善、结构优化
- **修复**：错误修正、链接修复、格式修正
- **删除**：移除的功能、废弃的文档
- **安全**：安全相关的修复

---

**最后更新**：2025-01-XX
**维护者**：FormalAI项目组
