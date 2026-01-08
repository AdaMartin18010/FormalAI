# 文件归档执行计划

**创建日期**：2025-01-XX
**状态**：待执行

---

## 一、归档目标

将项目中与核心主题无关的进度报告、状态更新、完成总结等文件归档到`archive/`目录，清理项目结构，提高核心内容的可见性。

## 二、归档文件清单

### 2.1 根目录文件（9个）

需要归档到`archive/progress-reports/2025/`：

1. `PROGRESS_SUMMARY.md`
2. `PROJECT_STATUS_SUMMARY.md`
3. `PROJECT_STATUS_UPDATE_2025.md`
4. `PROJECT_TASK_OVERVIEW.md`
5. `FINAL_PROJECT_STATUS.md`
6. `TASK1_PROGRESS_UPDATE.md`
7. `TASK1_CONTINUED_PROGRESS.md`
8. `TASKS_1_2_3_COMPLETION_SUMMARY.md`
9. `CROSS_REF_FIX_SUMMARY.md`

### 2.2 Concepts模块文件（12个）

需要归档到`archive/concepts-reports/`：

**Phase1相关（7个）** → `archive/concepts-reports/phase1/`：
1. `concepts/PHASE1_COMPLETION_SUMMARY.md`
2. `concepts/PHASE1_COMPREHENSIVE_EXPLANATION.md`
3. `concepts/PHASE1_DETAILED_EXPLANATION.md`
4. `concepts/PHASE1_FINAL_SUMMARY.md`
5. `concepts/PHASE1_IMPROVEMENT_PROGRESS.md`
6. `concepts/PHASE1_INTEGRATED_ANALYSIS.md`
7. `concepts/PHASE1_MASTER_SUMMARY.md`

**交叉引用相关（5个）** → `archive/concepts-reports/cross-ref/`：
1. `concepts/concepts_cross_ref_report.md`
2. `concepts/concepts_cross_ref_report_v2.md`
3. `concepts/cross_reference_report.md`
4. `concepts/REPAIR_LOGIC_EXPLANATION.md`
5. `concepts/VIEW_CONCEPTS_ALIGNMENT_REPORT.md`
6. `concepts/CONCEPTS_CRITICAL_EVALUATION_REPORT.md`（可选，保留在concepts/作为参考）

### 2.3 Philosophy模块文件（23个）

需要归档到`archive/philosophy-reports/`：

**执行相关（6个）** → `archive/philosophy-reports/execution/`：
1. `Philosophy/EXECUTION_PROGRESS_TRACKER.md`
2. `Philosophy/EXECUTION_PHASE_SUMMARY.md`
3. `Philosophy/EXECUTION_STATUS_DASHBOARD.md`
4. `Philosophy/EXECUTION_WEEKLY_REPORT.md`
5. `Philosophy/EXECUTION_ADVANCEMENT_GUIDE.md`
6. `Philosophy/EXECUTION_TEMPLATES.md`

**改进相关（8个）** → `archive/philosophy-reports/improvement/`：
1. `Philosophy/IMPROVEMENT_PROGRESS_SUMMARY.md`
2. `Philosophy/IMPROVEMENT_PROGRESS_UPDATE.md`
3. `Philosophy/IMPROVEMENT_COMPLETE_SUMMARY.md`
4. `Philosophy/IMPROVEMENT_COMPLETION_REPORT.md`
5. `Philosophy/IMPROVEMENT_WORK_SUMMARY.md`
6. `Philosophy/FINAL_IMPROVEMENT_REPORT_2025.md`
7. `Philosophy/FINAL_IMPROVEMENT_WORK_SUMMARY_2025.md`
8. `Philosophy/WORK_PROGRESS_SUMMARY.md`

**完成相关（9个）** → `archive/philosophy-reports/completion/`：
1. `Philosophy/PROJECT_100_PERCENT_COMPLETION_REPORT.md`
2. `Philosophy/PROJECT_COMPLETION_CHECKLIST.md`
3. `Philosophy/PROJECT_FINAL_COMPLETION_SUMMARY.md`
4. `Philosophy/PROJECT_STATUS_OVERVIEW.md`
5. `Philosophy/PROJECT_STATUS_SNAPSHOT.md`
6. `Philosophy/COMPREHENSIVE_IMPROVEMENT_REPORT_2025.md`
7. `Philosophy/COMPLETION_REPORT.md`
8. `Philosophy/FINAL_WORK_SUMMARY.md`
9. `Philosophy/philosophy_cross_ref_report.md`

### 2.4 Docs模块文件（4个）

需要归档到`archive/docs-reports/completion/`：

1. `docs/PROJECT_COMPLETION_REPORT.md`
2. `docs/PROJECT_COMPLETION_SUMMARY_2025.md`
3. `docs/LATEST_UPDATES_INDEX.md`
4. `docs/LATEST_UPDATES_INDEX_2025.md`

## 三、归档执行步骤

### 步骤1：创建归档目录结构

```powershell
# 已在archive/README.md中说明目录结构
# 目录已创建完成
```

### 步骤2：移动文件

**注意**：以下操作需要手动执行或使用脚本批量执行。

#### 2.1 移动根目录文件

```powershell
# 移动到 archive/progress-reports/2025/
Move-Item -Path "PROGRESS_SUMMARY.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "PROJECT_STATUS_SUMMARY.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "PROJECT_STATUS_UPDATE_2025.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "PROJECT_TASK_OVERVIEW.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "FINAL_PROJECT_STATUS.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "TASK1_PROGRESS_UPDATE.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "TASK1_CONTINUED_PROGRESS.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "TASKS_1_2_3_COMPLETION_SUMMARY.md" -Destination "archive\progress-reports\2025\"
Move-Item -Path "CROSS_REF_FIX_SUMMARY.md" -Destination "archive\progress-reports\2025\"
```

#### 2.2 移动Concepts模块文件

```powershell
# Phase1相关文件
Move-Item -Path "concepts\PHASE1_COMPLETION_SUMMARY.md" -Destination "archive\concepts-reports\phase1\"
Move-Item -Path "concepts\PHASE1_COMPREHENSIVE_EXPLANATION.md" -Destination "archive\concepts-reports\phase1\"
Move-Item -Path "concepts\PHASE1_DETAILED_EXPLANATION.md" -Destination "archive\concepts-reports\phase1\"
Move-Item -Path "concepts\PHASE1_FINAL_SUMMARY.md" -Destination "archive\concepts-reports\phase1\"
Move-Item -Path "concepts\PHASE1_IMPROVEMENT_PROGRESS.md" -Destination "archive\concepts-reports\phase1\"
Move-Item -Path "concepts\PHASE1_INTEGRATED_ANALYSIS.md" -Destination "archive\concepts-reports\phase1\"
Move-Item -Path "concepts\PHASE1_MASTER_SUMMARY.md" -Destination "archive\concepts-reports\phase1\"

# 交叉引用相关文件
Move-Item -Path "concepts\concepts_cross_ref_report.md" -Destination "archive\concepts-reports\cross-ref\"
Move-Item -Path "concepts\concepts_cross_ref_report_v2.md" -Destination "archive\concepts-reports\cross-ref\"
Move-Item -Path "concepts\cross_reference_report.md" -Destination "archive\concepts-reports\cross-ref\"
Move-Item -Path "concepts\REPAIR_LOGIC_EXPLANATION.md" -Destination "archive\concepts-reports\cross-ref\"
Move-Item -Path "concepts\VIEW_CONCEPTS_ALIGNMENT_REPORT.md" -Destination "archive\concepts-reports\cross-ref\"
```

#### 2.3 移动Philosophy模块文件

```powershell
# 执行相关文件
Move-Item -Path "Philosophy\EXECUTION_PROGRESS_TRACKER.md" -Destination "archive\philosophy-reports\execution\"
Move-Item -Path "Philosophy\EXECUTION_PHASE_SUMMARY.md" -Destination "archive\philosophy-reports\execution\"
Move-Item -Path "Philosophy\EXECUTION_STATUS_DASHBOARD.md" -Destination "archive\philosophy-reports\execution\"
Move-Item -Path "Philosophy\EXECUTION_WEEKLY_REPORT.md" -Destination "archive\philosophy-reports\execution\"
Move-Item -Path "Philosophy\EXECUTION_ADVANCEMENT_GUIDE.md" -Destination "archive\philosophy-reports\execution\"
Move-Item -Path "Philosophy\EXECUTION_TEMPLATES.md" -Destination "archive\philosophy-reports\execution\"

# 改进相关文件
Move-Item -Path "Philosophy\IMPROVEMENT_PROGRESS_SUMMARY.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\IMPROVEMENT_PROGRESS_UPDATE.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\IMPROVEMENT_COMPLETE_SUMMARY.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\IMPROVEMENT_COMPLETION_REPORT.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\IMPROVEMENT_WORK_SUMMARY.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\FINAL_IMPROVEMENT_REPORT_2025.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\FINAL_IMPROVEMENT_WORK_SUMMARY_2025.md" -Destination "archive\philosophy-reports\improvement\"
Move-Item -Path "Philosophy\WORK_PROGRESS_SUMMARY.md" -Destination "archive\philosophy-reports\improvement\"

# 完成相关文件
Move-Item -Path "Philosophy\PROJECT_100_PERCENT_COMPLETION_REPORT.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\PROJECT_COMPLETION_CHECKLIST.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\PROJECT_FINAL_COMPLETION_SUMMARY.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\PROJECT_STATUS_OVERVIEW.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\PROJECT_STATUS_SNAPSHOT.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\COMPREHENSIVE_IMPROVEMENT_REPORT_2025.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\COMPLETION_REPORT.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\FINAL_WORK_SUMMARY.md" -Destination "archive\philosophy-reports\completion\"
Move-Item -Path "Philosophy\philosophy_cross_ref_report.md" -Destination "archive\philosophy-reports\completion\"
```

#### 2.4 移动Docs模块文件

```powershell
# 完成相关文件
Move-Item -Path "docs\PROJECT_COMPLETION_REPORT.md" -Destination "archive\docs-reports\completion\"
Move-Item -Path "docs\PROJECT_COMPLETION_SUMMARY_2025.md" -Destination "archive\docs-reports\completion\"
Move-Item -Path "docs\LATEST_UPDATES_INDEX.md" -Destination "archive\docs-reports\completion\"
Move-Item -Path "docs\LATEST_UPDATES_INDEX_2025.md" -Destination "archive\docs-reports\completion\"
```

### 步骤3：更新引用链接

归档后，需要检查并更新所有引用这些文件的链接：

1. 检查README.md中的引用
2. 检查各模块README.md中的引用
3. 检查交叉引用文档
4. 创建重定向文档（可选）

### 步骤4：更新归档索引

更新`archive/ARCHIVE_INDEX.md`，记录所有已归档的文件。

## 四、注意事项

1. **备份**：归档前建议先备份所有文件
2. **链接检查**：归档后需要检查并更新所有引用链接
3. **保留核心报告**：某些核心报告（如`CONCEPTS_CRITICAL_EVALUATION_REPORT.md`）可以保留在原位置作为参考
4. **Git提交**：归档操作完成后，建议创建独立的Git提交

## 五、验收标准

- [ ] 所有进度报告文件已移动到归档目录
- [ ] 归档索引文档已更新
- [ ] 所有引用链接已更新或创建重定向
- [ ] README.md已更新，添加归档说明
- [ ] Git提交已完成

---

**创建日期**：2025-01-XX
**维护者**：FormalAI项目组
