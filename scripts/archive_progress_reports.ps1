# FormalAI项目进度报告文件归档脚本
# 创建日期：2025-01-XX
# 用途：批量移动进度报告文件到归档目录

Write-Host "开始归档进度报告文件..." -ForegroundColor Green

# 定义归档映射：源文件路径 -> 目标归档路径
$archiveMappings = @(
    # 根目录文件 -> archive/progress-reports/2025/
    @{Source = "PROGRESS_SUMMARY.md"; Target = "archive\progress-reports\2025\PROGRESS_SUMMARY.md"},
    @{Source = "PROJECT_STATUS_SUMMARY.md"; Target = "archive\progress-reports\2025\PROJECT_STATUS_SUMMARY.md"},
    @{Source = "PROJECT_STATUS_UPDATE_2025.md"; Target = "archive\progress-reports\2025\PROJECT_STATUS_UPDATE_2025.md"},
    @{Source = "PROJECT_TASK_OVERVIEW.md"; Target = "archive\progress-reports\2025\PROJECT_TASK_OVERVIEW.md"},
    @{Source = "FINAL_PROJECT_STATUS.md"; Target = "archive\progress-reports\2025\FINAL_PROJECT_STATUS.md"},
    @{Source = "TASK1_PROGRESS_UPDATE.md"; Target = "archive\progress-reports\2025\TASK1_PROGRESS_UPDATE.md"},
    @{Source = "TASK1_CONTINUED_PROGRESS.md"; Target = "archive\progress-reports\2025\TASK1_CONTINUED_PROGRESS.md"},
    @{Source = "TASKS_1_2_3_COMPLETION_SUMMARY.md"; Target = "archive\progress-reports\2025\TASKS_1_2_3_COMPLETION_SUMMARY.md"},
    @{Source = "CROSS_REF_FIX_SUMMARY.md"; Target = "archive\progress-reports\2025\CROSS_REF_FIX_SUMMARY.md"},
    
    # Concepts模块 Phase1相关 -> archive/concepts-reports/phase1/
    @{Source = "concepts\PHASE1_COMPLETION_SUMMARY.md"; Target = "archive\concepts-reports\phase1\PHASE1_COMPLETION_SUMMARY.md"},
    @{Source = "concepts\PHASE1_COMPREHENSIVE_EXPLANATION.md"; Target = "archive\concepts-reports\phase1\PHASE1_COMPREHENSIVE_EXPLANATION.md"},
    @{Source = "concepts\PHASE1_DETAILED_EXPLANATION.md"; Target = "archive\concepts-reports\phase1\PHASE1_DETAILED_EXPLANATION.md"},
    @{Source = "concepts\PHASE1_FINAL_SUMMARY.md"; Target = "archive\concepts-reports\phase1\PHASE1_FINAL_SUMMARY.md"},
    @{Source = "concepts\PHASE1_IMPROVEMENT_PROGRESS.md"; Target = "archive\concepts-reports\phase1\PHASE1_IMPROVEMENT_PROGRESS.md"},
    @{Source = "concepts\PHASE1_INTEGRATED_ANALYSIS.md"; Target = "archive\concepts-reports\phase1\PHASE1_INTEGRATED_ANALYSIS.md"},
    @{Source = "concepts\PHASE1_MASTER_SUMMARY.md"; Target = "archive\concepts-reports\phase1\PHASE1_MASTER_SUMMARY.md"},
    
    # Concepts模块 交叉引用相关 -> archive/concepts-reports/cross-ref/
    @{Source = "concepts\concepts_cross_ref_report.md"; Target = "archive\concepts-reports\cross-ref\concepts_cross_ref_report.md"},
    @{Source = "concepts\concepts_cross_ref_report_v2.md"; Target = "archive\concepts-reports\cross-ref\concepts_cross_ref_report_v2.md"},
    @{Source = "concepts\cross_reference_report.md"; Target = "archive\concepts-reports\cross-ref\cross_reference_report.md"},
    @{Source = "concepts\REPAIR_LOGIC_EXPLANATION.md"; Target = "archive\concepts-reports\cross-ref\REPAIR_LOGIC_EXPLANATION.md"},
    @{Source = "concepts\VIEW_CONCEPTS_ALIGNMENT_REPORT.md"; Target = "archive\concepts-reports\cross-ref\VIEW_CONCEPTS_ALIGNMENT_REPORT.md"},
    
    # Philosophy模块 执行相关 -> archive/philosophy-reports/execution/
    @{Source = "Philosophy\EXECUTION_PROGRESS_TRACKER.md"; Target = "archive\philosophy-reports\execution\EXECUTION_PROGRESS_TRACKER.md"},
    @{Source = "Philosophy\EXECUTION_PHASE_SUMMARY.md"; Target = "archive\philosophy-reports\execution\EXECUTION_PHASE_SUMMARY.md"},
    @{Source = "Philosophy\EXECUTION_STATUS_DASHBOARD.md"; Target = "archive\philosophy-reports\execution\EXECUTION_STATUS_DASHBOARD.md"},
    @{Source = "Philosophy\EXECUTION_WEEKLY_REPORT.md"; Target = "archive\philosophy-reports\execution\EXECUTION_WEEKLY_REPORT.md"},
    @{Source = "Philosophy\EXECUTION_ADVANCEMENT_GUIDE.md"; Target = "archive\philosophy-reports\execution\EXECUTION_ADVANCEMENT_GUIDE.md"},
    @{Source = "Philosophy\EXECUTION_TEMPLATES.md"; Target = "archive\philosophy-reports\execution\EXECUTION_TEMPLATES.md"},
    
    # Philosophy模块 改进相关 -> archive/philosophy-reports/improvement/
    @{Source = "Philosophy\IMPROVEMENT_PROGRESS_SUMMARY.md"; Target = "archive\philosophy-reports\improvement\IMPROVEMENT_PROGRESS_SUMMARY.md"},
    @{Source = "Philosophy\IMPROVEMENT_PROGRESS_UPDATE.md"; Target = "archive\philosophy-reports\improvement\IMPROVEMENT_PROGRESS_UPDATE.md"},
    @{Source = "Philosophy\IMPROVEMENT_COMPLETE_SUMMARY.md"; Target = "archive\philosophy-reports\improvement\IMPROVEMENT_COMPLETE_SUMMARY.md"},
    @{Source = "Philosophy\IMPROVEMENT_COMPLETION_REPORT.md"; Target = "archive\philosophy-reports\improvement\IMPROVEMENT_COMPLETION_REPORT.md"},
    @{Source = "Philosophy\IMPROVEMENT_WORK_SUMMARY.md"; Target = "archive\philosophy-reports\improvement\IMPROVEMENT_WORK_SUMMARY.md"},
    @{Source = "Philosophy\FINAL_IMPROVEMENT_REPORT_2025.md"; Target = "archive\philosophy-reports\improvement\FINAL_IMPROVEMENT_REPORT_2025.md"},
    @{Source = "Philosophy\FINAL_IMPROVEMENT_WORK_SUMMARY_2025.md"; Target = "archive\philosophy-reports\improvement\FINAL_IMPROVEMENT_WORK_SUMMARY_2025.md"},
    @{Source = "Philosophy\WORK_PROGRESS_SUMMARY.md"; Target = "archive\philosophy-reports\improvement\WORK_PROGRESS_SUMMARY.md"},
    
    # Philosophy模块 完成相关 -> archive/philosophy-reports/completion/
    @{Source = "Philosophy\PROJECT_100_PERCENT_COMPLETION_REPORT.md"; Target = "archive\philosophy-reports\completion\PROJECT_100_PERCENT_COMPLETION_REPORT.md"},
    @{Source = "Philosophy\PROJECT_COMPLETION_CHECKLIST.md"; Target = "archive\philosophy-reports\completion\PROJECT_COMPLETION_CHECKLIST.md"},
    @{Source = "Philosophy\PROJECT_FINAL_COMPLETION_SUMMARY.md"; Target = "archive\philosophy-reports\completion\PROJECT_FINAL_COMPLETION_SUMMARY.md"},
    @{Source = "Philosophy\PROJECT_STATUS_OVERVIEW.md"; Target = "archive\philosophy-reports\completion\PROJECT_STATUS_OVERVIEW.md"},
    @{Source = "Philosophy\PROJECT_STATUS_SNAPSHOT.md"; Target = "archive\philosophy-reports\completion\PROJECT_STATUS_SNAPSHOT.md"},
    @{Source = "Philosophy\COMPREHENSIVE_IMPROVEMENT_REPORT_2025.md"; Target = "archive\philosophy-reports\completion\COMPREHENSIVE_IMPROVEMENT_REPORT_2025.md"},
    @{Source = "Philosophy\COMPLETION_REPORT.md"; Target = "archive\philosophy-reports\completion\COMPLETION_REPORT.md"},
    @{Source = "Philosophy\FINAL_WORK_SUMMARY.md"; Target = "archive\philosophy-reports\completion\FINAL_WORK_SUMMARY.md"},
    @{Source = "Philosophy\philosophy_cross_ref_report.md"; Target = "archive\philosophy-reports\completion\philosophy_cross_ref_report.md"},
    
    # Docs模块 -> archive/docs-reports/completion/
    @{Source = "docs\PROJECT_COMPLETION_REPORT.md"; Target = "archive\docs-reports\completion\PROJECT_COMPLETION_REPORT.md"},
    @{Source = "docs\PROJECT_COMPLETION_SUMMARY_2025.md"; Target = "archive\docs-reports\completion\PROJECT_COMPLETION_SUMMARY_2025.md"},
    @{Source = "docs\LATEST_UPDATES_INDEX.md"; Target = "archive\docs-reports\completion\LATEST_UPDATES_INDEX.md"},
    @{Source = "docs\LATEST_UPDATES_INDEX_2025.md"; Target = "archive\docs-reports\completion\LATEST_UPDATES_INDEX_2025.md"}
)

$successCount = 0
$skipCount = 0
$errorCount = 0
$errors = @()

foreach ($mapping in $archiveMappings) {
    $source = $mapping.Source
    $target = $mapping.Target
    
    if (Test-Path $source) {
        try {
            # 确保目标目录存在
            $targetDir = Split-Path $target -Parent
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            
            # 移动文件
            Move-Item -Path $source -Destination $target -Force
            Write-Host "✓ 已归档: $source -> $target" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host "✗ 错误: $source - $($_.Exception.Message)" -ForegroundColor Red
            $errorCount++
            $errors += "$source : $($_.Exception.Message)"
        }
    }
    else {
        Write-Host "⊘ 跳过: $source (文件不存在)" -ForegroundColor Yellow
        $skipCount++
    }
}

Write-Host "`n归档完成统计:" -ForegroundColor Cyan
Write-Host "  成功: $successCount" -ForegroundColor Green
Write-Host "  跳过: $skipCount" -ForegroundColor Yellow
Write-Host "  错误: $errorCount" -ForegroundColor Red

if ($errors.Count -gt 0) {
    Write-Host "`n错误详情:" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  $error" -ForegroundColor Red
    }
}

Write-Host "`n归档操作完成！" -ForegroundColor Green
