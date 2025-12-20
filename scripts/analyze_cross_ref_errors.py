#!/usr/bin/env python3
"""
分析交叉引用错误报告，分类错误类型

使用方法：
    python scripts/analyze_cross_ref_errors.py [report_file]
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

def analyze_report(report_file: Path):
    """分析错误报告"""
    errors_by_type = defaultdict(list)
    errors_by_file = defaultdict(list)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取错误行
    error_pattern = r'\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|'
    for match in re.finditer(error_pattern, content):
        file_path = match.group(1)
        link_text = match.group(2)
        link_target = match.group(3)
        error_msg = match.group(4).strip()
        
        # 分类错误类型
        if '文件未在扫描范围内' in error_msg:
            error_type = '扫描范围问题（可忽略）'
        elif '文件不存在' in error_msg:
            error_type = '文件不存在'
        elif '锚点不存在' in error_msg:
            error_type = '锚点不存在'
        else:
            error_type = '其他错误'
        
        errors_by_type[error_type].append((file_path, link_text, link_target, error_msg))
        errors_by_file[file_path].append((link_text, link_target, error_msg))
    
    # 输出统计
    print("=" * 80)
    print("交叉引用错误分析报告")
    print("=" * 80)
    print()
    
    print("## 错误类型统计")
    print()
    for error_type, errors in sorted(errors_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"- **{error_type}**: {len(errors)} 个")
    print()
    
    print("## 需要修复的错误（排除扫描范围问题）")
    print()
    
    real_errors = []
    for error_type, errors in errors_by_type.items():
        if '扫描范围问题' not in error_type:
            real_errors.extend(errors)
    
    print(f"**总计**: {len(real_errors)} 个需要修复的错误")
    print()
    
    # 按文件分组
    print("## 按文件分组的错误")
    print()
    
    files_with_errors = defaultdict(list)
    for file_path, link_text, link_target, error_msg in real_errors:
        files_with_errors[file_path].append((link_text, link_target, error_msg))
    
    # 只显示前20个文件
    for i, (file_path, errors) in enumerate(sorted(files_with_errors.items())[:20]):
        print(f"### {i+1}. `{file_path}` ({len(errors)} 个错误)")
        for link_text, link_target, error_msg in errors[:5]:  # 每个文件只显示前5个
            print(f"  - `{link_text}` -> `{link_target}`: {error_msg}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
        print()
    
    if len(files_with_errors) > 20:
        print(f"... 还有 {len(files_with_errors) - 20} 个文件有错误")
    
    return len(real_errors), len(files_with_errors)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        report_file = Path(sys.argv[1])
    else:
        report_file = Path("concepts/concepts_cross_ref_report.md")
    
    if not report_file.exists():
        print(f"错误：报告文件不存在: {report_file}")
        sys.exit(1)
    
    real_error_count, files_count = analyze_report(report_file)
    
    print()
    print("=" * 80)
    print(f"总结：需要修复 {real_error_count} 个错误，涉及 {files_count} 个文件")
    print("=" * 80)
