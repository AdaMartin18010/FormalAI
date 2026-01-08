#!/usr/bin/env python3
"""
自动更新提醒脚本
检查项目文档是否需要更新，并生成更新提醒报告
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 需要检查的文档路径
DOCS_TO_CHECK = [
    "concepts/03-Scaling Law与收敛分析/README.md",
    "concepts/04-AI意识与认知模拟/README.md",
    "docs/07-alignment-safety/07.1-对齐理论/README.md",
    "docs/13-neural-symbolic/13.1-神经符号AI/README.md",
    "docs/05-multimodal-ai/05.1-视觉语言模型/README.md",
    "docs/00-foundations/00-mathematical-foundations/01-category-theory.md",
]

# 跟踪文档路径
TRACKER_DOC = "docs/LATEST_DEVELOPMENTS_TRACKER.md"


def check_last_update_date(file_path: Path) -> Tuple[datetime, bool]:
    """检查文件最后更新日期"""
    if not file_path.exists():
        return None, False
    
    # 读取文件内容
    content = file_path.read_text(encoding='utf-8')
    
    # 查找最后更新日期
    patterns = [
        r'最后更新[：:]\s*(\d{4}-\d{2}-\d{2})',
        r'Last updated[：:]\s*(\d{4}-\d{2}-\d{2})',
        r'**最后更新**[：:]\s*(\d{4}-\d{2}-\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            try:
                date_str = match.group(1)
                date = datetime.strptime(date_str, '%Y-%m-%d')
                return date, True
            except ValueError:
                continue
    
    # 如果找不到日期，使用文件修改时间
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    return mtime, False


def check_2025_section(file_path: Path) -> bool:
    """检查文件是否包含2025年最新发展章节"""
    if not file_path.exists():
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    # 检查是否包含2025年最新发展章节
    patterns = [
        r'2025年最新发展',
        r'Latest Developments 2025',
        r'十一、2025年最新发展',
        r'十、2025年最新发展',
    ]
    
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)


def generate_update_report() -> str:
    """生成更新提醒报告"""
    report_lines = []
    report_lines.append("# 文档更新提醒报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 检查每个文档
    needs_update = []
    up_to_date = []
    
    for doc_path_str in DOCS_TO_CHECK:
        doc_path = PROJECT_ROOT / doc_path_str
        last_update, has_date = check_last_update_date(doc_path)
        has_2025_section = check_2025_section(doc_path)
        
        days_ago = None
        if last_update:
            days_ago = (datetime.now() - last_update).days
        
        status = "✅"
        if days_ago and days_ago > 90:
            status = "⚠️"
            needs_update.append((doc_path_str, days_ago, has_2025_section))
        elif days_ago and days_ago > 60:
            status = "🔄"
            needs_update.append((doc_path_str, days_ago, has_2025_section))
        else:
            up_to_date.append((doc_path_str, days_ago, has_2025_section))
    
    # 需要更新的文档
    if needs_update:
        report_lines.append("## ⚠️ 需要更新的文档")
        report_lines.append("")
        for doc_path, days_ago, has_2025 in needs_update:
            report_lines.append(f"- **{doc_path}**")
            if days_ago:
                report_lines.append(f"  - 最后更新：{days_ago}天前")
            if has_2025:
                report_lines.append(f"  - ✅ 包含2025年最新发展章节")
            else:
                report_lines.append(f"  - ❌ 缺少2025年最新发展章节")
            report_lines.append("")
    
    # 最新的文档
    if up_to_date:
        report_lines.append("## ✅ 最新的文档")
        report_lines.append("")
        for doc_path, days_ago, has_2025 in up_to_date:
            report_lines.append(f"- **{doc_path}**")
            if days_ago:
                report_lines.append(f"  - 最后更新：{days_ago}天前")
            if has_2025:
                report_lines.append(f"  - ✅ 包含2025年最新发展章节")
            report_lines.append("")
    
    # 建议
    report_lines.append("## 📋 更新建议")
    report_lines.append("")
    report_lines.append("1. **每周检查**：每周五检查arXiv最新预印本")
    report_lines.append("2. **月度更新**：每月更新一次关键文档")
    report_lines.append("3. **季度审查**：每季度进行全面审查和更新")
    report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    """主函数"""
    print("正在生成更新提醒报告...")
    
    report = generate_update_report()
    
    # 保存报告
    report_path = PROJECT_ROOT / "docs" / "UPDATE_REMINDER_REPORT.md"
    report_path.write_text(report, encoding='utf-8')
    
    print(f"✅ 报告已生成：{report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
