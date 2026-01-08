#!/usr/bin/env python3
"""
文档质量检查脚本
检查文档的完整性、一致性和质量
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# 质量检查标准
QUALITY_STANDARDS = {
    'min_length': 100,  # 最小文档长度（行）
    'has_toc': True,  # 是否有目录
    'has_references': True,  # 是否有参考文献
    'has_2025_section': False,  # 是否有2025年最新发展章节（可选）
    'link_validity': True,  # 链接有效性
}

def find_markdown_files(root_dir: str) -> List[Path]:
    """查找所有Markdown文件"""
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过隐藏目录和特定目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != 'archive']
        
        for file in files:
            if file.endswith('.md'):
                md_files.append(Path(root) / file)
    return md_files

def check_document_structure(content: str, file_path: Path) -> Dict[str, any]:
    """检查文档结构"""
    issues = []
    lines = content.split('\n')
    
    # 检查最小长度
    if len(lines) < QUALITY_STANDARDS['min_length']:
        issues.append(f"文档过短（{len(lines)}行，建议≥{QUALITY_STANDARDS['min_length']}行）")
    
    # 检查是否有标题
    has_title = any(line.startswith('# ') for line in lines[:10])
    if not has_title:
        issues.append("缺少主标题（# 标题）")
    
    # 检查是否有目录
    if QUALITY_STANDARDS['has_toc']:
        has_toc = any(re.search(r'^##+\s*目录|^##+\s*Table of Contents|^##+\s*Contents', line, re.IGNORECASE) for line in lines[:50])
        if not has_toc:
            issues.append("建议添加目录")
    
    # 检查是否有参考文献
    if QUALITY_STANDARDS['has_references']:
        has_refs = any(re.search(r'##+\s*参考|##+\s*Reference|##+\s*参考文献', line, re.IGNORECASE) for line in lines)
        if not has_refs:
            issues.append("建议添加参考文献部分")
    
    # 检查是否有2025年最新发展章节（对于主要模块）
    if 'README.md' in str(file_path) or 'category-theory.md' in str(file_path):
        has_2025 = any(re.search(r'2025年最新发展|Latest Developments 2025', line, re.IGNORECASE) for line in lines)
        if not has_2025:
            issues.append("建议添加2025年最新发展章节")
    
    return {
        'file': str(file_path),
        'line_count': len(lines),
        'has_title': has_title,
        'issues': issues,
        'quality_score': max(0, 100 - len(issues) * 10)
    }

def check_links(content: str, file_path: Path) -> List[Dict[str, any]]:
    """检查文档中的链接"""
    link_issues = []
    
    # 查找所有Markdown链接
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    links = re.findall(link_pattern, content)
    
    for link_text, link_url in links:
        # 检查本地链接
        if link_url.startswith('./') or link_url.startswith('../') or not link_url.startswith('http'):
            # 解析相对路径
            if link_url.startswith('./'):
                target_path = file_path.parent / link_url[2:]
            elif link_url.startswith('../'):
                target_path = file_path.parent.parent / link_url[3:]
            else:
                target_path = file_path.parent / link_url
            
            # 检查锚点链接
            if '#' in str(target_path):
                file_part, anchor = str(target_path).split('#', 1)
                target_path = Path(file_part)
                # 锚点检查暂时跳过，因为需要解析目标文件
            
            if not target_path.exists():
                link_issues.append({
                    'text': link_text,
                    'url': link_url,
                    'issue': '链接目标不存在'
                })
    
    return link_issues

def check_content_consistency(content: str, file_path: Path) -> List[str]:
    """检查内容一致性"""
    issues = []
    
    # 检查日期格式一致性
    date_patterns = [
        r'2025-01-XX',
        r'2025-01-\d{2}',
        r'2025年1月',
    ]
    dates_found = []
    for pattern in date_patterns:
        dates_found.extend(re.findall(pattern, content))
    
    if len(set(dates_found)) > 1:
        issues.append("日期格式不一致")
    
    # 检查术语一致性
    # 这里可以添加更多术语检查
    
    return issues

def check_file_quality(file_path: Path) -> Dict[str, any]:
    """检查单个文件的质量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'quality_score': 0
        }
    
    structure_result = check_document_structure(content, file_path)
    link_issues = check_links(content, file_path)
    consistency_issues = check_content_consistency(content, file_path)
    
    all_issues = structure_result['issues'] + [f"链接问题: {li['text']} -> {li['issue']}" for li in link_issues] + consistency_issues
    
    return {
        'file': str(file_path),
        'line_count': structure_result['line_count'],
        'has_title': structure_result['has_title'],
        'link_issues': len(link_issues),
        'consistency_issues': len(consistency_issues),
        'total_issues': len(all_issues),
        'issues': all_issues,
        'quality_score': max(0, structure_result['quality_score'] - len(link_issues) * 5 - len(consistency_issues) * 5)
    }

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    md_files = find_markdown_files(str(project_root))
    
    print("=" * 80)
    print("FormalAI项目文档质量检查")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"找到 {len(md_files)} 个Markdown文件\n")
    
    results = []
    for md_file in md_files:
        result = check_file_quality(md_file)
        if 'error' not in result:
            results.append(result)
    
    # 按质量分数排序
    results.sort(key=lambda x: x['quality_score'])
    
    # 显示有问题的文件
    files_with_issues = [r for r in results if r['total_issues'] > 0]
    
    if files_with_issues:
        print(f"⚠️  发现 {len(files_with_issues)} 个文件有质量问题:\n")
        for result in files_with_issues[:20]:  # 只显示前20个
            print(f"📄 {result['file']}")
            print(f"   质量分数: {result['quality_score']}/100")
            print(f"   问题数: {result['total_issues']}")
            if result['issues']:
                for issue in result['issues'][:3]:  # 只显示前3个问题
                    print(f"   - {issue}")
                if len(result['issues']) > 3:
                    print(f"   ... 还有 {len(result['issues']) - 3} 个问题")
            print()
    else:
        print("✅ 所有文档质量检查通过！\n")
    
    # 统计总结
    print("=" * 80)
    print("质量检查总结:")
    print("-" * 80)
    
    total_files = len(results)
    avg_score = sum(r['quality_score'] for r in results) / total_files if total_files > 0 else 0
    high_quality = sum(1 for r in results if r['quality_score'] >= 80)
    medium_quality = sum(1 for r in results if 60 <= r['quality_score'] < 80)
    low_quality = sum(1 for r in results if r['quality_score'] < 60)
    
    print(f"总文件数: {total_files}")
    print(f"平均质量分数: {avg_score:.1f}/100")
    print(f"高质量文档 (≥80分): {high_quality} ({high_quality/total_files*100:.1f}%)" if total_files > 0 else "高质量文档: 0")
    print(f"中等质量文档 (60-79分): {medium_quality} ({medium_quality/total_files*100:.1f}%)" if total_files > 0 else "中等质量文档: 0")
    print(f"低质量文档 (<60分): {low_quality} ({low_quality/total_files*100:.1f}%)" if total_files > 0 else "低质量文档: 0")
    print(f"有问题的文件: {len(files_with_issues)}")
    
    if avg_score >= 80:
        print("\n✅ 整体文档质量优秀！")
    elif avg_score >= 60:
        print("\n⚠️  整体文档质量良好，但仍有改进空间")
    else:
        print("\n❌ 整体文档质量需要改进")

if __name__ == '__main__':
    main()
