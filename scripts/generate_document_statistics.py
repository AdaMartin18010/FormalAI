#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成文档统计报告
创建日期：2025-01-XX
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def count_lines(file_path):
    """统计文件行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0

def count_words(file_path):
    """统计文件字数（中英文）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 移除代码块和链接
            content = re.sub(r'```[\s\S]*?```', '', content)
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
            # 统计中文字符和英文单词
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            english_words = len(re.findall(r'\b[a-zA-Z]+\b', content))
            return chinese_chars + english_words
    except:
        return 0

def check_2025_section(file_path):
    """检查是否包含2025年最新发展章节"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return '2025年最新发展' in content or 'Latest Developments 2025' in content
    except:
        return False

def scan_directory(directory):
    """扫描目录，统计文档"""
    stats = {
        'total_files': 0,
        'readme_files': 0,
        'md_files': 0,
        'with_2025_section': 0,
        'total_lines': 0,
        'total_words': 0,
        'by_module': defaultdict(lambda: {'files': 0, 'lines': 0, 'words': 0, 'with_2025': 0})
    }
    
    for root, dirs, files in os.walk(directory):
        # 跳过归档目录和脚本目录
        if 'archive' in root or 'scripts' in root:
            continue
        
        for file in files:
            if file.endswith('.md') or file == 'README.md':
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                
                stats['total_files'] += 1
                if file == 'README.md':
                    stats['readme_files'] += 1
                if file.endswith('.md'):
                    stats['md_files'] += 1
                
                lines = count_lines(file_path)
                words = count_words(file_path)
                has_2025 = check_2025_section(file_path)
                
                stats['total_lines'] += lines
                stats['total_words'] += words
                if has_2025:
                    stats['with_2025_section'] += 1
                
                # 按模块统计
                module = rel_path.split(os.sep)[0] if os.sep in rel_path else 'root'
                stats['by_module'][module]['files'] += 1
                stats['by_module'][module]['lines'] += lines
                stats['by_module'][module]['words'] += words
                if has_2025:
                    stats['by_module'][module]['with_2025'] += 1
    
    return stats

def main():
    """主函数"""
    print("开始生成文档统计报告...\n")
    
    # 扫描docs目录
    docs_stats = scan_directory('docs')
    
    # 扫描concepts目录
    concepts_stats = scan_directory('concepts')
    
    print("=" * 60)
    print("文档统计报告")
    print("=" * 60)
    
    print(f"\n📚 Docs模块统计:")
    print(f"  总文件数: {docs_stats['total_files']}")
    print(f"  README文件: {docs_stats['readme_files']}")
    print(f"  Markdown文件: {docs_stats['md_files']}")
    print(f"  包含2025年最新发展章节: {docs_stats['with_2025_section']}")
    print(f"  总行数: {docs_stats['total_lines']:,}")
    print(f"  总字数: {docs_stats['total_words']:,}")
    
    print(f"\n📖 Concepts模块统计:")
    print(f"  总文件数: {concepts_stats['total_files']}")
    print(f"  README文件: {concepts_stats['readme_files']}")
    print(f"  Markdown文件: {concepts_stats['md_files']}")
    print(f"  包含2025年最新发展章节: {concepts_stats['with_2025_section']}")
    print(f"  总行数: {concepts_stats['total_lines']:,}")
    print(f"  总字数: {concepts_stats['total_words']:,}")
    
    print(f"\n📊 总体统计:")
    total_files = docs_stats['total_files'] + concepts_stats['total_files']
    total_with_2025 = docs_stats['with_2025_section'] + concepts_stats['with_2025_section']
    total_lines = docs_stats['total_lines'] + concepts_stats['total_lines']
    total_words = docs_stats['total_words'] + concepts_stats['total_words']
    
    print(f"  总文件数: {total_files}")
    print(f"  包含2025年最新发展章节: {total_with_2025} ({total_with_2025/total_files*100:.1f}%)")
    print(f"  总行数: {total_lines:,}")
    print(f"  总字数: {total_words:,}")
    
    print(f"\n📁 按模块统计（Docs）:")
    for module, data in sorted(docs_stats['by_module'].items()):
        if data['files'] > 0:
            print(f"  {module}:")
            print(f"    文件数: {data['files']}")
            print(f"    行数: {data['lines']:,}")
            print(f"    字数: {data['words']:,}")
            print(f"    包含2025章节: {data['with_2025']}")

if __name__ == "__main__":
    main()
