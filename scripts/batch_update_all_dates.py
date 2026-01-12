#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新所有concepts目录下文档的日期
"""
import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
TARGET_DIR = ROOT_DIR / "concepts"

# 日期替换规则
DATE_REPLACEMENTS = [
    (r'\*\*最后更新\*\*：2025-11-10', '**最后更新**：2025-01-15'),
    (r'最后更新.*：2025-11-10', '**最后更新**：2025-01-15'),
    (r'最后更新.*：2025-01-XX', '**最后更新**：2025-01-15'),
    (r'最后更新.*：2025-XX-XX', '**最后更新**：2025-01-15'),
    (r'2025-01-XX', '2025-01-15'),
    (r'2025-XX-XX', '2025-01-15'),
]

def update_file(file_path):
    """更新单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for pattern, replacement in DATE_REPLACEMENTS:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """主函数"""
    updated = 0
    total = 0
    errors = 0
    
    if not TARGET_DIR.exists():
        print(f"Directory not found: {TARGET_DIR}")
        return
    
    for md_file in TARGET_DIR.rglob("*.md"):
        total += 1
        try:
            if update_file(md_file):
                print(f"✓ Updated: {md_file.relative_to(ROOT_DIR)}")
                updated += 1
        except Exception as e:
            print(f"✗ Error: {md_file.relative_to(ROOT_DIR)} - {e}")
            errors += 1
    
    print(f"\n完成统计:")
    print(f"  - 总文件数: {total}")
    print(f"  - 已更新: {updated}")
    print(f"  - 错误: {errors}")
    print(f"  - 跳过: {total - updated - errors}")

if __name__ == "__main__":
    main()
