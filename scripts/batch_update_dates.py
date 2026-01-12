#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新文档中的占位日期
"""
import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
TARGET_DIRS = [
    "concepts/06-AI反实践判定系统",
]

# 日期替换规则
DATE_REPLACEMENTS = [
    (r'\*\*最后更新\*\*：2025-11-10', '**最后更新**：2025-01-15'),
    (r'最后更新.*：2025-11-10', '**最后更新**：2025-01-15'),
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
    
    for target_dir in TARGET_DIRS:
        dir_path = ROOT_DIR / target_dir
        if not dir_path.exists():
            print(f"Directory not found: {dir_path}")
            continue
        
        for md_file in dir_path.rglob("*.md"):
            total += 1
            if update_file(md_file):
                print(f"✓ Updated: {md_file.relative_to(ROOT_DIR)}")
                updated += 1
    
    print(f"\n完成: 更新了 {updated}/{total} 个文件")

if __name__ == "__main__":
    main()
