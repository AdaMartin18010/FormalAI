#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新剩余的占位日期（2025-01-XX, 2025-XX-XX）
"""
import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
TARGET_DIRS = [
    ROOT_DIR / "concepts",
    ROOT_DIR / "docs",
    ROOT_DIR / "Philosophy",
]

# 日期替换规则
DATE_REPLACEMENTS = [
    # 先处理更具体的模式
    (r'下次更新.*：2025-01-XX', '下次更新：2025-01-22'),
    (r'下次检查.*：2025-01-XX', '下次检查：2025-04-01'),
    (r'创建日期.*：2025-11-10', '创建日期：2025-01-10'),
    (r'\*\*创建日期\*\*：2025-11-10', '**创建日期**：2025-01-10'),
    (r'\*\*创建日期\*\*：2025-01-XX', '**创建日期**：2025-01-10'),
    (r'创建日期.*：2025-01-XX', '创建日期：2025-01-10'),
    (r'\*\*最后更新\*\*：2025-01-XX', '**最后更新**：2025-01-15'),
    (r'最后更新.*：2025-01-XX', '最后更新：2025-01-15'),
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
        # 跳过编码错误的文件
        if 'codec' in str(e).lower() or 'encoding' in str(e).lower():
            return False
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """主函数"""
    updated = 0
    total = 0
    errors = 0
    
    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            continue
        
        for md_file in target_dir.rglob("*.md"):
            total += 1
            try:
                if update_file(md_file):
                    print(f"✓ Updated: {md_file.relative_to(ROOT_DIR)}")
                    updated += 1
            except Exception as e:
                # 跳过编码错误的文件
                if 'codec' not in str(e).lower() and 'encoding' not in str(e).lower():
                    print(f"✗ Error: {md_file.relative_to(ROOT_DIR)} - {e}")
                    errors += 1
    
    print(f"\n完成统计:")
    print(f"  - 总文件数: {total}")
    print(f"  - 已更新: {updated}")
    print(f"  - 错误: {errors}")
    print(f"  - 跳过: {total - updated - errors}")

if __name__ == "__main__":
    main()
