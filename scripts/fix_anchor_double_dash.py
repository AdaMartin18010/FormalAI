#!/usr/bin/env python3
"""
修复锚点链接中的双连字符问题

问题：链接中使用 `--`（双连字符）替代箭头符号，但实际锚点应该是 `-`（单连字符）

使用方法：
    python scripts/fix_anchor_double_dash.py [--path PATH] [--dry-run]
"""

import os
import re
from pathlib import Path
import argparse

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

def fix_double_dash_anchors(file_path: Path, dry_run: bool = False) -> bool:
    """修复文件中的双连字符锚点"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 查找所有包含双连字符的锚点链接
        # 模式：`#数字-文字--文字` 或 `(#数字-文字--文字)`
        pattern = r'(\(#\d+[-\w]+)--([-\w]+\))'

        def replace_double_dash(match):
            nonlocal fixed
            fixed = True
            # 将双连字符替换为单连字符
            return match.group(1) + '-' + match.group(2)

        content = re.sub(pattern, replace_double_dash, content)

        if fixed and content != original_content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            return True

        return False
    except Exception as e:
        print(f"警告：无法处理文件 {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='修复锚点链接中的双连字符问题')
    parser.add_argument('--path', type=str, default=None, help='要修复的路径（默认为concepts目录）')
    parser.add_argument('--dry-run', action='store_true', help='只检查不修复')

    args = parser.parse_args()

    if args.path:
        target_path = Path(args.path)
    else:
        target_path = ROOT_DIR / "concepts"

    fixed_count = 0
    checked_count = 0

    # 遍历所有Markdown文件
    for md_file in target_path.rglob("*.md"):
        # 跳过特定文件
        skip_files = ['README.md', 'INDEX.md', 'CHANGELOG.md', 'concepts_cross_ref_report.md']
        if md_file.name in skip_files:
            continue

        checked_count += 1

        # 修复链接
        if fix_double_dash_anchors(md_file, args.dry_run):
            rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
            if args.dry_run:
                print(f"需要修复: {rel_path}")
            else:
                print(f"✓ 已修复: {rel_path}")
            fixed_count += 1

    print(f"\n处理完成:")
    print(f"  - 检查文件: {checked_count} 个")
    if args.dry_run:
        print(f"  - 需要修复: {fixed_count} 个文件")
        print(f"  - 使用 --dry-run 模式，未实际修复")
    else:
        print(f"  - 修复文件: {fixed_count} 个")

if __name__ == "__main__":
    main()
