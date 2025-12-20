#!/usr/bin/env python3
"""
修复文件路径问题

功能：
    - 修复URL编码的文件路径
    - 修复文件路径引用错误
    - 验证文件是否存在

使用方法：
    python scripts/fix_file_paths.py [--path PATH] [--dry-run]
"""

import os
import re
from pathlib import Path
import argparse
from urllib.parse import unquote

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

def decode_url_path(path: str) -> str:
    """解码URL编码的路径"""
    try:
        return unquote(path)
    except Exception:
        return path

def find_actual_file(reference_path: Path, link_target: str) -> Path:
    """查找实际文件路径"""
    # 解码URL编码
    decoded_target = decode_url_path(link_target)

    # 如果是相对路径，从引用文件所在目录解析
    if not decoded_target.startswith('/') and not decoded_target.startswith('http'):
        # 处理相对路径
        if decoded_target.startswith('../'):
            # 向上级目录
            target_path = reference_path.parent.parent / decoded_target[3:]
        elif decoded_target.startswith('./'):
            # 当前目录
            target_path = reference_path.parent / decoded_target[2:]
        else:
            # 同级目录
            target_path = reference_path.parent / decoded_target

        # 尝试解析为绝对路径
        target_path = target_path.resolve()

        # 检查文件是否存在
        if target_path.exists():
            return target_path

    return None

def fix_file_paths_in_file(file_path: Path) -> bool:
    """修复文件中的路径问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 查找所有Markdown链接
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        def replace_link(match):
            link_text = match.group(1)
            link_target = match.group(2)

            # 跳过HTTP链接和锚点链接
            if link_target.startswith('http') or link_target.startswith('#'):
                return match.group(0)

            # 解码URL编码
            decoded_target = decode_url_path(link_target)

            # 如果路径包含URL编码，尝试修复
            if '%' in link_target and decoded_target != link_target:
                # 查找实际文件
                actual_file = find_actual_file(file_path, decoded_target)
                if actual_file:
                    # 计算相对路径
                    try:
                        rel_path = os.path.relpath(actual_file, file_path.parent)
                        # 规范化路径分隔符
                        rel_path = rel_path.replace('\\', '/')
                        nonlocal fixed
                        fixed = True
                        return f"[{link_text}]({rel_path})"
                    except ValueError:
                        # 如果无法计算相对路径，使用绝对路径
                        pass

            return match.group(0)

        content = re.sub(link_pattern, replace_link, content)

        if fixed and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False
    except Exception as e:
        print(f"警告：无法修复文件 {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='修复文件路径问题')
    parser.add_argument('--path', type=str, default='concepts', help='要修复的路径（默认为concepts目录）')
    parser.add_argument('--dry-run', action='store_true', help='只检查不修复')

    args = parser.parse_args()

    target_path = ROOT_DIR / args.path
    if not target_path.exists():
        print(f"错误：路径不存在：{target_path}")
        return

    fixed_count = 0
    checked_count = 0

    # 遍历所有Markdown文件
    for md_file in target_path.rglob("*.md"):
        # 跳过特定文件
        if md_file.name in ['README.md', 'INDEX.md', 'CHANGELOG.md']:
            continue

        checked_count += 1

        # 修复路径
        if not args.dry_run:
            if fix_file_paths_in_file(md_file):
                rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
                print(f"✓ 已修复: {rel_path}")
                fixed_count += 1
        else:
            # 只检查
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            for match in re.finditer(link_pattern, content):
                link_target = match.group(2)
                if '%' in link_target and not link_target.startswith('http'):
                    decoded = decode_url_path(link_target)
                    if decoded != link_target:
                        rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
                        print(f"需要修复: {rel_path} - {link_target} -> {decoded}")

    print(f"\n处理完成:")
    print(f"  - 检查文件: {checked_count} 个")
    if not args.dry_run:
        print(f"  - 修复文件: {fixed_count} 个")
    else:
        print(f"  - 使用 --dry-run 模式，未实际修复")

if __name__ == "__main__":
    main()
