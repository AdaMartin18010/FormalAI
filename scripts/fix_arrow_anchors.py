#!/usr/bin/env python3
"""
修复箭头符号锚点问题

功能：
    - 修复"与三层模型的关系"章节中的箭头符号锚点
    - 将 `→` 符号正确转换为锚点格式

使用方法：
    python scripts/fix_arrow_anchors.py [--path PATH] [--dry-run]
"""

import os
import re
from pathlib import Path
import argparse

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

def title_to_anchor(title: str) -> str:
    """将标题转换为锚点格式"""
    # 移除Markdown链接
    title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
    # 转换为小写，替换箭头为连字符
    anchor = title.lower()
    anchor = anchor.replace('→', '-')
    anchor = anchor.replace('→', '-')  # 确保替换
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    anchor = re.sub(r'[\s_]+', '-', anchor)
    anchor = re.sub(r'-+', '-', anchor)  # 合并多个连字符
    anchor = anchor.strip('-')
    return anchor

def extract_anchors_from_file(file_path: Path) -> set:
    """从文件中提取所有锚点"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        anchors = set()
        # 提取标题作为锚点
        pattern = r'^#{1,6}\s+(.+)$'
        for match in re.finditer(pattern, content, re.MULTILINE):
            title = match.group(1).strip()
            anchor = title_to_anchor(title)
            anchors.add(anchor)

        return anchors
    except Exception as e:
        print(f"警告：无法读取文件 {file_path}: {e}")
        return set()

def fix_arrow_anchors(file_path: Path, anchors: set) -> bool:
    """修复文件中的箭头符号锚点"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 查找所有链接
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        def replace_link(match):
            nonlocal fixed
            link_text = match.group(1)
            link_target = match.group(2)

            # 检查是否是锚点链接
            if '#' in link_target:
                file_part, anchor_part = link_target.split('#', 1)
                # 如果文件部分为空，说明是当前文件的锚点
                if not file_part or file_part == file_path.name:
                    # 检查锚点是否存在
                    if anchor_part not in anchors:
                        # 尝试从链接文本生成锚点
                        potential_anchor = title_to_anchor(link_text)
                        if potential_anchor in anchors:
                            new_target = f"#{potential_anchor}" if not file_part else f"{file_part}#{potential_anchor}"
                            fixed = True
                            return f"[{link_text}]({new_target})"
                        # 尝试修复双连字符问题
                        elif anchor_part.replace('--', '-') in anchors:
                            fixed_anchor = anchor_part.replace('--', '-')
                            new_target = f"#{fixed_anchor}" if not file_part else f"{file_part}#{fixed_anchor}"
                            fixed = True
                            return f"[{link_text}]({new_target})"

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
    parser = argparse.ArgumentParser(description='修复箭头符号锚点问题')
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

        # 提取锚点
        anchors = extract_anchors_from_file(md_file)

        # 修复链接
        if not args.dry_run:
            if fix_arrow_anchors(md_file, anchors):
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
                if '#' in link_target:
                    file_part, anchor_part = link_target.split('#', 1)
                    if not file_part or file_part == md_file.name:
                        if anchor_part not in anchors:
                            # 检查是否是双连字符问题
                            if anchor_part.replace('--', '-') in anchors:
                                rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
                                print(f"需要修复: {rel_path} - {link_target} -> #{anchor_part.replace('--', '-')}")

    print(f"\n处理完成:")
    print(f"  - 检查文件: {checked_count} 个")
    if not args.dry_run:
        print(f"  - 修复文件: {fixed_count} 个")
    else:
        print(f"  - 使用 --dry-run 模式，未实际修复")

if __name__ == "__main__":
    main()
