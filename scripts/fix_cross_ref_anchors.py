#!/usr/bin/env python3
"""
修复交叉引用中的锚点问题

功能：
    - 修复"与三层模型的关系"章节中的锚点链接
    - 确保锚点格式正确

使用方法：
    python scripts/fix_cross_ref_anchors.py [--path PATH]
"""

import os
import re
from pathlib import Path
import argparse

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 锚点映射规则
ANCHOR_FIXES = {
    r'#61-执行层--控制层': r'#61-执行层--控制层',
    r'#62-执行层--数据层': r'#62-执行层--数据层',
    r'#71-执行层--数据层': r'#71-执行层--数据层',
    r'#72-执行层--控制层': r'#72-执行层--控制层',
    r'#81-执行层--数据层': r'#81-执行层--数据层',
    r'#82-执行层--控制层': r'#82-执行层--控制层',
    r'#61-控制层--数据层': r'#61-控制层--数据层',
    r'#62-控制层--执行层': r'#62-控制层--执行层',
    r'#51-控制层--数据层': r'#51-控制层--数据层',
    r'#52-控制层--执行层': r'#52-控制层--执行层',
    r'#71-控制层--数据层': r'#71-控制层--数据层',
    r'#72-控制层--执行层': r'#72-控制层--执行层',
    r'#71-数据层--执行层': r'#71-数据层--执行层',
    r'#72-数据层--控制层': r'#72-数据层--控制层',
    r'#91-数据层--执行层': r'#91-数据层--执行层',
    r'#92-数据层--控制层': r'#92-数据层--控制层',
    r'#61-数据层--控制层': r'#61-数据层--控制层',
    r'#62-数据层--执行层': r'#62-数据层--执行层',
}

def title_to_anchor(title: str) -> str:
    """将标题转换为锚点格式"""
    # 移除Markdown链接
    title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
    # 转换为小写，替换空格为连字符
    anchor = title.lower()
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    anchor = re.sub(r'[\s_]+', '-', anchor)
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

def fix_anchor_links(file_path: Path, anchors: set) -> bool:
    """修复文件中的锚点链接"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 查找所有链接
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        def replace_link(match):
            link_text = match.group(1)
            link_target = match.group(2)

            # 检查是否是锚点链接
            if '#' in link_target:
                file_part, anchor_part = link_target.split('#', 1)
                # 如果文件部分为空，说明是当前文件的锚点
                if not file_part or file_part == file_path.name:
                    # 检查锚点是否存在
                    if anchor_part not in anchors:
                        # 尝试修复锚点
                        # 如果链接文本看起来像标题，尝试从文本生成锚点
                        potential_anchor = title_to_anchor(link_text)
                        if potential_anchor in anchors:
                            new_target = f"#{potential_anchor}" if not file_part else f"{file_part}#{potential_anchor}"
                            nonlocal fixed
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
    parser = argparse.ArgumentParser(description='修复交叉引用中的锚点问题')
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
        if md_file.name in ['README.md', 'INDEX.md', 'CHANGELOG.md']:
            continue

        checked_count += 1

        # 提取锚点
        anchors = extract_anchors_from_file(md_file)

        # 修复链接
        if not args.dry_run:
            if fix_anchor_links(md_file, anchors):
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
                            rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
                            print(f"需要修复: {rel_path} - {link_target}")

    print(f"\n处理完成:")
    print(f"  - 检查文件: {checked_count} 个")
    if not args.dry_run:
        print(f"  - 修复文件: {fixed_count} 个")
    else:
        print(f"  - 使用 --dry-run 模式，未实际修复")

if __name__ == "__main__":
    main()
