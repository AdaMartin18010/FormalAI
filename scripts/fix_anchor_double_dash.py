#!/usr/bin/env python3
"""
修复锚点中的双连字符问题

功能：
    - 将锚点中的双连字符（--）替换为单连字符（-）
    - 匹配实际标题生成的锚点格式
"""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def title_to_anchor(title: str) -> str:
    """将标题转换为锚点格式（模拟Markdown的锚点生成规则）"""
    # 移除Markdown链接
    title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
    # 转换为小写
    anchor = title.lower()
    # 替换箭头为连字符
    anchor = anchor.replace('→', '-')
    anchor = anchor.replace('→', '-')
    # 移除所有非字母数字和连字符的字符
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    # 将空格和多个连字符替换为单个连字符
    anchor = re.sub(r'[\s_]+', '-', anchor)
    anchor = re.sub(r'-+', '-', anchor)
    # 移除首尾连字符
    anchor = anchor.strip('-')
    return anchor

def extract_anchors_from_file(file_path: Path) -> dict:
    """从文件中提取所有标题和对应的锚点"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        anchors = {}
        # 提取所有标题
        pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            anchor = title_to_anchor(title)
            anchors[anchor] = title

        return anchors
    except Exception as e:
        print(f"警告：无法读取文件 {file_path}: {e}")
        return {}

def fix_anchors_in_file(file_path: Path) -> bool:
    """修复文件中的锚点链接"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 提取文件中的所有锚点
        anchors = extract_anchors_from_file(file_path)

        # 查找所有Markdown链接
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        def replace_link(match):
            nonlocal fixed
            link_text = match.group(1)
            link_target = match.group(2)

            # 检查是否是锚点链接
            if '#' in link_target:
                file_part, anchor_part = link_target.split('#', 1)
                # 如果文件部分为空或者是当前文件名，说明是当前文件的锚点
                if not file_part or file_part == file_path.name:
                    # 检查锚点是否存在
                    if anchor_part not in anchors:
                        # 尝试修复双连字符
                        fixed_anchor = anchor_part.replace('--', '-')
                        if fixed_anchor in anchors:
                            new_target = f"#{fixed_anchor}" if not file_part else f"{file_part}#{fixed_anchor}"
                            fixed = True
                            return f"[{link_text}]({new_target})"
                        # 尝试从链接文本生成锚点
                        potential_anchor = title_to_anchor(link_text)
                        if potential_anchor in anchors:
                            new_target = f"#{potential_anchor}" if not file_part else f"{file_part}#{potential_anchor}"
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
    target_path = ROOT_DIR / 'concepts'
    if not target_path.exists():
        print(f"错误：路径不存在：{target_path}")
        return

    fixed_count = 0
    checked_count = 0

    # 遍历所有Markdown文件
    for md_file in target_path.rglob("*.md"):
        # 跳过报告文件
        if 'report' in md_file.name.lower() or 'cross_ref' in md_file.name.lower():
            continue

        checked_count += 1
        if fix_anchors_in_file(md_file):
            rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
            print(f"✓ 已修复: {rel_path}")
            fixed_count += 1

    print(f"\n处理完成:")
    print(f"  - 检查文件: {checked_count} 个")
    print(f"  - 修复文件: {fixed_count} 个")

if __name__ == "__main__":
    main()
