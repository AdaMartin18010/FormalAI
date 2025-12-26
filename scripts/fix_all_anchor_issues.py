#!/usr/bin/env python3
"""
修复所有锚点问题：将双连字符替换为单连字符
"""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def fix_file(file_path: Path) -> bool:
    """修复文件中的锚点问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 修复锚点中的双连字符（--改为-）
        # 匹配模式：[文本](#锚点) 或 [文本](文件.md#锚点)
        def fix_anchor(match):
            nonlocal fixed
            full_link = match.group(0)
            link_text = match.group(1)
            link_target = match.group(2)

            # 如果是锚点链接（包含#）
            if '#' in link_target:
                file_part, anchor_part = link_target.split('#', 1)
                # 修复双连字符
                if '--' in anchor_part:
                    new_anchor = anchor_part.replace('--', '-')
                    new_target = f"{file_part}#{new_anchor}" if file_part else f"#{new_anchor}"
                    fixed = True
                    return f"[{link_text}]({new_target})"

            return full_link

        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        content = re.sub(link_pattern, fix_anchor, content)

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
        if fix_file(md_file):
            rel_path = str(md_file).replace(str(ROOT_DIR), '').lstrip('\\/')
            print(f"✓ 已修复: {rel_path}")
            fixed_count += 1

    print(f"\n处理完成:")
    print(f"  - 检查文件: {checked_count} 个")
    print(f"  - 修复文件: {fixed_count} 个")

if __name__ == "__main__":
    main()
