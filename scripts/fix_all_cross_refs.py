#!/usr/bin/env python3
"""
批量修复所有交叉引用问题

功能：
    - 修复双连字符锚点（--改为-）
    - 修复URL编码路径
    - 修复路径引用问题
"""

import os
import re
from pathlib import Path
from urllib.parse import unquote

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

def decode_url_path(path: str) -> str:
    """解码URL编码的路径"""
    try:
        return unquote(path)
    except Exception:
        return path

def fix_file(file_path: Path) -> bool:
    """修复文件中的交叉引用问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 1. 修复双连字符锚点（--改为-）
        # 匹配模式：锚点中的双连字符
        def fix_double_dash_anchor(match):
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

        # 匹配所有Markdown链接
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        content = re.sub(link_pattern, fix_double_dash_anchor, content)

        # 2. 修复URL编码路径
        def fix_url_encoded_path(match):
            nonlocal fixed
            full_link = match.group(0)
            link_text = match.group(1)
            link_target = match.group(2)

            # 跳过HTTP链接和纯锚点链接
            if link_target.startswith('http') or (link_target.startswith('#') and '/' not in link_target):
                return full_link

            # 解码URL编码
            decoded = decode_url_path(link_target)
            if decoded != link_target:
                fixed = True
                return f"[{link_text}]({decoded})"

            return full_link

        content = re.sub(link_pattern, fix_url_encoded_path, content)

        # 3. 修复路径引用问题（03-Scaling Law 与收敛分析 -> 03-Scaling Law与收敛分析）
        if '03-Scaling Law 与收敛分析' in content:
            content = content.replace('03-Scaling Law 与收敛分析', '03-Scaling Law与收敛分析')
            fixed = True

        # 4. 修复Level文件的路径引用
        # Level 1 -> Level-1, Level 2 -> Level-2 等
        level_patterns = [
            (r'Level\s+1\s*:', 'Level 1:'),
            (r'Level\s+2\s*:', 'Level 2:'),
            (r'Level\s+3\s*:', 'Level 3:'),
            (r'Level\s+4\s*:', 'Level 4:'),
            (r'Level\s+5\s*:', 'Level 5:'),
        ]

        def fix_level_path(match):
            nonlocal fixed
            full_link = match.group(0)
            link_text = match.group(1)
            link_target = match.group(2)

            # 检查是否是Level文件的路径
            if 'Level' in link_target and ('%20' in link_target or 'Level ' in link_target):
                # 修复URL编码的Level路径
                decoded = decode_url_path(link_target)
                # 确保路径格式正确
                if 'Level ' in decoded:
                    # 替换为正确的文件名格式
                    fixed_path = decoded.replace('Level ', 'Level-').replace(':', '-')
                    if fixed_path != link_target:
                        fixed = True
                        return f"[{link_text}]({fixed_path})"

            return full_link

        content = re.sub(link_pattern, fix_level_path, content)

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
