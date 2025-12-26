#!/usr/bin/env python3
"""
正确修复锚点：将双连字符改为单连字符（匹配实际生成的锚点格式）
"""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 需要修复的锚点模式映射（双连字符改为单连字符）
ANCHOR_FIXES = {
    r'#61-执行层--控制层': r'#61-执行层-控制层',
    r'#62-执行层--数据层': r'#62-执行层-数据层',
    r'#71-执行层--数据层': r'#71-执行层-数据层',
    r'#72-执行层--控制层': r'#72-执行层-控制层',
    r'#81-执行层--数据层': r'#81-执行层-数据层',
    r'#82-执行层--控制层': r'#82-执行层-控制层',
    r'#61-控制层--数据层': r'#61-控制层-数据层',
    r'#62-控制层--执行层': r'#62-控制层-执行层',
    r'#51-控制层--数据层': r'#51-控制层-数据层',
    r'#52-控制层--执行层': r'#52-控制层-执行层',
    r'#71-控制层--数据层': r'#71-控制层-数据层',
    r'#72-控制层--执行层': r'#72-控制层-执行层',
    r'#71-数据层--执行层': r'#71-数据层-执行层',
    r'#72-数据层--控制层': r'#72-数据层-控制层',
    r'#91-数据层--执行层': r'#91-数据层-执行层',
    r'#92-数据层--控制层': r'#92-数据层-控制层',
    r'#61-数据层--控制层': r'#61-数据层-控制层',
    r'#62-数据层--执行层': r'#62-数据层-执行层',
    r'#四控制层--数据层协同': r'#四控制层-数据层协同',
    r'#五数据层--执行层协同': r'#五数据层-执行层协同',
    r'#六执行层--控制层协同': r'#六执行层-控制层协同',
}

def fix_file(file_path: Path) -> bool:
    """修复文件中的锚点链接"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        fixed = False

        # 修复所有锚点链接
        for old_anchor, new_anchor in ANCHOR_FIXES.items():
            if old_anchor in content:
                content = content.replace(old_anchor, new_anchor)
                fixed = True

        if fixed and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False
    except Exception as e:
        print(f"警告：无法修复文件 {file_path}: {e}")
        return False

def main():
    target_path = ROOT_DIR / 'concepts' / '01-AI三层模型架构'
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
