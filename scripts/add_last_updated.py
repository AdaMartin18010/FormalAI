#!/usr/bin/env python3
"""
批量添加"最后更新"标记到concepts模块的文档

使用方法：
    python scripts/add_last_updated.py

功能：
    - 扫描concepts目录下的所有.md文件
    - 检查是否已有"最后更新"标记
    - 如果没有，在文档末尾添加标记
"""

import os
import re
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
CONCEPTS_DIR = ROOT_DIR / "concepts"

# 要添加的标记
LAST_UPDATED_MARKER = """---

**最后更新**：2025-01-XX
**维护者**：FormalAI项目组"""

def has_last_updated(content: str) -> bool:
    """检查文档是否已有"最后更新"标记"""
    patterns = [
        r'^最后更新',
        r'^最后更新日期',
        r'^最后修改',
        r'\*\*最后更新\*\*',
    ]
    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
            return True
    return False

def add_last_updated(file_path: Path) -> bool:
    """为文档添加"最后更新"标记"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 如果已有标记，跳过
        if has_last_updated(content):
            return False

        # 移除末尾的空白行
        content = content.rstrip()

        # 如果末尾没有分隔线，添加一个
        if not content.endswith('---'):
            content += '\n\n---'

        # 添加"最后更新"标记
        content += LAST_UPDATED_MARKER

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    processed = 0
    skipped = 0
    errors = 0

    # 排除的文件和目录
    exclude_dirs = {'__pycache__', '.git'}
    exclude_files = {
        'README.md', 'INDEX.md', 'CHANGELOG.md',
        'PHASE1_*.md', 'CONCEPTS_*.md', 'DOCUMENT_*.md',
        'REPAIR_*.md', 'VIEW_*.md', 'AI_ARGUMENTS_*.md'
    }

    # 遍历所有.md文件
    for md_file in CONCEPTS_DIR.rglob("*.md"):
        # 跳过排除的文件
        if md_file.name in exclude_files:
            continue

        # 跳过排除的目录
        if any(part in exclude_dirs for part in md_file.parts):
            continue

        # 处理文件
        if add_last_updated(md_file):
            print(f"✓ 已处理: {md_file.relative_to(ROOT_DIR)}")
            processed += 1
        else:
            skipped += 1

    print(f"\n处理完成:")
    print(f"  - 已添加标记: {processed} 个文件")
    print(f"  - 已跳过（已有标记）: {skipped} 个文件")
    print(f"  - 错误: {errors} 个文件")

if __name__ == "__main__":
    main()
