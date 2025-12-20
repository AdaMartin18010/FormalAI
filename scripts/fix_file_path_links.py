#!/usr/bin/env python3
"""
修复文件路径链接问题

问题：
1. URL编码问题（如 `Level%201%20黑箱经验层.md` 应该是 `Level 1 黑箱经验层.md`）
2. 文件名中的冒号问题（如 `Level 2: 模式提炼层.md` 应该是 `Level 2-模式提炼层.md`）
3. 目录名空格问题（如 `03-Scaling Law 与收敛分析` 应该是 `03-Scaling Law与收敛分析`）

使用方法：
    python scripts/fix_file_path_links.py [--path PATH] [--dry-run]
"""

import os
import re
from pathlib import Path
import argparse
from urllib.parse import unquote

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

def fix_file_path_links(file_path: Path, dry_run: bool = False) -> bool:
    """修复文件中的路径链接"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixed = False
        
        # 模式1：修复URL编码的文件名
        # 如：`02.2.1-Level%201%20黑箱经验层.md` -> `02.2.1-Level 1 黑箱经验层.md`
        def decode_url(match):
            nonlocal fixed
            encoded = match.group(1)
            try:
                decoded = unquote(encoded)
                fixed = True
                return decoded
            except:
                return match.group(0)
        
        # 匹配URL编码的文件名
        url_pattern = r'([\w\.-]+%[\w%]+\.md)'
        content = re.sub(url_pattern, decode_url, content)
        
        # 模式2：修复冒号问题
        # 如：`Level 2: 模式提炼层.md` -> `Level 2-模式提炼层.md`
        colon_pattern = r'(\d+:\s+[^)]+\.md)'
        def fix_colon(match):
            nonlocal fixed
            text = match.group(1)
            fixed_text = text.replace(': ', '-', 1)
            fixed = True
            return fixed_text
        
        content = re.sub(colon_pattern, fix_colon, content)
        
        # 模式3：修复目录名空格问题（需要检查实际目录名）
        # 这个比较复杂，需要逐个检查
        
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
    parser = argparse.ArgumentParser(description='修复文件路径链接问题')
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
        skip_files = ['README.md', 'INDEX.md', 'CHANGELOG.md', 'concepts_cross_ref_report.md', 'concepts_cross_ref_report_v2.md']
        if md_file.name in skip_files:
            continue
        
        checked_count += 1
        
        # 修复链接
        if fix_file_path_links(md_file, args.dry_run):
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
