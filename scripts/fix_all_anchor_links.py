#!/usr/bin/env python3
"""
修复所有锚点链接，使其匹配检查脚本生成的锚点格式

问题：标题中的箭头符号→被移除后，留下的空格被转换为连字符，导致双连字符
解决：修复所有锚点链接，将双连字符替换为单连字符，或匹配实际生成的锚点
"""

import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def title_to_anchor_check_style(title: str) -> str:
    """按照检查脚本的方式生成锚点"""
    # 移除Markdown链接
    title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
    # 转换为小写
    anchor = title.lower()
    # 移除所有非字母数字、空格和连字符的字符（包括箭头→）
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    # 将空格和下划线替换为连字符
    anchor = re.sub(r'[\s_]+', '-', anchor)
    # 移除首尾连字符
    anchor = anchor.strip('-')
    return anchor

def extract_anchors_from_file(file_path: Path) -> dict:
    """从文件中提取所有标题和对应的锚点（按检查脚本的方式）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        anchors = {}
        # 提取所有标题
        pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(pattern, content, re.MULTILINE):
            title = match.group(2).strip()
            anchor = title_to_anchor_check_style(title)
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
        
        # 提取文件中的所有锚点（按检查脚本的方式）
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
                    # 从链接文本生成正确的锚点
                    correct_anchor = title_to_anchor_check_style(link_text)
                    
                    # 如果当前锚点不正确，修复它
                    if anchor_part != correct_anchor:
                        new_target = f"#{correct_anchor}" if not file_part else f"{file_part}#{correct_anchor}"
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
