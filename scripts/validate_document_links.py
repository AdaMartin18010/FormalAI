#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档链接验证工具
验证文档中的内部链接是否有效
创建日期：2025-01-XX
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# 需要检查的文档列表
DOCS_TO_CHECK = [
    "docs/LATEST_AI_DEVELOPMENTS_2025.md",
    "docs/CASE_STUDIES_UPDATE_2025.md",
    "docs/DOCUMENT_INDEX_2025.md",
    "docs/GLOBAL_NAVIGATION.md",
    "README.md",
]

def extract_links(content, base_path):
    """提取文档中的所有链接"""
    links = []
    
    # Markdown链接格式: [text](path)
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    matches = re.findall(pattern, content)
    
    for text, link in matches:
        # 跳过外部链接
        if link.startswith('http://') or link.startswith('https://') or link.startswith('mailto:'):
            continue
        
        # 处理相对路径
        if link.startswith('./'):
            link = link[2:]
        elif link.startswith('../'):
            # 计算相对路径
            link = os.path.normpath(os.path.join(os.path.dirname(base_path), link))
        
        links.append({
            'text': text,
            'link': link,
            'base_path': base_path
        })
    
    return links

def check_link_exists(link_path, base_path):
    """检查链接是否存在"""
    # 如果是锚点链接（包含#），只检查文件部分
    if '#' in link_path:
        file_path, anchor = link_path.split('#', 1)
    else:
        file_path = link_path
        anchor = None
    
    # 处理相对路径
    if not os.path.isabs(file_path):
        file_path = os.path.normpath(os.path.join(os.path.dirname(base_path), file_path))
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False, f"文件不存在: {file_path}"
    
    # 如果有锚点，检查锚点是否存在（简化检查）
    if anchor:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单的锚点检查（检查标题）
                anchor_pattern = anchor.lower().replace('-', ' ').replace('_', ' ')
                if anchor_pattern not in content.lower():
                    return True, f"文件存在但锚点可能不存在: {anchor}"
        except:
            pass
    
    return True, "OK"

def validate_document(file_path):
    """验证单个文档的链接"""
    if not os.path.exists(file_path):
        return {"status": "missing", "file": file_path, "links": []}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        links = extract_links(content, file_path)
        results = []
        
        for link_info in links:
            exists, message = check_link_exists(link_info['link'], link_info['base_path'])
            results.append({
                'text': link_info['text'],
                'link': link_info['link'],
                'exists': exists,
                'message': message
            })
        
        return {
            "status": "ok",
            "file": file_path,
            "links": results
        }
    
    except Exception as e:
        return {"status": "error", "file": file_path, "error": str(e)}

def main():
    """主函数"""
    print("开始验证文档链接...\n")
    
    all_results = []
    broken_links = []
    
    for file_path in DOCS_TO_CHECK:
        result = validate_document(file_path)
        all_results.append(result)
        
        if result["status"] == "ok":
            print(f"检查: {file_path}")
            broken_in_file = [l for l in result["links"] if not l["exists"]]
            if broken_in_file:
                print(f"  ✗ 发现 {len(broken_in_file)} 个无效链接:")
                for link in broken_in_file:
                    print(f"    - [{link['text']}]({link['link']}) - {link['message']}")
                    broken_links.append({
                        'file': file_path,
                        'link': link
                    })
            else:
                print(f"  ✓ 所有链接有效 ({len(result['links'])} 个链接)")
        elif result["status"] == "missing":
            print(f"  ⊘ 跳过: {file_path} (文件不存在)")
        elif result["status"] == "error":
            print(f"  ✗ 错误: {file_path} - {result.get('error', 'Unknown')}")
    
    print(f"\n验证完成统计:")
    total_links = sum(len(r.get("links", [])) for r in all_results if r["status"] == "ok")
    print(f"  总链接数: {total_links}")
    print(f"  无效链接: {len(broken_links)}")
    print(f"  有效链接: {total_links - len(broken_links)}")
    
    if broken_links:
        print(f"\n无效链接列表:")
        for item in broken_links:
            print(f"  - {item['file']}: [{item['link']['text']}]({item['link']['link']})")

if __name__ == "__main__":
    main()
