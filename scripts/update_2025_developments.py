#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新文档，添加2025年最新发展章节引用
创建日期：2025-01-XX
"""

import os
import re
from pathlib import Path

# 需要更新的文档列表
DOCS_TO_UPDATE = [
    # Docs模块文档
    "docs/02-machine-learning/02.2-深度学习理论/README.md",
    "docs/09-philosophy-ethics/09.2-意识理论/README.md",
    "docs/16-agi-theory/16.1-通用智能理论/README.md",
    "docs/16-agi-theory/16.2-意识与自我/README.md",
    "docs/19-neuro-symbolic-advanced/19.1-知识图谱推理/README.md",
    "docs/10-embodied-ai/10.1-具身智能/README.md",
    "docs/12-quantum-ai/12.1-量子机器学习/README.md",
]

# 2025年最新发展章节模板
LATEST_DEVELOPMENTS_SECTION = """

---

## 2025年最新发展 / Latest Developments 2025

### 最新技术发展

**2025年最新研究**：
- 参见 [2024-2025年最新AI技术发展总结](../../LATEST_AI_DEVELOPMENTS_2025.md)

**详细内容**：本文档的最新发展内容已整合到 [2024-2025年最新AI技术发展总结](../../LATEST_AI_DEVELOPMENTS_2025.md) 中，请参考该文档获取最新信息。

---

**最后更新**：2025-01-XX
"""

def find_insertion_point(content):
    """找到插入2025年最新发展章节的位置"""
    # 查找"进一步阅读"或"参考文献"章节
    patterns = [
        r'##\s*进一步阅读',
        r'##\s*参考文献',
        r'##\s*References',
        r'##\s*参考文档',
        r'\*\*最后更新\*\*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.start()
    
    # 如果没找到，在文档末尾插入
    return len(content)

def update_document(file_path):
    """更新单个文档"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经包含2025年最新发展章节
        if '2025年最新发展' in content or 'Latest Developments 2025' in content:
            print(f"⊘ 跳过: {file_path} (已包含2025年最新发展章节)")
            return False
        
        # 找到插入位置
        insertion_point = find_insertion_point(content)
        
        # 插入2025年最新发展章节
        new_content = content[:insertion_point] + LATEST_DEVELOPMENTS_SECTION + content[insertion_point:]
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✓ 已更新: {file_path}")
        return True
    
    except Exception as e:
        print(f"✗ 错误: {file_path} - {e}")
        return False

def main():
    """主函数"""
    print("开始批量更新文档，添加2025年最新发展章节...")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for doc_path in DOCS_TO_UPDATE:
        if os.path.exists(doc_path):
            if update_document(doc_path):
                success_count += 1
            else:
                skip_count += 1
        else:
            print(f"⊘ 跳过: {doc_path} (文件不存在)")
            skip_count += 1
    
    print(f"\n更新完成统计:")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")

if __name__ == "__main__":
    main()
