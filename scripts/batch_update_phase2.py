#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新docs模块文档（第二阶段），添加2025年最新发展章节
创建日期：2025-01-XX
"""

import os
import re
from pathlib import Path

# 需要更新的文档列表（第二阶段）
DOCS_TO_UPDATE = [
    # AGI理论模块
    "docs/16-agi-theory/16.2-意识与自我/README.md",
    "docs/16-agi-theory/16.3-创造性AI/README.md",
    "docs/16-agi-theory/16.4-AGI安全与对齐/README.md",
    
    # 社会AI模块
    "docs/17-social-ai/17.1-多智能体系统/README.md",
    "docs/17-social-ai/17.2-社会认知/README.md",
    "docs/17-social-ai/17.3-集体智能/README.md",
    
    # 认知架构模块
    "docs/18-cognitive-architecture/18.1-认知模型/README.md",
    "docs/18-cognitive-architecture/18.2-记忆系统/README.md",
    "docs/18-cognitive-architecture/18.3-注意力机制/README.md",
    "docs/18-cognitive-architecture/18.4-决策系统/README.md",
    
    # 神经符号AI模块
    "docs/19-neuro-symbolic-advanced/19.2-逻辑神经网络/README.md",
    "docs/19-neuro-symbolic-advanced/19.3-符号学习/README.md",
    "docs/19-neuro-symbolic-advanced/19.4-混合推理/README.md",
    
    # AI哲学高级模块
    "docs/20-ai-philosophy-advanced/20.2-自由意志/README.md",
    "docs/20-ai-philosophy-advanced/20.3-机器意识/README.md",
    "docs/20-ai-philosophy-advanced/20.4-AI存在论/README.md",
    
    # 其他模块
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
    patterns = [
        r'##\s*进一步阅读',
        r'##\s*参考文献',
        r'##\s*References',
        r'##\s*参考文档',
        r'\*\*最后更新\*\*',
        r'---\s*\n\s*\*\*最后更新\*\*',
        r'_本模块为FormalAI',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.start()
    
    return len(content)

def update_document(file_path):
    """更新单个文档"""
    try:
        if not os.path.exists(file_path):
            print(f"⊘ 跳过: {file_path} (文件不存在)")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '2025年最新发展' in content or 'Latest Developments 2025' in content:
            print(f"⊘ 跳过: {file_path} (已包含2025年最新发展章节)")
            return False
        
        insertion_point = find_insertion_point(content)
        new_content = content[:insertion_point] + LATEST_DEVELOPMENTS_SECTION + content[insertion_point:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✓ 已更新: {file_path}")
        return True
    
    except Exception as e:
        print(f"✗ 错误: {file_path} - {e}")
        return False

def main():
    """主函数"""
    print("开始批量更新docs模块文档（第二阶段），添加2025年最新发展章节...\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for doc_path in DOCS_TO_UPDATE:
        if update_document(doc_path):
            success_count += 1
        else:
            skip_count += 1
    
    print(f"\n更新完成统计:")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")
    print(f"  总计: {len(DOCS_TO_UPDATE)}")

if __name__ == "__main__":
    main()
