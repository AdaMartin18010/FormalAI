#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新docs模块文档，添加2025年最新发展章节
创建日期：2025-01-XX
"""

import os
import re
from pathlib import Path

# 需要更新的文档列表（扩大范围）
DOCS_TO_UPDATE = [
    # 语言模型模块
    "docs/04-language-models/04.2-形式语义/README.md",
    "docs/04-language-models/04.3-知识表示/README.md",
    "docs/04-language-models/04.4-推理机制/README.md",
    
    # 多模态AI模块
    "docs/05-multimodal-ai/05.2-多模态融合/README.md",
    "docs/05-multimodal-ai/05.3-跨模态推理/README.md",
    
    # 可解释AI模块
    "docs/06-interpretable-ai/06.1-可解释性理论/README.md",
    "docs/06-interpretable-ai/06.2-公平性与偏见/README.md",
    "docs/06-interpretable-ai/06.3-鲁棒性理论/README.md",
    
    # 对齐与安全模块
    "docs/07-alignment-safety/07.2-价值学习/README.md",
    "docs/07-alignment-safety/07.3-安全机制/README.md",
    
    # 涌现与复杂性模块
    "docs/08-emergence-complexity/08.2-复杂系统/README.md",
    "docs/08-emergence-complexity/08.3-自组织/README.md",
    
    # 哲学与伦理模块
    "docs/09-philosophy-ethics/09.1-AI哲学/README.md",
    "docs/09-philosophy-ethics/09.3-伦理框架/README.md",
    
    # 形式化方法模块
    "docs/03-formal-methods/03.1-形式化验证/README.md",
    "docs/03-formal-methods/03.2-程序综合/README.md",
    "docs/03-formal-methods/03.3-类型理论/README.md",
    "docs/03-formal-methods/03.4-证明系统/README.md",
    
    # 机器学习模块
    "docs/02-machine-learning/02.1-统计学习理论/README.md",
    "docs/02-machine-learning/02.3-强化学习理论/README.md",
    "docs/02-machine-learning/02.4-因果推理/README.md",
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
        r'---\s*\n\s*\*\*最后更新\*\*',
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
        if not os.path.exists(file_path):
            print(f"⊘ 跳过: {file_path} (文件不存在)")
            return False
        
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
    print("开始批量更新docs模块文档，添加2025年最新发展章节...\n")
    
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
