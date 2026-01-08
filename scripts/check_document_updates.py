#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档更新检查工具
检查docs模块和concepts模块的文档是否包含2025年最新发展章节
创建日期：2025-01-XX
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# 需要检查的文档列表
DOCS_TO_CHECK = {
    "concepts": [
        "concepts/01-AI三层模型架构/README.md",
        "concepts/02-AI炼金术转化度模型/README.md",
        "concepts/03-Scaling Law与收敛分析/README.md",
        "concepts/04-AI意识与认知模拟/README.md",
        "concepts/05-AI科学理论/README.md",
        "concepts/06-AI反实践判定系统/README.md",
        "concepts/07-AI框架批判与重构/README.md",
        "concepts/08-AI历史进程与原理演进/README.md",
    ],
    "docs": [
        "docs/04-language-models/04.1-大型语言模型/README.md",
        "docs/04-language-models/04.5-AI代理/README.md",
        "docs/05-multimodal-ai/05.1-视觉语言模型/README.md",
        "docs/07-alignment-safety/07.1-对齐理论/README.md",
        "docs/08-emergence-complexity/08.1-涌现理论/README.md",
        "docs/02-machine-learning/02.2-深度学习理论/README.md",
        "docs/09-philosophy-ethics/09.2-意识理论/README.md",
    ]
}

def check_document(file_path):
    """检查单个文档是否包含2025年最新发展章节"""
    if not os.path.exists(file_path):
        return {"status": "missing", "file": file_path}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含2025年最新发展章节
        patterns = [
            r'2025年最新发展',
            r'Latest Developments 2025',
            r'LATEST_AI_DEVELOPMENTS_2025\.md',
        ]
        
        has_2025_section = any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)
        
        return {
            "status": "ok" if has_2025_section else "missing",
            "file": file_path,
            "has_2025_section": has_2025_section
        }
    
    except Exception as e:
        return {"status": "error", "file": file_path, "error": str(e)}

def main():
    """主函数"""
    print("开始检查文档更新状态...\n")
    
    results = defaultdict(list)
    
    for module_type, files in DOCS_TO_CHECK.items():
        print(f"检查 {module_type} 模块:")
        for file_path in files:
            result = check_document(file_path)
            results[result["status"]].append(result)
            
            if result["status"] == "ok":
                print(f"  ✓ {file_path}")
            elif result["status"] == "missing":
                print(f"  ✗ {file_path} (缺少2025年最新发展章节)")
            elif result["status"] == "error":
                print(f"  ⊘ {file_path} (检查错误: {result.get('error', 'Unknown')})")
    
    print(f"\n检查完成统计:")
    print(f"  已更新: {len(results['ok'])}")
    print(f"  待更新: {len(results['missing'])}")
    print(f"  错误: {len(results['error'])}")
    
    if results['missing']:
        print(f"\n待更新文档列表:")
        for result in results['missing']:
            print(f"  - {result['file']}")
    
    if results['error']:
        print(f"\n检查错误文档列表:")
        for result in results['error']:
            print(f"  - {result['file']}: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()
