#!/usr/bin/env python3
"""
参考文献格式检查脚本
检查所有文档中的引用格式是否符合标准
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Dict

# 引用格式模式
CITATION_PATTERNS = {
    'arxiv': r'arXiv:\d{4}\.\d{5}',
    'pubmed': r'PubMed:\s*\d+',
    'doi': r'DOI:\s*10\.\d+/[^\s]+',
    'url': r'https?://[^\s]+',
}

# 标准格式模式
STANDARD_FORMAT = r'^[A-Z][^.]*\.\s*\((\d{4})\)\.\s*[^.]+\.[^.]*\.'

def find_markdown_files(root_dir: str) -> List[Path]:
    """查找所有Markdown文件"""
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过隐藏目录和特定目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']
        
        for file in files:
            if file.endswith('.md'):
                md_files.append(Path(root) / file)
    return md_files

def extract_citations(content: str) -> List[Tuple[int, str]]:
    """提取文档中的引用"""
    citations = []
    lines = content.split('\n')
    
    # 查找参考文献部分
    in_references = False
    for i, line in enumerate(lines):
        if re.search(r'##+\s*参考|##+\s*Reference|##+\s*参考文献', line, re.IGNORECASE):
            in_references = True
            continue
        
        if in_references:
            # 检查是否是引用行（以数字开头或包含作者）
            if re.match(r'^\d+\.', line.strip()) or re.match(r'^[A-Z][^.]*\.\s*\(', line.strip()):
                citations.append((i + 1, line.strip()))
    
    return citations

def check_citation_format(citation: str) -> Dict[str, any]:
    """检查单个引用的格式"""
    issues = []
    
    # 检查基本格式
    if not re.match(STANDARD_FORMAT, citation):
        issues.append("格式不符合标准")
    
    # 检查年份
    year_match = re.search(r'\((\d{4})\)', citation)
    if not year_match:
        issues.append("缺少年份")
    else:
        year = int(year_match.group(1))
        if year < 2020 or year > 2026:
            issues.append(f"年份异常: {year}")
    
    # 检查来源
    has_source = False
    for pattern_name, pattern in CITATION_PATTERNS.items():
        if re.search(pattern, citation, re.IGNORECASE):
            has_source = True
            break
    
    if not has_source and not re.search(r'Wikipedia|\.com|\.org', citation):
        issues.append("缺少明确的来源标识")
    
    return {
        'citation': citation,
        'issues': issues,
        'valid': len(issues) == 0
    }

def check_file_citations(file_path: Path) -> Dict[str, any]:
    """检查文件中的引用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'citations': []
        }
    
    citations = extract_citations(content)
    checked_citations = []
    
    for line_num, citation in citations:
        result = check_citation_format(citation)
        result['line'] = line_num
        checked_citations.append(result)
    
    return {
        'file': str(file_path),
        'citations': checked_citations,
        'total': len(citations),
        'valid': sum(1 for c in checked_citations if c['valid']),
        'invalid': sum(1 for c in checked_citations if not c['valid'])
    }

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    md_files = find_markdown_files(str(project_root))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    print("=" * 80)
    
    total_citations = 0
    total_valid = 0
    total_invalid = 0
    files_with_issues = []
    
    for md_file in md_files:
        result = check_file_citations(md_file)
        
        if result.get('error'):
            print(f"❌ {result['file']}: {result['error']}")
            continue
        
        if result['total'] > 0:
            total_citations += result['total']
            total_valid += result['valid']
            total_invalid += result['invalid']
            
            if result['invalid'] > 0:
                files_with_issues.append(result)
                print(f"\n⚠️  {result['file']}")
                print(f"   总引用数: {result['total']}, 有效: {result['valid']}, 无效: {result['invalid']}")
                
                for citation in result['citations']:
                    if not citation['valid']:
                        print(f"   第 {citation['line']} 行:")
                        print(f"     {citation['citation'][:100]}...")
                        for issue in citation['issues']:
                            print(f"     - {issue}")
    
    print("\n" + "=" * 80)
    print("引用格式检查总结:")
    print(f"  总文件数: {len(md_files)}")
    print(f"  总引用数: {total_citations}")
    print(f"  有效引用: {total_valid} ({total_valid/total_citations*100:.1f}%)" if total_citations > 0 else "  有效引用: 0")
    print(f"  无效引用: {total_invalid} ({total_invalid/total_citations*100:.1f}%)" if total_citations > 0 else "  无效引用: 0")
    print(f"  有问题的文件: {len(files_with_issues)}")
    
    if total_invalid == 0 and total_citations > 0:
        print("\n✅ 所有引用格式正确！")
    elif total_citations > 0:
        print(f"\n⚠️  发现 {total_invalid} 个格式问题，请参考 CITATION_STYLE_GUIDE.md 进行修正")

if __name__ == '__main__':
    main()
