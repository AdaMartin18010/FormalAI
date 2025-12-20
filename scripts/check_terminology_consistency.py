#!/usr/bin/env python3
"""
术语一致性检查脚本

功能：
    - 检查所有术语是否在model/07中定义
    - 检查术语使用是否一致
    - 生成检查报告

使用方法：
    python scripts/check_terminology_consistency.py [--path PATH]
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Set
import argparse

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 术语表文件（假设在Philosophy/model/07中）
TERMINOLOGY_FILE = ROOT_DIR / "Philosophy" / "model" / "07-术语表与概念索引.md"

class TerminologyChecker:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.defined_terms: Set[str] = set()
        self.undefined_terms: List[Tuple[Path, str]] = []  # (file, term)
        self.files_checked = 0
        
    def load_terminology(self):
        """从术语表文件加载已定义的术语"""
        if not TERMINOLOGY_FILE.exists():
            print(f"警告：术语表文件不存在: {TERMINOLOGY_FILE}")
            return
        
        try:
            with open(TERMINOLOGY_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取术语（假设格式为：**术语名**：定义）
            pattern = r'\*\*([^*]+)\*\*\s*[：:]\s*'
            for match in re.finditer(pattern, content):
                term = match.group(1).strip()
                self.defined_terms.add(term.lower())
        except Exception as e:
            print(f"警告：无法读取术语表文件: {e}")
    
    def _should_skip(self, file_path: Path) -> bool:
        """判断是否应该跳过该文件"""
        skip_dirs = {'node_modules', '.git', '__pycache__', '.venv', 'scripts'}
        
        # 跳过特定目录
        for part in file_path.parts:
            if part in skip_dirs:
                return True
        
        return False
    
    def check_file(self, file_path: Path):
        """检查单个文件的术语使用"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.files_checked += 1
            
            # 查找可能的术语（加粗文本）
            term_pattern = r'\*\*([^*]+)\*\*'
            for match in re.finditer(term_pattern, content):
                term = match.group(1).strip()
                # 跳过太短的词和常见词
                if len(term) < 3 or term.lower() in ['the', 'and', 'or', 'not', 'is', 'are']:
                    continue
                
                # 检查是否在术语表中定义
                if term.lower() not in self.defined_terms:
                    # 检查是否是已知的常见术语（如Ontology, DKB等）
                    known_terms = ['ontology', 'dkb', 'ari', 'hr', 'phronesis', 'mitsein']
                    if term.lower() not in known_terms:
                        self.undefined_terms.append((file_path, term))
        except Exception as e:
            print(f"警告：无法检查文件 {file_path}: {e}")
    
    def check_all(self, path: Path = None):
        """检查所有文件"""
        if path is None:
            path = self.root_dir
        
        for md_file in path.rglob("*.md"):
            if not self._should_skip(md_file):
                self.check_file(md_file)
    
    def generate_report(self) -> str:
        """生成检查报告"""
        report = []
        report.append("# 术语一致性检查报告\n")
        report.append(f"**检查文件数**：{self.files_checked}\n")
        report.append(f"**已定义术语数**：{len(self.defined_terms)}\n")
        report.append(f"**发现未定义术语数**：{len(self.undefined_terms)}\n\n")
        
        if not self.undefined_terms:
            report.append("✅ **所有术语使用一致！**\n")
        else:
            report.append("## 未定义术语列表\n\n")
            report.append("| 文件 | 术语 |\n")
            report.append("|------|------|\n")
            
            # 按术语分组
            by_term: Dict[str, List[Path]] = {}
            for file_path, term in self.undefined_terms:
                if term not in by_term:
                    by_term[term] = []
                by_term[term].append(file_path)
            
            for term, files in sorted(by_term.items()):
                files_str = ", ".join([f"`{f.relative_to(self.root_dir)}`" for f in files[:5]])
                if len(files) > 5:
                    files_str += f" ... (共{len(files)}个文件)"
                report.append(f"| `{term}` | {files_str} |\n")
        
        return "".join(report)

def main():
    parser = argparse.ArgumentParser(description='检查Markdown文档的术语一致性')
    parser.add_argument('--path', type=str, default=None, help='要检查的路径（默认为项目根目录）')
    parser.add_argument('--output', type=str, default='terminology_consistency_report.md', help='输出报告文件')
    
    args = parser.parse_args()
    
    root_dir = Path(args.path) if args.path else ROOT_DIR
    checker = TerminologyChecker(root_dir)
    
    print("加载术语表...")
    checker.load_terminology()
    print(f"加载了 {len(checker.defined_terms)} 个已定义术语")
    
    print("检查术语使用...")
    checker.check_all(root_dir)
    
    print(f"检查了 {checker.files_checked} 个文件")
    print(f"发现 {len(checker.undefined_terms)} 个未定义术语使用")
    
    # 生成报告
    report = checker.generate_report()
    output_file = root_dir / args.output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {output_file}")
    
    if checker.undefined_terms:
        return 1
    else:
        print("\n✅ 所有术语使用一致！")
        return 0

if __name__ == "__main__":
    exit(main())
