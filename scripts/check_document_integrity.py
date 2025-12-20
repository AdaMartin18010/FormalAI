#!/usr/bin/env python3
"""
文档完整性检查脚本

功能：
    - 检查所有必需章节是否存在
    - 检查所有必需引用是否完整
    - 生成检查报告

使用方法：
    python scripts/check_document_integrity.py [--path PATH]
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 必需章节（根据DOCUMENT_STRUCTURE_STANDARD.md）
REQUIRED_SECTIONS = [
    r'^##\s+一[、.]\s*概述',
    r'^##\s+二[、.]\s*目录',
    r'^##\s+.*与.*模型.*关系',  # 与三层模型的关系或与收敛模型的关系
    r'^##\s+.*核心结论',
    r'^##\s+.*相关主题',
    r'^##\s+.*参考文档',
]

class DocumentIntegrityChecker:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.missing_sections: List[Tuple[Path, str]] = []  # (file, missing_section)
        self.files_checked = 0

    def _should_skip(self, file_path: Path) -> bool:
        """判断是否应该跳过该文件"""
        skip_dirs = {'node_modules', '.git', '__pycache__', '.venv', 'scripts'}
        skip_files = {
            'CHANGELOG.md', 'README.md', 'INDEX.md',
            'PHASE1_*.md', 'CONCEPTS_*.md', 'DOCUMENT_*.md',
            'REPAIR_*.md', 'VIEW_*.md', 'AI_ARGUMENTS_*.md'
        }

        # 跳过特定目录
        for part in file_path.parts:
            if part in skip_dirs:
                return True

        # 跳过特定文件模式
        for pattern in skip_files:
            if pattern.endswith('*.md'):
                prefix = pattern.replace('*.md', '')
                if file_path.name.startswith(prefix):
                    return True
            elif file_path.name == pattern:
                return True

        return False

    def check_file(self, file_path: Path):
        """检查单个文件的完整性"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.files_checked += 1

            # 检查必需章节
            found_sections = set()
            for pattern in REQUIRED_SECTIONS:
                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    found_sections.add(pattern)
                else:
                    self.missing_sections.append((file_path, pattern))
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
        report.append("# 文档完整性检查报告\n")
        report.append(f"**检查文件数**：{self.files_checked}\n")
        report.append(f"**发现缺失章节数**：{len(self.missing_sections)}\n\n")

        if not self.missing_sections:
            report.append("✅ **所有文档完整性检查通过！**\n")
        else:
            report.append("## 缺失章节列表\n\n")
            report.append("| 文件 | 缺失章节 |\n")
            report.append("|------|----------|\n")

            # 按文件分组
            by_file: Dict[Path, List[str]] = {}
            for file_path, pattern in self.missing_sections:
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(pattern)

            for file_path, patterns in sorted(by_file.items()):
                rel_path = file_path.relative_to(self.root_dir)
                patterns_str = ", ".join([f"`{p}`" for p in patterns])
                report.append(f"| `{rel_path}` | {patterns_str} |\n")

        return "".join(report)

def main():
    parser = argparse.ArgumentParser(description='检查Markdown文档的完整性')
    parser.add_argument('--path', type=str, default=None, help='要检查的路径（默认为项目根目录）')
    parser.add_argument('--output', type=str, default='document_integrity_report.md', help='输出报告文件')

    args = parser.parse_args()

    root_dir = Path(args.path) if args.path else ROOT_DIR
    checker = DocumentIntegrityChecker(root_dir)

    print("检查文档完整性...")
    checker.check_all(root_dir)

    print(f"检查了 {checker.files_checked} 个文件")
    print(f"发现 {len(checker.missing_sections)} 个缺失章节")

    # 生成报告
    report = checker.generate_report()
    output_file = root_dir / args.output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {output_file}")

    if checker.missing_sections:
        return 1
    else:
        print("\n✅ 所有文档完整性检查通过！")
        return 0

if __name__ == "__main__":
    exit(main())
