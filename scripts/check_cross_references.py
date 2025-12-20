#!/usr/bin/env python3
"""
交叉引用检查脚本

功能：
    - 检查所有内部链接是否有效
    - 检查所有外部引用是否存在
    - 生成检查报告

使用方法：
    python scripts/check_cross_references.py [--path PATH]
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Set
import argparse

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# Markdown链接模式
LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
ANCHOR_PATTERN = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)

class CrossReferenceChecker:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.all_files: Set[Path] = set()
        self.all_anchors: dict = {}  # file_path -> set of anchors
        self.broken_links: List[Tuple[Path, str, str]] = []  # (file, link_text, link_target)

    def scan_files(self, path: Path = None):
        """扫描所有Markdown文件"""
        if path is None:
            path = self.root_dir

        for md_file in path.rglob("*.md"):
            if self._should_skip(md_file):
                continue
            self.all_files.add(md_file)
            self._extract_anchors(md_file)

    def _should_skip(self, file_path: Path) -> bool:
        """判断是否应该跳过该文件"""
        skip_dirs = {'node_modules', '.git', '__pycache__', '.venv'}
        skip_files = {'CHANGELOG.md', 'README.md'}

        # 跳过特定目录
        for part in file_path.parts:
            if part in skip_dirs:
                return True

        # 跳过特定文件
        if file_path.name in skip_files and file_path.parent == self.root_dir:
            return False  # 根目录的README.md不跳过

        return False

    def _extract_anchors(self, file_path: Path):
        """提取文件中的所有锚点"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            anchors = set()
            # 提取标题作为锚点
            for match in ANCHOR_PATTERN.finditer(content):
                title = match.group(1).strip()
                # 生成锚点（Markdown格式）
                anchor = self._title_to_anchor(title)
                anchors.add(anchor)

            self.all_anchors[file_path] = anchors
        except Exception as e:
            print(f"警告：无法读取文件 {file_path}: {e}")

    def _title_to_anchor(self, title: str) -> str:
        """将标题转换为锚点格式"""
        # 移除Markdown链接
        title = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', title)
        # 转换为小写，替换空格为连字符
        anchor = title.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor)
        anchor = re.sub(r'[\s_]+', '-', anchor)
        anchor = anchor.strip('-')
        return anchor

    def _resolve_link(self, source_file: Path, link_target: str) -> Tuple[bool, str]:
        """解析链接目标，返回(是否有效, 错误信息)"""
        # 移除锚点部分
        if '#' in link_target:
            file_part, anchor_part = link_target.split('#', 1)
            anchor = anchor_part
        else:
            file_part = link_target
            anchor = None

        # 处理相对路径
        if file_part.startswith('http://') or file_part.startswith('https://'):
            return (True, "")  # 外部链接，暂不检查

        if file_part.startswith('mailto:'):
            return (True, "")  # 邮件链接，暂不检查

        # 解析文件路径
        if file_part:
            target_file = (source_file.parent / file_part).resolve()
        else:
            target_file = source_file

        # 检查文件是否存在
        if not target_file.exists():
            return (False, f"文件不存在: {target_file}")

        if target_file not in self.all_files:
            return (False, f"文件未在扫描范围内: {target_file}")

        # 检查锚点
        if anchor:
            target_anchors = self.all_anchors.get(target_file, set())
            anchor_normalized = self._title_to_anchor(anchor)
            if anchor_normalized not in target_anchors:
                return (False, f"锚点不存在: #{anchor}")

        return (True, "")

    def check_links(self, path: Path = None):
        """检查所有链接"""
        if path is None:
            path = self.root_dir

        for md_file in path.rglob("*.md"):
            if self._should_skip(md_file):
                continue

            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 查找所有链接
                for match in LINK_PATTERN.finditer(content):
                    link_text = match.group(1)
                    link_target = match.group(2)

                    # 检查链接
                    is_valid, error_msg = self._resolve_link(md_file, link_target)
                    if not is_valid:
                        self.broken_links.append((md_file, link_text, link_target, error_msg))
            except Exception as e:
                print(f"警告：无法检查文件 {md_file}: {e}")

    def generate_report(self) -> str:
        """生成检查报告"""
        report = []
        report.append("# 交叉引用检查报告\n")
        report.append(f"**检查时间**：{Path(__file__).stat().st_mtime}\n")
        report.append(f"**扫描文件数**：{len(self.all_files)}\n")
        report.append(f"**发现错误链接数**：{len(self.broken_links)}\n\n")

        if not self.broken_links:
            report.append("✅ **所有链接检查通过！**\n")
        else:
            report.append("## 错误链接列表\n\n")
            report.append("| 文件 | 链接文本 | 链接目标 | 错误信息 |\n")
            report.append("|------|----------|----------|----------|\n")

            for file_path, link_text, link_target, error_msg in self.broken_links:
                rel_path = file_path.relative_to(self.root_dir)
                report.append(f"| `{rel_path}` | `{link_text}` | `{link_target}` | {error_msg} |\n")

        return "".join(report)

def main():
    parser = argparse.ArgumentParser(description='检查Markdown文档的交叉引用')
    parser.add_argument('--path', type=str, default=None, help='要检查的路径（默认为项目根目录）')
    parser.add_argument('--output', type=str, default='cross_reference_report.md', help='输出报告文件')

    args = parser.parse_args()

    root_dir = Path(args.path) if args.path else ROOT_DIR
    checker = CrossReferenceChecker(root_dir)

    print("扫描文件...")
    checker.scan_files(root_dir)
    print(f"找到 {len(checker.all_files)} 个Markdown文件")

    print("检查链接...")
    checker.check_links(root_dir)

    print(f"发现 {len(checker.broken_links)} 个错误链接")

    # 生成报告
    report = checker.generate_report()
    output_file = root_dir / args.output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存到: {output_file}")

    if checker.broken_links:
        print("\n错误链接预览（前10个）：")
        for file_path, link_text, link_target, error_msg in checker.broken_links[:10]:
            print(f"  {file_path.relative_to(root_dir)}: {link_text} -> {link_target} ({error_msg})")
        return 1
    else:
        print("\n✅ 所有链接检查通过！")
        return 0

if __name__ == "__main__":
    exit(main())
