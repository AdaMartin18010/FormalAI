#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
from typing import List, Tuple

FOUR_LANG_TITLE_RE = re.compile(r"^#\s+\d+\.\d+\s+[^/]+\s/\s[^/]+\s/\s[^/]+\s/\s[^/]+\s*$")
SECOND_LEVEL_NUMBERED_RE = re.compile(r"^##\s+\d+\.[^\n]*$")
CHAPTER_DIR_RE = re.compile(r"^(\d{2})-[^/\\]+$")


def is_chapter_markdown(path: str, root: str) -> bool:
	# 仅对 docs 下的分册目录生效，如 01-foundations/.../README.md
	rel = os.path.relpath(path, root)
	parts = rel.replace("\\", "/").split("/")
	if len(parts) < 2:
		return False
	# 第一层必须是 NN-xxxx
	return bool(CHAPTER_DIR_RE.match(parts[0]))


def scan_file(path: str, root: str) -> Tuple[List[str], List[str]]:
	warnings: List[str] = []
	errors: List[str] = []
	try:
		with open(path, "r", encoding="utf-8") as f:
			lines = f.read().splitlines()
	except Exception as e:
		errors.append(f"IO错误: {path}: {e}")
		return warnings, errors

	enforce_h1 = is_chapter_markdown(path, root)

	# Rule 1: first H1 should be four-language with numbering like `# 3.2 标题 / Title / Titel / Titre`
	if enforce_h1:
		for i, line in enumerate(lines[:5]):
			if line.startswith("# "):
				if not FOUR_LANG_TITLE_RE.match(line):
					errors.append(f"四语言标题不合规: {path}: 第{i+1}行: '{line}'")
				break
		else:
			errors.append(f"缺少一级标题: {path}")

	# Rule 2: check that second-level numbered sections (## 1., ## 2., ...) if present are consecutively increasing
	section_nums: List[int] = []
	for line in lines:
		m = SECOND_LEVEL_NUMBERED_RE.match(line)
		if m:
			# extract the integer before the dot
			head = m.group(0).split()[1]
			try:
				n = int(head.split(".")[0])
				section_nums.append(n)
			except Exception:
				warnings.append(f"编号解析失败: {path}: '{line}'")
	if section_nums:
		for idx in range(1, len(section_nums)):
			if section_nums[idx] != section_nums[idx-1] + 1:
				warnings.append(f"二级编号可能不连续: {path}: {section_nums[idx-1]} -> {section_nums[idx]}")

	return warnings, errors


def main() -> int:
	if len(sys.argv) < 2:
		print("用法: python tools/qa/formalai_qa.py docs/ [--report REPORT.md]")
		return 2
	target = sys.argv[1]
	report_path = None
	if len(sys.argv) >= 4 and sys.argv[2] == "--report":
		report_path = sys.argv[3]

	all_warnings: List[str] = []
	all_errors: List[str] = []

	for root, _, files in os.walk(target):
		for name in files:
			if not name.lower().endswith(".md"):
				continue
			path = os.path.join(root, name)
			w, e = scan_file(path, target)
			all_warnings.extend(w)
			all_errors.extend(e)

	status = 0 if not all_errors else 2

	lines: List[str] = []
	lines.append("# FormalAI QA Report")
	lines.append("")
	lines.append(f"目标目录: {target}")
	lines.append(f"错误数: {len(all_errors)}  警告数: {len(all_warnings)}")
	lines.append("")
	if all_errors:
		lines.append("## 错误 / Errors")
		lines.extend([f"- {e}" for e in all_errors])
		lines.append("")
	if all_warnings:
		lines.append("## 警告 / Warnings")
		lines.extend([f"- {w}" for w in all_warnings])

	output = "\n".join(lines)
	if report_path:
		try:
			with open(report_path, "w", encoding="utf-8") as f:
				f.write(output)
			print(f"报告已写入: {report_path}")
		except Exception as e:
			print(f"写报告失败: {e}")
	else:
		print(output)

	return status


if __name__ == "__main__":
	sys.exit(main()) 