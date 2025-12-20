#!/usr/bin/env python3
"""
变更影响分析脚本

功能：
    - 分析文档变更的影响范围
    - 识别受影响的文档
    - 评估变更风险
    - 生成影响分析报告

使用方法：
    python scripts/analyze_change_impact.py [--file FILE] [--path PATH] [--report]
"""

import os
import re
from pathlib import Path
import argparse
from typing import List, Dict, Set, Tuple
import subprocess

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 变更类型定义
CHANGE_TYPES = {
    'content_update': '内容更新',
    'structure_change': '结构变更',
    'concept_change': '概念变更',
    'axiom_theorem_change': '公理定理变更',
    'cross_module_integration': '跨模块整合',
    'version_upgrade': '版本升级'
}

# 风险等级定义
RISK_LEVELS = {
    'low': '低',
    'medium': '中',
    'high': '高',
    'critical': '极高'
}

def extract_cross_references(file_path: Path) -> List[str]:
    """从文件中提取交叉引用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取Markdown链接
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        links = re.findall(link_pattern, content)

        # 提取文件路径
        file_refs = []
        for link_text, link_target in links:
            # 移除锚点
            if '#' in link_target:
                file_part = link_target.split('#')[0]
            else:
                file_part = link_target

            if file_part and not file_part.startswith('http'):
                file_refs.append(file_part)

        return file_refs
    except Exception as e:
        print(f"警告：无法读取文件 {file_path}: {e}")
        return []

def find_referencing_files(target_file: Path, search_path: Path) -> List[Path]:
    """查找引用目标文件的所有文件"""
    referencing_files = []
    target_rel_path = target_file.relative_to(search_path)

    for md_file in search_path.rglob("*.md"):
        if md_file == target_file:
            continue

        refs = extract_cross_references(md_file)
        for ref in refs:
            # 解析相对路径
            ref_path = (md_file.parent / ref).resolve()
            if ref_path == target_file.resolve():
                referencing_files.append(md_file)
                break

    return referencing_files

def identify_change_type(file_path: Path, git_diff: str) -> str:
    """识别变更类型"""
    # 检查是否涉及公理/定理
    if re.search(r'(公理|定理|Axiom|Theorem)', git_diff, re.IGNORECASE):
        return 'axiom_theorem_change'

    # 检查是否涉及结构变更（章节、目录）
    if re.search(r'(^#+\s|目录|Table of Contents)', git_diff, re.MULTILINE):
        return 'structure_change'

    # 检查是否涉及概念定义
    if re.search(r'(定义|Definition|概念|Concept)', git_diff, re.IGNORECASE):
        return 'concept_change'

    # 检查是否涉及跨模块引用
    if re.search(r'(\.\./docs/|\.\./concepts/|\.\./view/)', git_diff):
        return 'cross_module_integration'

    # 检查是否涉及版本号变更
    if re.search(r'(version|版本|v\d+\.\d+\.\d+)', git_diff, re.IGNORECASE):
        return 'version_upgrade'

    # 默认为内容更新
    return 'content_update'

def assess_risk_level(change_type: str, affected_files: int) -> str:
    """评估风险等级"""
    risk_map = {
        'content_update': 'low',
        'structure_change': 'medium',
        'concept_change': 'high' if affected_files > 10 else 'medium',
        'axiom_theorem_change': 'critical',
        'cross_module_integration': 'medium',
        'version_upgrade': 'medium'
    }

    base_risk = risk_map.get(change_type, 'low')

    # 根据影响文件数量调整风险
    if affected_files > 20:
        if base_risk == 'low':
            return 'medium'
        elif base_risk == 'medium':
            return 'high'

    return base_risk

def analyze_change_impact(file_path: Path, search_path: Path) -> Dict:
    """分析变更影响"""
    # 获取Git diff
    try:
        result = subprocess.run(
            ['git', 'diff', 'HEAD', str(file_path)],
            capture_output=True,
            text=True,
            cwd=ROOT_DIR
        )
        git_diff = result.stdout
    except Exception as e:
        print(f"警告：无法获取Git diff: {e}")
        git_diff = ""

    # 识别变更类型
    change_type = identify_change_type(file_path, git_diff)

    # 查找引用该文件的其他文件
    referencing_files = find_referencing_files(file_path, search_path)

    # 提取当前文件的交叉引用
    cross_refs = extract_cross_references(file_path)

    # 评估风险等级
    risk_level = assess_risk_level(change_type, len(referencing_files))

    return {
        'file': file_path,
        'change_type': change_type,
        'referencing_files': referencing_files,
        'cross_refs': cross_refs,
        'risk_level': risk_level,
        'git_diff': git_diff
    }

def generate_report(analysis: Dict, output_file: Path = None):
    """生成影响分析报告"""
    file_path = analysis['file']
    change_type = CHANGE_TYPES.get(analysis['change_type'], analysis['change_type'])
    risk_level = RISK_LEVELS.get(analysis['risk_level'], analysis['risk_level'])

    report = f"""# 变更影响分析报告

**报告日期**：{Path(__file__).stat().st_mtime}
**变更文件**：`{file_path.relative_to(ROOT_DIR)}`
**变更类型**：{change_type}
**风险等级**：{risk_level}

## 变更内容

（需要手动填写变更内容描述）

## 影响范围分析

### 直接影响
- **当前文档**：`{file_path.relative_to(ROOT_DIR)}`

### 间接影响
"""

    if analysis['referencing_files']:
        report += "\n**引用该文档的文件**：\n"
        for ref_file in analysis['referencing_files']:
            report += f"- `{ref_file.relative_to(ROOT_DIR)}`\n"
    else:
        report += "\n**无其他文件引用此文档**\n"

    if analysis['cross_refs']:
        report += "\n**该文档引用的其他文件**：\n"
        for ref in set(analysis['cross_refs'][:10]):  # 限制显示数量
            report += f"- `{ref}`\n"
        if len(analysis['cross_refs']) > 10:
            report += f"- ... 还有 {len(analysis['cross_refs']) - 10} 个引用\n"

    report += f"""
## 风险评估

**风险等级**：{risk_level}

**潜在问题**：
- 需要验证变更的正确性
- 需要检查交叉引用的完整性
- 需要验证文档的一致性

**应对策略**：
- 执行变更前备份相关文档
- 执行变更后运行自动化检查
- 更新所有相关文档的交叉引用

## 验证检查清单

- [ ] 变更正确性验证
- [ ] 交叉引用完整性检查
- [ ] 文档一致性验证
- [ ] 逻辑正确性验证（如适用）

## 变更状态

- [ ] 待执行
- [ ] 执行中
- [ ] 已完成
- [ ] 已回滚
"""

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 报告已生成：{output_file}")
    else:
        print(report)

    return report

def main():
    parser = argparse.ArgumentParser(description='分析文档变更的影响范围')
    parser.add_argument('--file', type=str, help='要分析的文件路径')
    parser.add_argument('--path', type=str, default='Philosophy', help='要分析的目录路径（默认为Philosophy）')
    parser.add_argument('--report', action='store_true', help='生成影响分析报告')

    args = parser.parse_args()

    if args.file:
        file_path = ROOT_DIR / args.file
        if not file_path.exists():
            print(f"错误：文件不存在：{file_path}")
            return
        search_path = file_path.parent
    else:
        search_path = ROOT_DIR / args.path
        if not search_path.exists():
            print(f"错误：目录不存在：{search_path}")
            return
        # 如果没有指定文件，分析最近修改的文件
        print("提示：未指定文件，请使用 --file 参数指定要分析的文件")
        return

    # 分析变更影响
    print(f"分析变更影响：{file_path.relative_to(ROOT_DIR)}")
    analysis = analyze_change_impact(file_path, search_path)

    # 显示分析结果
    print(f"\n变更类型：{CHANGE_TYPES.get(analysis['change_type'], analysis['change_type'])}")
    print(f"风险等级：{RISK_LEVELS.get(analysis['risk_level'], analysis['risk_level'])}")
    print(f"引用该文档的文件数：{len(analysis['referencing_files'])}")
    print(f"该文档引用的文件数：{len(analysis['cross_refs'])}")

    if analysis['referencing_files']:
        print("\n引用该文档的文件：")
        for ref_file in analysis['referencing_files']:
            print(f"  - {ref_file.relative_to(ROOT_DIR)}")

    # 生成报告
    if args.report:
        report_file = file_path.parent / f"{file_path.stem}_change_impact_report.md"
        generate_report(analysis, report_file)

if __name__ == "__main__":
    main()
