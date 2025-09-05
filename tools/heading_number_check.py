import os
import re
import sys
from typing import List, Tuple


HEADING_REGEX = re.compile(r"^(?P<hashes>#+)\s+(?P<num>(\d+)(?:\.(\d+))*)\b")


def find_markdown_files(root: str) -> List[str]:
    md_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".md"):
                md_files.append(os.path.join(dirpath, fn))
    return sorted(md_files)


def parse_headings(path: str) -> List[Tuple[int, str, int]]:
    """
    Return list of (level, number_string, line_no)
    """
    results: List[Tuple[int, str, int]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                m = HEADING_REGEX.match(line.rstrip())
                if m:
                    level = len(m.group("hashes"))
                    num = m.group("num")
                    results.append((level, num, i))
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
    return results


def is_consecutive(prev: str, curr: str) -> bool:
    """
    Check whether curr is a valid continuation after prev at the same level.
    Examples: 1 -> 2, 1.2 -> 1.3, 2.9 -> 2.10
    """
    try:
        a = [int(x) for x in prev.split(".")]
        b = [int(x) for x in curr.split(".")]
    except ValueError:
        return False

    if len(a) != len(b):
        return False

    # Must match prefix except last, and last increases by 1
    if a[:-1] != b[:-1]:
        return False
    return b[-1] == a[-1] + 1


def is_child(prev: str, curr: str) -> bool:
    # child if curr == prev + ".1" (first subsection)
    try:
        a = [int(x) for x in prev.split(".")]
        b = [int(x) for x in curr.split(".")]
    except ValueError:
        return False
    return len(b) == len(a) + 1 and b[:-1] == a and b[-1] == 1


def lint_file(path: str) -> List[str]:
    messages: List[str] = []
    headings = parse_headings(path)
    if not headings:
        return messages

    # Track a stack of last numbers per level
    last_at_level: dict[int, str] = {}

    for level, num, line_no in headings:
        if level == 1:
            # Top-level: either first or consecutive with previous h1
            last = last_at_level.get(1)
            if last is None:
                last_at_level[1] = num
            else:
                if not is_consecutive(last, num):
                    messages.append(f"{path}:{line_no}: H1 jump/inconsistency: {last} -> {num}")
                last_at_level[1] = num
            # reset deeper levels when a new H1 starts
            for k in list(last_at_level.keys()):
                if k > 1:
                    del last_at_level[k]
        else:
            parent_level = level - 1
            parent = last_at_level.get(parent_level)
            last_here = last_at_level.get(level)

            if parent is None:
                messages.append(f"{path}:{line_no}: H{level} without parent before: {num}")
            else:
                if last_here is None:
                    if not is_child(parent, num):
                        messages.append(
                            f"{path}:{line_no}: H{level} should start at {parent}.1 but got {num}"
                        )
                    last_at_level[level] = num
                else:
                    # Either consecutive at same level or a new child of current parent
                    if is_consecutive(last_here, num):
                        last_at_level[level] = num
                    elif is_child(parent, num):
                        # new subsection series restarting under same parent -> ok
                        last_at_level[level] = num
                    else:
                        messages.append(
                            f"{path}:{line_no}: H{level} numbering inconsistent at {last_here} -> {num} (parent {parent})"
                        )

            # prune deeper levels
            for k in list(last_at_level.keys()):
                if k > level:
                    del last_at_level[k]

    return messages


def main() -> int:
    root = os.path.join(os.path.dirname(__file__), "..", "docs")
    root = os.path.abspath(root)

    md_files = find_markdown_files(root)
    all_msgs: List[str] = []
    for path in md_files:
        msgs = lint_file(path)
        all_msgs.extend(msgs)

    if all_msgs:
        print("Heading numbering issues found:\n")
        for m in all_msgs:
            print(m)
        print(f"\nTotal issues: {len(all_msgs)}")
        return 1
    else:
        print("No heading numbering issues found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())


