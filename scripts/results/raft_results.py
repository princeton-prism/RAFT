#!/usr/bin/env python3
"""Print a table of RAFT evaluation results (stdlib only).

  raft_results.py <eval_root>

<eval_root> holds one dir per eval, named <axis>[_suffix] (scripts/eval/eval_axis.sh writes these). The success line is `Success count: X, Total count: Y` in .../running_log.log. The runner can
exit nonzero after writing it (temp-dir cleanup), so only this line counts, not the exit code.
"""
import argparse
import re
import sys
from pathlib import Path

LINE = re.compile(r"Success count: (\d+), Total count: (\d+)")
DIR = re.compile(r"^([a-z0-9]+?)(_rep\w*)?$")


def result(d: Path):
    hits = []
    for log in d.rglob("running_log.log"):
        hits += [(int(a), int(b)) for a, b in LINE.findall(log.read_text(errors="ignore"))]
    return hits[-1] if hits else None


def collect(root: Path):
    out = {}
    for d in sorted(root.iterdir()):
        m = DIR.match(d.name)
        if d.is_dir() and m:
            out[(m.group(1), m.group(2) or "")] = result(d)
    return out


def table(root: Path) -> str:
    rows = ["| axis | success | rate |", "|---|---|---|"]
    for (axis, suf), r in sorted(collect(root).items()):
        cell = f"{r[0]}/{r[1]}" if r else "not finished"
        rate = f"{100 * r[0] / r[1]:.2f}%" if r else "—"
        rows.append(f"| {axis}{suf} | {cell} | {rate} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    print(table(a.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
