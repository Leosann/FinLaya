# -*- coding: utf-8 -*-
"""检查各测试集与训练集之间是否存在 state 文本重叠（泄漏检查）。

输出每个测试集的：总行数、state 与训练集完全相同的行数、state 相同的比例。
"""
import hashlib
import json
import os
import sys

DATA = os.environ.get("LAYA_DATA", "/data/laya-fin/laya")


def states(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s = d.get("state") or {}
            # state 可能是 {"text": ...} 或纯字符串
            text = s.get("text") if isinstance(s, dict) else s
            out.append(hashlib.sha256((text or "").encode("utf-8")).hexdigest())
    return out


def main():
    # 训练集按来源拆开算：整体 + 逐子集
    train_files = {
        "cfsc_absa_train": os.path.join(DATA, "cfsc_absa_train.jsonl"),
        "finfe_train": os.path.join(DATA, "finfe_train.jsonl"),
        "finnsp_train": os.path.join(DATA, "finnsp_train.jsonl"),
    }
    subsets = {k: set(states(v)) for k, v in train_files.items()}
    all_train = set().union(*subsets.values())
    print("train unique states: total=%d  cfsc=%d finfe=%d finnsp=%d"
          % (len(all_train), len(subsets["cfsc_absa_train"]),
             len(subsets["finfe_train"]), len(subsets["finnsp_train"])))

    for name in ("finfe_eval", "finnsp_eval", "cfsc_absa_test"):
        p = os.path.join(DATA, name + ".jsonl")
        h = states(p)
        n = len(h)
        overlap = sum(1 for x in h if x in all_train)
        print("%-16s rows=%-6d 与训练集重叠=%-6d (%.1f%%)  | cfsc=%d finfe=%d finnsp=%d"
              % (name, n, overlap, 100.0 * overlap / n,
                 sum(1 for x in h if x in subsets["cfsc_absa_train"]),
                 sum(1 for x in h if x in subsets["finfe_train"]),
                 sum(1 for x in h if x in subsets["finnsp_train"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
